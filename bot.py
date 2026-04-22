"""
Telegram-ассистент для студентов: Ollama + семантический поиск по Markdown-базе знаний.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import re
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from pipeline import (
    COLLEGE_HINTS,
    FALLBACK_CLARIFY,
    FALLBACK_FILTER,
    FALLBACK_NO_MATCH,
    answer_with_rag,
)
from rag import RAGIndex, get_chunk_builders, load_or_build_index

load_dotenv()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
TOPIC_MAP = {
    "оцен": "оценки",
    "распис": "расписание",
    "пересда": "пересдача",
    "пропуск": "пропуски",
    "поступ": "поступление",
    "специаль": "специальности",
    "практик": "практика",
    "допуск": "допуск",
    "справк": "справки",
    "колледж": "колледж",
}
GREETING_WORDS = {"привет", "здравствуйте", "салам", "добрый день", "добрый вечер", "хай"}
RUNTIME_OVERRIDES_FILE = "_runtime_overrides.json"
BOT_META_PATTERNS = (
    "как тебя зовут",
    "как вас зовут",
    "кто ты",
    "кто вы",
    "ты бот",
    "вы бот",
    "ты ии",
    "ты ai",
    "какое у тебя имя",
    "какое ваше имя",
)


def _looks_like_college_question(text: str) -> bool:
    low = text.lower().strip()
    return any(h in low for h in COLLEGE_HINTS)


def _is_bot_meta_question(text: str) -> bool:
    low = text.lower().strip()
    return any(p in low for p in BOT_META_PATTERNS)


def _bot_meta_reply(text: str) -> str | None:
    low = text.lower()
    if not _is_bot_meta_question(text):
        return None
    if "зовут" in low or "имя" in low:
        return (
            "Меня можно называть EKEB AI ассистент. Я отвечаю по вопросам колледжа "
            "строго по материалам базы знаний."
        )
    if "кто ты" in low or "кто вы" in low or "бот" in low or "ии" in low or "ai" in low:
        return "Я текстовый ассистент колледжа: подсказываю по учебным вопросам и правилам из базы знаний."
    return None


def _ensure_event_loop() -> None:
    """Python 3.12+ не создаёт цикл в MainThread; PTB run_polling() ожидает установленный loop."""
    # Python 3.14+ помечает set_event_loop_policy как deprecated.
    # Для 3.13 и ниже оставляем WindowsSelector policy для совместимости PTB.
    if sys.platform == "win32" and sys.version_info < (3, 14):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def load_knowledge(path: Path) -> dict:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Папка знаний не найдена: {path}. Ожидается каталог data/college_ai_knowledge."
        )
    return _load_markdown_knowledge(path)


def save_knowledge(path: Path, data: dict) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Папка знаний не найдена: {path}. Сохранение поддерживается только в data/college_ai_knowledge."
        )
    runtime_path = path / RUNTIME_OVERRIDES_FILE
    payload = {
        "knowledge": data.get("knowledge", {}),
        "examples": data.get("examples", []),
        "college": data.get("college", {}),
        "specialties": data.get("specialties", []),
    }
    with runtime_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _clean_markdown_text(text: str) -> str:
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "---":
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[\-\*\u2022]\s*", "", line)
        out.append(line)
    return "\n".join(out).strip()


def _parse_faq_examples(text: str) -> list[dict]:
    examples: list[dict] = []
    q: str | None = None
    a_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line == "---":
            continue
        if line.startswith("В:"):
            if q and a_lines:
                a = " ".join(a_lines).strip()
                if q and a:
                    examples.append({"q": q, "a": a})
            q = line[2:].strip()
            a_lines = []
            continue
        if line.startswith("О:"):
            a_lines = [line[2:].strip()]
            continue
        if q is not None and a_lines:
            a_lines.append(line)
    if q and a_lines:
        a = " ".join(a_lines).strip()
        if q and a:
            examples.append({"q": q, "a": a})
    return examples


def _extract_college_fields_from_about(text: str) -> dict:
    def pick(pattern: str) -> str:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else ""

    name = pick(r"^\s*Название:\s*(.+)$")
    short = pick(r"^\s*Сокращ[её]нн[оео]\s+название:\s*(.+)$")
    address = pick(r"^\s*Адрес:\s*(.+)$")
    website = pick(r"^\s*Сайт:\s*(.+)$")
    phone = pick(r"^\s*Телефон:\s*(.+)$")
    email = pick(r"^\s*Email:\s*(.+)$")
    city_line = pick(r"^\s*Город:\s*(.+)$")
    if not address and city_line:
        address = city_line
    result = {}
    if city_line:
        result["city"] = city_line
    if name:
        result["name"] = name
    if short:
        result["short_name"] = short
    if address:
        result["address"] = address
    if website:
        result["website"] = website
    if phone:
        result["phone"] = phone
    if email:
        result["email"] = email
    return result


def _merge_runtime_overrides(data: dict, root: Path) -> dict:
    runtime_path = root / RUNTIME_OVERRIDES_FILE
    if not runtime_path.is_file():
        return data
    try:
        with runtime_path.open(encoding="utf-8") as f:
            runtime = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning("Не удалось прочитать runtime overrides: %s", runtime_path)
        return data

    knowledge = runtime.get("knowledge")
    if isinstance(knowledge, dict):
        data.setdefault("knowledge", {}).update(knowledge)

    college = runtime.get("college")
    if isinstance(college, dict):
        data.setdefault("college", {}).update(college)

    specialties = runtime.get("specialties")
    if isinstance(specialties, list):
        data["specialties"] = specialties

    examples = runtime.get("examples")
    if isinstance(examples, list):
        base = data.setdefault("examples", [])
        seen = {(str(x.get("q", "")).strip(), str(x.get("a", "")).strip()) for x in base if isinstance(x, dict)}
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            q = str(ex.get("q", "")).strip()
            a = str(ex.get("a", "")).strip()
            if not q or not a:
                continue
            pair = (q, a)
            if pair in seen:
                continue
            base.append({"q": q, "a": a})
            seen.add(pair)
    return data


def _load_markdown_knowledge(root: Path) -> dict:
    data: dict = {"college": {}, "specialties": [], "knowledge": {}, "rules": [], "examples": []}
    md_files = sorted(root.rglob("*.md"))
    for path in md_files:
        raw = path.read_text(encoding="utf-8")
        plain = _clean_markdown_text(raw)
        rel = path.relative_to(root).with_suffix("").as_posix().replace("/", "__")
        if plain:
            data["knowledge"][f"md_{rel}"] = plain

        if path.name.lower() == "about.md":
            data["college"].update(_extract_college_fields_from_about(raw))
        if path.name.lower() == "faq.md":
            faq_examples = _parse_faq_examples(raw)
            if faq_examples:
                data["examples"].extend(faq_examples)
                for i, ex in enumerate(faq_examples, start=1):
                    data["knowledge"][f"faq_{i}"] = f"Вопрос: {ex['q']}. Ответ: {ex['a']}"
    return _merge_runtime_overrides(data, root)


def _resolve_knowledge_path(base: Path) -> Path:
    md_default = base / "data" / "college_ai_knowledge"
    if not md_default.is_dir():
        raise SystemExit(
            f"Не найдена папка знаний: {md_default}. "
            "Создайте data/college_ai_knowledge и добавьте markdown-файлы."
        )
    return md_default


def _get_user_id(update: Update) -> int | None:
    if update.effective_user is None:
        return None
    return update.effective_user.id


def _is_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    admin_id = context.bot_data.get("admin_telegram_id")
    user_id = _get_user_id(update)
    return admin_id is not None and user_id == admin_id


def _next_admin_note_key(knowledge: dict) -> str:
    now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"admin_note_{now}"
    key = base
    suffix = 1
    while key in knowledge:
        suffix += 1
        key = f"{base}_{suffix}"
    return key


def _extract_topic_hint(text: str) -> str | None:
    low = text.lower().strip()
    for part, topic in TOPIC_MAP.items():
        if part in low:
            return topic
    return None


def _normalize_question(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^\w\sа-яё]", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_greeting_only(text: str) -> bool:
    norm = _normalize_question(text)
    if not norm:
        return False
    return norm in GREETING_WORDS


def _direct_college_reply(text: str, data: dict) -> str | None:
    low = text.lower()
    college = data.get("college")
    if not isinstance(college, dict):
        return None

    address = str(college.get("address", "")).strip()
    city = str(college.get("city", "")).strip()
    website = str(college.get("website", "")).strip()
    hours = str(college.get("working_hours", "")).strip()
    director = str(college.get("director", "")).strip()
    name = str(college.get("name", "")).strip()

    if ("город" in low) or ("в каком городе" in low) or ("какой город" in low):
        if city:
            return f"Колледж находится: {city}."
        if address:
            return f"В базе указан адрес: {address}. Точный город лучше уточнить у администрации."
        return "Город сейчас не указан в материалах, уточните у администрации."

    if ("адрес" in low) or ("где находится" in low) or ("где колледж" in low):
        if address:
            return f"Адрес колледжа: {address}."
        return "Точный адрес лучше уточнить у администрации колледжа."
    if "сайт" in low:
        if website:
            return f"Официальный сайт колледжа: {website}."
        return "Сайт сейчас не указан, уточните у администрации."
    if ("режим" in low) or ("время работы" in low) or ("часы работы" in low):
        if hours:
            return f"Колледж работает: {hours}."
        return "График работы лучше уточнить у администрации."
    if ("директор" in low) or ("руководител" in low):
        if director:
            return f"Директор колледжа: {director}."
        return "Информацию о директоре лучше уточнить у администрации."
    if ("как называется колледж" in low) or ("название колледжа" in low):
        if name:
            return f"Название колледжа: {name}."
    return None


def _get_dialog_state(context: ContextTypes.DEFAULT_TYPE, user_id: int | None) -> dict:
    states = context.bot_data.setdefault("dialog_states", {})
    if user_id is None:
        return {}
    state = states.get(user_id)
    if state is None:
        state = {
            "awaiting_clarification": False,
            "fallback_stage": 0,
            "last_user_question": "",
            "last_bot_answer": "",
            "suggested_topic": "",
            "recent_questions": [],
        }
        states[user_id] = state
    return state


def _extract_qa_examples(data: dict) -> list[dict]:
    raw = data.get("examples")
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        q = str(item.get("q", "")).strip()
        a = str(item.get("a", "")).strip()
        if q and a:
            out.append({"q": q, "a": a})
    return out


def _parse_qa_examples_from_text(text: str) -> list[dict]:
    """
    Парсит блоки вида:
    1. Вопрос: ...
       Ответ:
       ...
    """
    pattern = re.compile(
        r"(?:^|\n)\s*(?:\d+\.\s*)?Вопрос:\s*(?P<q>.+?)\s*\n\s*Ответ:\s*(?P<a>.+?)(?=\n\s*(?:---|\d+\.\s*Вопрос:)|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    out: list[dict] = []
    for m in pattern.finditer(text):
        q = " ".join(m.group("q").split()).strip()
        a = re.sub(r"\n{2,}", "\n", m.group("a")).strip()
        if q and a:
            out.append({"q": q, "a": a})
    return out


def _migrate_examples_from_knowledge_notes(data: dict) -> tuple[int, int]:
    """
    Ищет в knowledge ключи admin_note_* и переносит Q/A в data['examples'].
    Возвращает (добавлено, найдено_в_заметках).
    """
    knowledge = data.get("knowledge")
    if not isinstance(knowledge, dict):
        return 0, 0

    examples = data.setdefault("examples", [])
    if not isinstance(examples, list):
        return 0, 0

    existing = {(str(x.get("q", "")).strip(), str(x.get("a", "")).strip()) for x in examples if isinstance(x, dict)}
    found_total = 0
    added = 0

    for key, value in knowledge.items():
        if not str(key).startswith("admin_note_"):
            continue
        if not isinstance(value, str) or "Вопрос:" not in value or "Ответ:" not in value:
            continue
        parsed = _parse_qa_examples_from_text(value)
        found_total += len(parsed)
        for ex in parsed:
            pair = (ex["q"], ex["a"])
            if pair in existing:
                continue
            examples.append(ex)
            existing.add(pair)
            added += 1
    return added, found_total


async def post_shutdown(application: Application) -> None:
    client: httpx.AsyncClient | None = application.bot_data.get("http_client")
    if client is not None:
        await client.aclose()


async def _build_rag(application: Application) -> RAGIndex:
    path: Path = application.bot_data["knowledge_path"]
    data: dict = application.bot_data["knowledge"]
    client: httpx.AsyncClient = application.bot_data["http_client"]
    base = application.bot_data["ollama_base_url"]
    embed_model = application.bot_data["ollama_embed_model"]
    builders = get_chunk_builders()
    return await load_or_build_index(client, base, embed_model, path, data, builders=builders)


async def _persist_and_rebuild(context: ContextTypes.DEFAULT_TYPE) -> None:
    path = Path(context.bot_data["knowledge_path"])
    data = context.bot_data["knowledge"]
    save_knowledge(path, data)
    context.bot_data["rag_index"] = await _build_rag(context.application)


async def post_init(application: Application) -> None:
    try:
        application.bot_data["rag_index"] = await _build_rag(application)
    except Exception:
        logger.exception("Не удалось построить RAG при старте")
        application.bot_data["rag_index"] = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я помогу по вопросам о колледже. Напиши свой вопрос текстом.\n"
        "Команда /reload — перечитать папку знаний с диска и обновить поиск.\n"
        "Для администратора: /myid, /learn <текст>, /learnqa вопрос | ответ, "
        "/migrate_examples, /cleanup_notes, /learnmode_on, /learnmode_off."
    )


async def myid(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = _get_user_id(update)
    if user_id is None:
        return
    await update.message.reply_text(f"Ваш Telegram ID: {user_id}")


async def reload_knowledge(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    path = Path(context.bot_data["knowledge_path"])
    try:
        data = load_knowledge(path)
        context.bot_data["knowledge"] = data
        context.bot_data["qa_examples"] = _extract_qa_examples(data)
        context.bot_data["rag_index"] = await _build_rag(context.application)
        await update.message.reply_text("Папка знаний обновлена, поиск и кэш эмбеддингов пересобраны.")
    except Exception as e:
        logger.exception("reload failed")
        await update.message.reply_text(f"Не удалось загрузить: {e}")


async def learn(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_admin(update, context):
        await update.message.reply_text("Эта команда доступна только администратору.")
        return
    if not update.message:
        return

    text = " ".join(context.args).strip()
    if not text:
        await update.message.reply_text("Использование: /learn <новая информация о колледже>")
        return

    data: dict = context.bot_data["knowledge"]
    knowledge = data.setdefault("knowledge", {})
    if not isinstance(knowledge, dict):
        await update.message.reply_text("Раздел knowledge в файле имеет неверный формат.")
        return

    key = _next_admin_note_key(knowledge)
    knowledge[key] = text
    try:
        await _persist_and_rebuild(context)
    except Exception as e:
        logger.exception("learn failed")
        await update.message.reply_text(f"Не удалось сохранить: {e}")
        return
    await update.message.reply_text(f"Сохранил новую информацию: {key}")


async def learnmode_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_admin(update, context):
        await update.message.reply_text("Эта команда доступна только администратору.")
        return
    context.bot_data["admin_learn_mode"] = True
    await update.message.reply_text(
        "Режим обучения включён. Каждое ваше текстовое сообщение будет сохраняться в базу."
    )


async def learnmode_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_admin(update, context):
        await update.message.reply_text("Эта команда доступна только администратору.")
        return
    context.bot_data["admin_learn_mode"] = False
    await update.message.reply_text("Режим обучения выключен.")


async def learnqa(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_admin(update, context):
        await update.message.reply_text("Эта команда доступна только администратору.")
        return
    if not update.message:
        return
    raw = " ".join(context.args).strip()
    if "|" not in raw:
        await update.message.reply_text("Использование: /learnqa вопрос | ответ")
        return
    q, a = raw.split("|", 1)
    q = q.strip()
    a = a.strip()
    if not q or not a:
        await update.message.reply_text("Нужны и вопрос, и ответ. Формат: /learnqa вопрос | ответ")
        return

    data: dict = context.bot_data["knowledge"]
    examples = data.setdefault("examples", [])
    if not isinstance(examples, list):
        await update.message.reply_text("Раздел examples в файле имеет неверный формат.")
        return
    examples.append({"q": q, "a": a})
    try:
        await _persist_and_rebuild(context)
    except Exception as e:
        logger.exception("learnqa failed")
        await update.message.reply_text(f"Не удалось сохранить пример: {e}")
        return
    context.bot_data["qa_examples"] = _extract_qa_examples(data)
    await update.message.reply_text("Пример вопрос-ответ сохранён.")


async def migrate_examples(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_admin(update, context):
        await update.message.reply_text("Эта команда доступна только администратору.")
        return
    data: dict = context.bot_data["knowledge"]
    added, found_total = _migrate_examples_from_knowledge_notes(data)
    if found_total == 0:
        await update.message.reply_text("В admin_note_* не найдено блоков формата «Вопрос/Ответ».")
        return
    try:
        await _persist_and_rebuild(context)
    except Exception as e:
        logger.exception("migrate examples failed")
        await update.message.reply_text(f"Не удалось сохранить миграцию: {e}")
        return
    context.bot_data["qa_examples"] = _extract_qa_examples(data)
    await update.message.reply_text(
        f"Готово. Найдено примеров: {found_total}, добавлено новых: {added}."
    )


async def cleanup_notes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_admin(update, context):
        await update.message.reply_text("Эта команда доступна только администратору.")
        return

    data: dict = context.bot_data["knowledge"]
    knowledge = data.get("knowledge")
    if not isinstance(knowledge, dict):
        await update.message.reply_text("Раздел knowledge в файле имеет неверный формат.")
        return

    to_delete = [k for k in knowledge.keys() if str(k).startswith("admin_note_")]
    if not to_delete:
        await update.message.reply_text("Служебные заметки admin_note_* не найдены.")
        return

    for key in to_delete:
        del knowledge[key]

    try:
        await _persist_and_rebuild(context)
    except Exception as e:
        logger.exception("cleanup notes failed")
        await update.message.reply_text(f"Не удалось удалить заметки: {e}")
        return
    await update.message.reply_text(f"Удалено служебных заметок: {len(to_delete)}.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()
    if not text:
        return
    user_id = _get_user_id(update)
    state = _get_dialog_state(context, user_id)
    normalized_q = _normalize_question(text)
    is_repeat_question = normalized_q and normalized_q in state.get("recent_questions", [])

    if _is_admin(update, context) and context.bot_data.get("admin_learn_mode", False):
        data: dict = context.bot_data["knowledge"]
        knowledge = data.setdefault("knowledge", {})
        if not isinstance(knowledge, dict):
            await update.message.reply_text("Раздел knowledge в файле имеет неверный формат.")
            return
        key = _next_admin_note_key(knowledge)
        knowledge[key] = text
        try:
            await _persist_and_rebuild(context)
            await update.message.reply_text(f"Запомнил и сохранил: {key}")
        except Exception as e:
            logger.exception("admin learn mode save failed")
            await update.message.reply_text(f"Не удалось сохранить: {e}")
        return

    if _is_greeting_only(text):
        answer = "Привет! Напишите ваш вопрос по колледжу, и я коротко подскажу."
        state["last_user_question"] = text
        state["last_bot_answer"] = answer
        await update.message.reply_text(answer)
        return

    meta = _bot_meta_reply(text)
    if meta:
        state["fallback_stage"] = 0
        state["awaiting_clarification"] = False
        state["last_user_question"] = text
        state["last_bot_answer"] = meta
        await update.message.reply_text(meta)
        return

    direct = _direct_college_reply(text, context.bot_data.get("knowledge", {}))
    if direct:
        if is_repeat_question:
            direct = f"Вы уже задавали этот вопрос ранее. {direct}"
        state["fallback_stage"] = 0
        state["awaiting_clarification"] = False
        state["last_user_question"] = text
        state["last_bot_answer"] = direct
        recent = state.get("recent_questions", [])
        if normalized_q:
            recent.append(normalized_q)
            if len(recent) > 8:
                recent = recent[-8:]
        state["recent_questions"] = recent
        await update.message.reply_text(direct)
        return

    base_url = context.bot_data["ollama_base_url"]
    quick_model = context.bot_data["ollama_model_quick"]
    explain_model = context.bot_data["ollama_model_explain"]
    complex_model = context.bot_data["ollama_model_complex"]
    review_model = context.bot_data["ollama_model_review"]
    embed_model = context.bot_data["ollama_embed_model"]
    client: httpx.AsyncClient = context.bot_data["http_client"]
    rag: RAGIndex | None = context.bot_data.get("rag_index")
    top_k = context.bot_data["rag_top_k"]
    min_sim = context.bot_data["rag_min_similarity"]
    min_conf = context.bot_data["rag_min_confidence"]
    max_ctx = context.bot_data["rag_max_context_blocks"]
    qa_examples = context.bot_data.get("qa_examples", [])
    topic_hint = _extract_topic_hint(text)
    query_for_pipeline = text

    # Короткие темы («расписание», «оценки»): не склеивать с нерелевантным прошлым вопросом.
    if topic_hint and len(text.split()) <= 4:
        prev_q = state.get("last_user_question", "")
        awaiting = bool(state.get("awaiting_clarification", False))
        suggested = str(state.get("suggested_topic", "")).strip().lower()
        topic_l = topic_hint.lower()
        merge_ok = awaiting or (
            suggested and (topic_l in suggested or suggested in topic_l)
        ) or (bool(prev_q) and _looks_like_college_question(prev_q))
        if merge_ok and prev_q:
            query_for_pipeline = f"{prev_q}. Уточнение студента: тема — {topic_hint}."
        else:
            query_for_pipeline = (
                f"Вопрос студента по теме колледжа: {topic_hint}. "
                f"Исходная фраза: «{text}». Ответь по материалам колледжа, кратко."
            )
        state["awaiting_clarification"] = False
    elif state.get("awaiting_clarification", False) and len(text.split()) <= 5:
        prev_q = state.get("last_user_question", "")
        query_for_pipeline = f"{prev_q}. Уточнение студента: {text}."
        state["awaiting_clarification"] = False

    await update.message.chat.send_action(action="typing")
    if rag is None:
        answer = (
            "Поиск по материалам колледжа сейчас недоступен. Проверьте логи сервера и подключение к Ollama."
        )
    else:
        try:
            answer = await answer_with_rag(
                client,
                base_url,
                quick_model,
                explain_model,
                complex_model,
                review_model,
                embed_model,
                rag,
                query_for_pipeline,
                top_k=top_k,
                min_similarity=min_sim,
                min_confidence=min_conf,
                max_context_blocks=max_ctx,
                qa_examples=qa_examples,
            )
        except httpx.ConnectError:
            answer = (
                "Не удаётся подключиться к Ollama. Запустите Ollama на этом компьютере "
                "и проверьте OLLAMA_BASE_URL в .env."
            )
        except httpx.HTTPStatusError as e:
            answer = (
                f"Ошибка Ollama (HTTP {e.response.status_code}). "
                "Проверьте модели в OLLAMA_MODEL_QUICK / OLLAMA_MODEL_EXPLAIN / "
                "OLLAMA_MODEL_COMPLEX / OLLAMA_REVIEW_MODEL и OLLAMA_EMBED_MODEL."
            )
            logger.warning("ollama http error: %s", e.response.text[:500])
        except Exception as e:
            logger.exception("answer error")
            answer = f"Произошла ошибка: {e}"

    # Многоуровневый fallback, чтобы не зацикливаться на одной фразе
    if answer == FALLBACK_CLARIFY:
        stage = int(state.get("fallback_stage", 0))
        if stage >= 1:
            guess = topic_hint or state.get("suggested_topic") or "оценки"
            answer = f"Вы имеете в виду {guess}? Могу подсказать по этой теме."
            state["fallback_stage"] = 2
            state["awaiting_clarification"] = True
            state["suggested_topic"] = guess
        else:
            state["fallback_stage"] = 1
            state["awaiting_clarification"] = True
            state["last_user_question"] = text
            state["suggested_topic"] = topic_hint or ""
    elif answer in {FALLBACK_FILTER, FALLBACK_NO_MATCH}:
        stage = int(state.get("fallback_stage", 0))
        if stage >= 2:
            answer = (
                "Пока не могу точно понять вопрос. Напишите, пожалуйста, чуть конкретнее, "
                "например: «Где посмотреть оценки?» или «Как узнать расписание?»"
            )
            state["fallback_stage"] = 3
        else:
            state["fallback_stage"] = stage + 1
    else:
        state["fallback_stage"] = 0
        state["awaiting_clarification"] = False
        if topic_hint:
            state["suggested_topic"] = topic_hint

    state["last_user_question"] = text
    if is_repeat_question:
        reminder = "Вы уже задавали этот вопрос ранее."
        if reminder.lower() not in answer.lower():
            answer = f"{reminder} {answer}"
    state["last_bot_answer"] = answer
    recent = state.get("recent_questions", [])
    if normalized_q:
        recent.append(normalized_q)
        if len(recent) > 8:
            recent = recent[-8:]
    state["recent_questions"] = recent

    max_len = 4000
    if len(answer) <= max_len:
        await update.message.reply_text(answer)
        return

    await update.message.reply_text(answer[:max_len] + "…")
    rest = answer[max_len:]
    while rest:
        chunk = rest[:max_len]
        rest = rest[max_len:]
        await update.message.reply_text(chunk + ("…" if rest else ""))


def main() -> None:
    _ensure_event_loop()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Задайте TELEGRAM_BOT_TOKEN в .env (см. .env.example)")

    base = Path(__file__).resolve().parent
    knowledge_path = _resolve_knowledge_path(base)

    data = load_knowledge(knowledge_path)
    ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    # Базовая совместимость: если задан только OLLAMA_MODEL, используем его как дефолт.
    legacy_model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    ollama_model_quick = os.environ.get("OLLAMA_MODEL_QUICK", "phi3").strip() or "phi3"
    ollama_model_explain = os.environ.get("OLLAMA_MODEL_EXPLAIN", "mistral").strip() or "mistral"
    ollama_model_complex = os.environ.get("OLLAMA_MODEL_COMPLEX", "llama3").strip() or "llama3"
    ollama_review_model = os.environ.get("OLLAMA_REVIEW_MODEL", legacy_model).strip() or legacy_model
    ollama_embed = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    try:
        rag_top_k = max(1, int(os.environ.get("RAG_TOP_K", "4")))
    except ValueError:
        rag_top_k = 4
    try:
        rag_min_sim = float(os.environ.get("RAG_MIN_SIMILARITY", "0.32"))
    except ValueError:
        rag_min_sim = 0.32
    try:
        rag_min_conf = float(os.environ.get("RAG_MIN_CONFIDENCE", "0.38"))
    except ValueError:
        rag_min_conf = 0.38
    try:
        rag_ctx_blocks = int(os.environ.get("RAG_MAX_CONTEXT_BLOCKS", "2"))
    except ValueError:
        rag_ctx_blocks = 2
    rag_ctx_blocks = max(1, min(2, rag_ctx_blocks))

    admin_id_raw = os.environ.get("ADMIN_TELEGRAM_ID", "").strip()
    admin_id: int | None = None
    if admin_id_raw:
        try:
            admin_id = int(admin_id_raw)
        except ValueError as e:
            raise SystemExit(
                "ADMIN_TELEGRAM_ID должен быть числом (ваш Telegram User ID)."
            ) from e

    application = (
        Application.builder()
        .token(token)
        .connect_timeout(30.0)
        .read_timeout(60.0)
        .write_timeout(30.0)
        .pool_timeout(30.0)
        .get_updates_connect_timeout(30.0)
        .get_updates_read_timeout(90.0)
        .get_updates_write_timeout(30.0)
        .get_updates_pool_timeout(30.0)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    application.bot_data["knowledge_path"] = knowledge_path
    application.bot_data["knowledge"] = data
    application.bot_data["ollama_base_url"] = ollama_base
    application.bot_data["ollama_model_quick"] = ollama_model_quick
    application.bot_data["ollama_model_explain"] = ollama_model_explain
    application.bot_data["ollama_model_complex"] = ollama_model_complex
    application.bot_data["ollama_model_review"] = ollama_review_model
    application.bot_data["ollama_embed_model"] = ollama_embed
    application.bot_data["rag_top_k"] = rag_top_k
    application.bot_data["rag_min_similarity"] = rag_min_sim
    application.bot_data["rag_min_confidence"] = rag_min_conf
    application.bot_data["rag_max_context_blocks"] = rag_ctx_blocks
    application.bot_data["http_client"] = httpx.AsyncClient()
    application.bot_data["admin_telegram_id"] = admin_id
    application.bot_data["admin_learn_mode"] = False
    application.bot_data["qa_examples"] = _extract_qa_examples(data)
    application.bot_data["dialog_states"] = {}

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("myid", myid))
    application.add_handler(CommandHandler("reload", reload_knowledge))
    application.add_handler(CommandHandler("learn", learn))
    application.add_handler(CommandHandler("learnqa", learnqa))
    application.add_handler(CommandHandler("migrate_examples", migrate_examples))
    application.add_handler(CommandHandler("cleanup_notes", cleanup_notes))
    application.add_handler(CommandHandler("learnmode_on", learnmode_on))
    application.add_handler(CommandHandler("learnmode_off", learnmode_off))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info(
        "Бот запущен. quick=%s, explain=%s, complex=%s, review=%s, эмбеддинги=%s, top_k=%s, min_sim=%s, min_conf=%s, ctx_blocks=%s, admin_id=%s",
        ollama_model_quick,
        ollama_model_explain,
        ollama_model_complex,
        ollama_review_model,
        ollama_embed,
        rag_top_k,
        rag_min_sim,
        rag_min_conf,
        rag_ctx_blocks,
        admin_id,
    )
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

