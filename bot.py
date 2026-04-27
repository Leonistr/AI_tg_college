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
from logging.handlers import RotatingFileHandler
from pathlib import Path

import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from dialog_rules import (
    bot_meta_reply,
    direct_college_reply,
    direct_study_reply,
    extract_topic_hint,
    is_abusive_message,
    is_greeting_or_small_talk,
    is_non_college_math,
    is_smalltalk_message,
    looks_like_college_question,
    needs_recent_context,
    normalize_question,
    pick_recent_college_message,
)
from pipeline import (
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
REPEAT_REMINDER_VARIANTS = (
    "Я уже отвечал на это, давай коротко двинемся дальше по делу.",
    "Вижу повтор вопроса. Чтобы не ходить по кругу, уточни, что именно осталось непонятно.",
    "Повторяется та же тема. Давай точечно: что именно уточнить — вход, сроки или где смотреть?",
    "Я помню этот вопрос. Чтобы помочь лучше, добавь новую деталь, а не повтор дословно.",
)
SUMMARY_SYSTEM_PROMPT = """Сожми контекст диалога студент-ассистент в 2-3 коротких предложения.
Пиши только факты из диалога: что уже выяснили, что ещё нужно уточнить, чего делать нельзя.
Без списков, без нумерации, без воды."""
RUNTIME_OVERRIDES_FILE = "_runtime_overrides.json"


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


def _setup_file_logging(base: Path) -> Path:
    """
    Подключает запись логов в отдельный файл с ротацией.
    По умолчанию: logs/bot.log, 5MB, 5 бэкапов.
    """
    rel = os.environ.get("BOT_LOG_FILE", "logs/bot.log").strip() or "logs/bot.log"
    max_bytes_raw = os.environ.get("BOT_LOG_MAX_BYTES", "5242880").strip()
    backups_raw = os.environ.get("BOT_LOG_BACKUP_COUNT", "5").strip()
    try:
        max_bytes = max(1024, int(max_bytes_raw))
    except ValueError:
        max_bytes = 5 * 1024 * 1024
    try:
        backup_count = max(1, int(backups_raw))
    except ValueError:
        backup_count = 5

    log_path = (base / rel).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()

    # Не дублируем хендлер при повторной инициализации.
    for h in root.handlers:
        if isinstance(h, RotatingFileHandler) and Path(getattr(h, "baseFilename", "")).resolve() == log_path:
            return log_path

    file_handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root.addHandler(file_handler)
    return log_path


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


def _remember_user_message(state: dict, text: str, max_items: int = 4) -> None:
    history = state.get("recent_user_messages", [])
    if not isinstance(history, list):
        history = []
    history.append(text)
    if len(history) > max_items:
        history = history[-max_items:]
    state["recent_user_messages"] = history


def _remember_bot_message(state: dict, text: str, max_items: int = 4) -> None:
    history = state.get("recent_bot_messages", [])
    if not isinstance(history, list):
        history = []
    history.append(text)
    if len(history) > max_items:
        history = history[-max_items:]
    state["recent_bot_messages"] = history


def _get_recent_user_context(state: dict, limit: int = 3) -> list[str]:
    history = state.get("recent_user_messages", [])
    if not isinstance(history, list) or len(history) <= 1:
        return []
    # Последний элемент — текущее сообщение, берём только предыдущие.
    prev = [str(x).strip() for x in history[:-1] if str(x).strip()]
    if not prev:
        return []
    return prev[-limit:]


def _get_recent_bot_context(state: dict, limit: int = 3) -> list[str]:
    history = state.get("recent_bot_messages", [])
    if not isinstance(history, list):
        return []
    prev = [str(x).strip() for x in history if str(x).strip()]
    if not prev:
        return []
    return prev[-limit:]


def _dialog_context_snippet(state: dict) -> str:
    summary = str(state.get("dialog_summary", "")).strip()
    return summary


def _update_repeat_streak(state: dict, normalized_q: str) -> int:
    last_q = str(state.get("last_normalized_question", "")).strip()
    if normalized_q and normalized_q == last_q:
        streak = int(state.get("repeat_streak", 1)) + 1
    else:
        streak = 1
    state["repeat_streak"] = streak
    state["last_normalized_question"] = normalized_q
    return streak


def _repeat_reminder_prefix(state: dict) -> str:
    idx_raw = state.get("repeat_notice_idx", 0)
    idx = int(idx_raw) if isinstance(idx_raw, (int, str)) else 0
    idx = idx % len(REPEAT_REMINDER_VARIANTS)
    state["repeat_notice_idx"] = (idx + 1) % len(REPEAT_REMINDER_VARIANTS)
    return REPEAT_REMINDER_VARIANTS[idx]


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
            "recent_user_messages": [],
            "recent_bot_messages": [],
            "dialog_summary": "",
            "repeat_streak": 1,
            "last_normalized_question": "",
            "message_count": 0,
            "repeat_notice_idx": 0,
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


async def _maybe_refresh_dialog_summary(context: ContextTypes.DEFAULT_TYPE, state: dict) -> None:
    count = int(state.get("message_count", 0))
    if count <= 0 or count % 3 != 0:
        return
    client: httpx.AsyncClient | None = context.bot_data.get("http_client")
    base_url = str(context.bot_data.get("ollama_base_url", "")).strip()
    model = str(context.bot_data.get("ollama_model_explain", "mistral")).strip() or "mistral"
    if client is None or not base_url:
        return
    old_summary = str(state.get("dialog_summary", "")).strip() or "Нет."
    user_hist = _get_recent_user_context(state, limit=4)
    bot_hist = _get_recent_bot_context(state, limit=4)
    if not user_hist and not bot_hist:
        return
    user_block = "\n".join(f"- {x}" for x in user_hist) or "- Нет данных"
    bot_block = "\n".join(f"- {x}" for x in bot_hist) or "- Нет данных"
    prompt = (
        f"Предыдущее summary:\n{old_summary}\n\n"
        f"Последние сообщения пользователя:\n{user_block}\n\n"
        f"Последние ответы ассистента:\n{bot_block}\n\n"
        "Обнови summary."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    try:
        url = f"{base_url.rstrip('/')}/api/chat"
        r = await client.post(url, json=payload, timeout=45.0)
        r.raise_for_status()
        body = r.json()
        msg = (body.get("message") or {}).get("content", "")
        summary = str(msg).strip()
        if summary:
            state["dialog_summary"] = re.sub(r"\s{2,}", " ", summary)
    except Exception:
        logger.debug("Не удалось обновить dialog summary", exc_info=True)


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
    normalized_q = normalize_question(text)
    is_repeat_question = normalized_q and normalized_q in state.get("recent_questions", [])
    repeat_streak = _update_repeat_streak(state, normalized_q)
    state["message_count"] = int(state.get("message_count", 0)) + 1
    _remember_user_message(state, text, max_items=4)
    recent_user_context = _get_recent_user_context(state, limit=3)

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

    if is_greeting_or_small_talk(text):
        if is_repeat_question:
            prefix = _repeat_reminder_prefix(state)
            answer = f"{prefix} Привет. Если готов, задай один конкретный вопрос по колледжу."
        else:
            state["repeat_notice_idx"] = 0
            answer = "Привет! Спрашивай по колледжу или учёбе — отвечу коротко и по делу."
        state["fallback_stage"] = 0
        state["awaiting_clarification"] = False
        state["suggested_topic"] = ""
        state["last_user_question"] = text
        state["last_bot_answer"] = answer
        recent = state.get("recent_questions", [])
        if normalized_q:
            recent.append(normalized_q)
            if len(recent) > 8:
                recent = recent[-8:]
        state["recent_questions"] = recent
        _remember_bot_message(state, answer, max_items=4)
        await _maybe_refresh_dialog_summary(context, state)
        await update.message.reply_text(answer)
        return

    meta = bot_meta_reply(text)
    if meta:
        state["fallback_stage"] = 0
        state["awaiting_clarification"] = False
        state["last_user_question"] = text
        state["last_bot_answer"] = meta
        _remember_bot_message(state, meta, max_items=4)
        await _maybe_refresh_dialog_summary(context, state)
        await update.message.reply_text(meta)
        return

    if is_abusive_message(text):
        answer = "Давай без оскорблений. Если есть вопрос по колледжу или учёбе, помогу по делу."
        state["fallback_stage"] = 0
        state["awaiting_clarification"] = False
        state["last_user_question"] = text
        state["last_bot_answer"] = answer
        _remember_bot_message(state, answer, max_items=4)
        await _maybe_refresh_dialog_summary(context, state)
        await update.message.reply_text(answer)
        return

    if is_smalltalk_message(text):
        answer = "Могу поболтать коротко, но полезнее — задай вопрос по учёбе или колледжу."
        state["fallback_stage"] = 0
        state["awaiting_clarification"] = False
        state["last_user_question"] = text
        state["last_bot_answer"] = answer
        _remember_bot_message(state, answer, max_items=4)
        await _maybe_refresh_dialog_summary(context, state)
        await update.message.reply_text(answer)
        return

    direct = direct_college_reply(text, context.bot_data.get("knowledge", {}))
    if direct:
        if is_repeat_question:
            prefix = _repeat_reminder_prefix(state)
            if prefix.lower() not in direct.lower():
                direct = f"{prefix} {direct}"
        else:
            state["repeat_notice_idx"] = 0
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
        _remember_bot_message(state, direct, max_items=4)
        await _maybe_refresh_dialog_summary(context, state)
        await update.message.reply_text(direct)
        return

    direct_study = direct_study_reply(text)
    if direct_study:
        if is_repeat_question:
            prefix = _repeat_reminder_prefix(state)
            if prefix.lower() not in direct_study.lower():
                direct_study = f"{prefix} {direct_study}"
        else:
            state["repeat_notice_idx"] = 0
        state["fallback_stage"] = 0
        state["awaiting_clarification"] = False
        state["last_user_question"] = text
        state["last_bot_answer"] = direct_study
        _remember_bot_message(state, direct_study, max_items=4)
        await _maybe_refresh_dialog_summary(context, state)
        await update.message.reply_text(direct_study)
        return

    if is_non_college_math(text):
        answer = "Я помогаю по колледжу и учёбе. По математике вне контекста колледжа не считаю."
        state["fallback_stage"] = 0
        state["awaiting_clarification"] = False
        state["last_user_question"] = text
        state["last_bot_answer"] = answer
        _remember_bot_message(state, answer, max_items=4)
        await _maybe_refresh_dialog_summary(context, state)
        await update.message.reply_text(answer)
        return

    if repeat_streak >= 3 and len(text.split()) >= 4:
        last_answer = str(state.get("last_bot_answer", "")).strip()
        answer = (
            "Вижу, что мы застряли на одном вопросе. "
            "Давай точнее: что именно нужно — где зайти, какие данные для входа или к кому обратиться?"
        )
        if last_answer:
            answer = f"{answer} Коротко повторю: {last_answer}"
        state["last_user_question"] = text
        state["last_bot_answer"] = answer
        _remember_bot_message(state, answer, max_items=4)
        await _maybe_refresh_dialog_summary(context, state)
        await update.message.reply_text(answer)
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
    topic_hint = extract_topic_hint(text)
    query_for_pipeline = text
    dialog_summary = _dialog_context_snippet(state)
    if dialog_summary and (state.get("awaiting_clarification", False) or needs_recent_context(text)):
        query_for_pipeline = f"Контекст диалога: {dialog_summary}\nТекущий вопрос: {query_for_pipeline}"
    if needs_recent_context(text) and recent_user_context:
        anchor = pick_recent_college_message(recent_user_context)
        if anchor:
            query_for_pipeline = f"{anchor}. Уточнение студента: {text}."

    # Уточнение «туда войти» после ответа про SmartNation — в эмбеддинг явно добавит pipeline
    lb = str(state.get("last_bot_answer", "")).lower()
    if ("smart" in lb or "смарт" in lb) and re.search(
        r"\b(туда|туда\s+же|войти|зайти|вход|логин|парол)\w*\b", text.lower()
    ):
        query_for_pipeline = (
            f"{text}\n(контекст: только что речь шла о системе SmartNation — оценки и расписание; "
            f"нужен вход, логин или пароль по правилам колледжа.)"
        )

    # Короткие темы («расписание», «оценки»): не склеивать с нерелевантным прошлым вопросом.
    if topic_hint and len(text.split()) <= 4:
        prev_q = state.get("last_user_question", "")
        awaiting = bool(state.get("awaiting_clarification", False))
        suggested = str(state.get("suggested_topic", "")).strip().lower()
        topic_l = topic_hint.lower()
        merge_ok = (
            (awaiting and bool(prev_q) and looks_like_college_question(prev_q))
            or (suggested and (topic_l in suggested or suggested in topic_l))
            or (bool(prev_q) and looks_like_college_question(prev_q))
        )
        if merge_ok and prev_q:
            query_for_pipeline = f"{prev_q}. Уточнение студента: тема — {topic_hint}."
        else:
            query_for_pipeline = (
                f"Вопрос студента по теме колледжа: {topic_hint}. "
                f"Исходная фраза: «{text}». Ответь по материалам колледжа, кратко."
            )
        state["awaiting_clarification"] = False
    elif (
        state.get("awaiting_clarification", False)
        and len(text.split()) <= 5
        and looks_like_college_question(state.get("last_user_question", ""))
    ):
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
                "Проверьте OLLAMA_MODEL_UNIFIED и OLLAMA_EMBED_MODEL в .env."
            )
            logger.warning("ollama http error: %s", e.response.text[:500])
        except (httpx.ReadTimeout, TimeoutError):
            answer = (
                "Сервер моделей отвечает слишком долго. "
                "Попробуй повторить вопрос чуть короче или проверь, что Ollama не перегружен."
            )
            logger.warning("ollama timeout while answering")
        except Exception as e:
            logger.exception("answer error")
            details = str(e).strip() or e.__class__.__name__
            answer = f"Произошла ошибка: {details}"

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
        prefix = _repeat_reminder_prefix(state)
        if prefix.lower() not in answer.lower():
            answer = f"{prefix} {answer}"
    else:
        state["repeat_notice_idx"] = 0
    state["last_bot_answer"] = answer
    _remember_bot_message(state, answer, max_items=4)
    await _maybe_refresh_dialog_summary(context, state)
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
    log_path = _setup_file_logging(base)
    logger.info("Логи пишутся в файл: %s", log_path)
    knowledge_path = _resolve_knowledge_path(base)

    data = load_knowledge(knowledge_path)
    ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    # Совместимость: можно задать одну модель на всё (OLLAMA_MODEL_UNIFIED),
    # или раздельные модели на этапы пайплайна.
    unified_model = os.environ.get("OLLAMA_MODEL_UNIFIED", "mistral").strip() or "mistral"
    ollama_model_quick = os.environ.get("OLLAMA_MODEL_QUICK", unified_model).strip() or unified_model
    ollama_model_explain = (
        os.environ.get("OLLAMA_MODEL_EXPLAIN", unified_model).strip() or unified_model
    )
    ollama_model_complex = (
        os.environ.get("OLLAMA_MODEL_COMPLEX", unified_model).strip() or unified_model
    )
    ollama_review_model = os.environ.get("OLLAMA_MODEL_REVIEW", unified_model).strip() or unified_model
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
    # Больше контекстных блоков полезно для multi-intent вопросов, но не даём разрастись бесконечно.
    rag_ctx_blocks = max(1, min(8, rag_ctx_blocks))

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

