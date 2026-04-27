from __future__ import annotations

import re

from pipeline import COLLEGE_HINTS

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
SMALLTALK_PATTERNS = (
    "как дела",
    "как жизнь",
    "чем занимаешься",
    "скучно",
    "давай поговорим",
    "что делаешь",
)
ABUSE_PATTERNS = ("тупой", "дебил", "чмо", "мудак", "ты гей", "идиот")
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


def looks_like_college_question(text: str) -> bool:
    low = text.lower().strip()
    return any(h in low for h in COLLEGE_HINTS)


def is_bot_meta_question(text: str) -> bool:
    low = text.lower().strip()
    return any(p in low for p in BOT_META_PATTERNS)


def bot_meta_reply(text: str) -> str | None:
    low = text.lower()
    if not is_bot_meta_question(text):
        return None
    if "зовут" in low or "имя" in low:
        return (
            "Меня можно называть EKEB AI ассистент. Я отвечаю по вопросам колледжа "
            "строго по материалам базы знаний."
        )
    if "кто ты" in low or "кто вы" in low or "бот" in low or "ии" in low or "ai" in low:
        return "Я текстовый ассистент колледжа: подсказываю по учебным вопросам и правилам из базы знаний."
    return None


def extract_topic_hint(text: str) -> str | None:
    low = text.lower().strip()
    for part, topic in TOPIC_MAP.items():
        if part in low:
            return topic
    return None


def normalize_question(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^\w\sа-яё]", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_greeting_only(text: str) -> bool:
    norm = normalize_question(text)
    if not norm:
        return False
    return norm in GREETING_WORDS


def is_greeting_or_small_talk(text: str) -> bool:
    if looks_like_college_question(text):
        return False
    if is_greeting_only(text):
        return True
    t = text.strip()
    if not t or len(t.split()) > 14:
        return False
    low = t.lower()
    if re.search(r"\bпривет(ик|ики|очка)?\b", low) or (low.startswith("привет") and len(t.split()) <= 3):
        return True
    if re.search(r"\b(здорово|здрасьте|здравствуй|салам|хай)\b", low):
        return True
    if re.search(r"\bдобрый\s+(день|вечер|утро)\b", low) or re.search(r"\bдоброе\s+утро\b", low):
        return True
    if re.search(r"поздароваться|поздороваться|поздоровк", low):
        return True
    if re.search(r"\bхотел[аи]?\s+(просто\s+)?(поздороваться|поздароваться)\b", low):
        return True
    if re.search(r"\bпросто\s+(поздороваться|поздароваться|привет)\b", low):
        return True
    return False


def direct_college_reply(text: str, data: dict) -> str | None:
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
    if "сайт" in low and re.search(r"(какой|где|дай|подскажи|официаль|ссылк|адрес)\w*", low):
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


def direct_study_reply(text: str) -> str | None:
    low = text.lower().strip()
    if not low:
        return None
    if re.search(r"(исправ|измен|поднят|повыс)\w*\s+оцен", low):
        return "По материалам колледжа: по изменению оценки лучше сразу подойти к преподавателю."
    if re.search(r"(где|как).{0,30}(посмотр|узнат|провер).{0,20}оцен", low) or (
        "оцен" in low and "smart" in low
    ):
        return (
            "Оценки смотри в SmartNation: логин — ИИН, пароль — последние 6 цифр ИИН + abc. "
            "Если не пускает, напиши преподавателю или в администрацию."
        )
    if ("не понял" in low or "не понимаю" in low) and (
        "тема" in low or "информат" in low or "програм" in low
    ):
        return (
            "Ок, разберём. Уточни, это теория или практика, и какая тема: "
            "основы ПК, файлы, базы данных или программирование."
        )
    return None


def is_non_college_math(text: str) -> bool:
    low = text.lower().strip()
    if not low or looks_like_college_question(low):
        return False
    if re.fullmatch(r"[\d\s\+\-\*/\(\)=\.,]+", low):
        return True
    if re.search(r"\b\d+\s*[\+\-\*/]\s*\d+\b", low):
        return True
    return False


def is_smalltalk_message(text: str) -> bool:
    low = text.lower().strip()
    if not low or looks_like_college_question(low):
        return False
    return any(p in low for p in SMALLTALK_PATTERNS)


def is_abusive_message(text: str) -> bool:
    low = text.lower().strip()
    return any(p in low for p in ABUSE_PATTERNS)


def needs_recent_context(text: str) -> bool:
    low = text.lower().strip()
    if not low or looks_like_college_question(low):
        return False
    if len(low.split()) > 8:
        return False
    return bool(
        re.search(r"\b(это|этот|эта|эти|там|туда|так|тогда|он|она|они|его|ее|её|их)\b", low)
        or re.search(r"\b(а как|а где|и как|и где)\b", low)
    )


def pick_recent_college_message(messages: list[str]) -> str | None:
    for msg in reversed(messages):
        s = str(msg).strip()
        if s and looks_like_college_question(s):
            return s
    return None
