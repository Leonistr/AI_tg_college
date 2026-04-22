# Student assistant bot (EKEB)

Telegram-бот: ответы по базе знаний колледжа (`data/college_ai_knowledge`), RAG через эмбеддинги Ollama, генерация ответов несколькими моделями + проверка второй моделью.

## Требования

- Python **3.11+** (на Windows — см. `bot.py`, учтён event loop для PTB)
- [Ollama](https://ollama.com/) локально или по сети
- Токен Telegram-бота от [@BotFather](https://t.me/BotFather)

## Быстрый старт

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Заполни в `.env` как минимум `TELEGRAM_BOT_TOKEN` и при необходимости `ADMIN_TELEGRAM_ID`.

Скачай модели в Ollama (имена должны совпадать с `.env`):

```bash
ollama pull phi3
ollama pull mistral
ollama pull llama3
ollama pull nomic-embed-text
```

Запуск:

```bash
python bot.py
```

В Telegram: `/start`, для админа — `/myid`, `/reload` после правок в markdown.

## Структура

| Путь | Назначение |
|------|------------|
| `bot.py` | Telegram handlers, загрузка знаний из папки |
| `pipeline.py` | Классификация, RAG-ответ, роутинг моделей, review |
| `rag.py` | Чанки, эмбеддинги, кэш векторов |
| `tools/eval_dialogs.py` | Быстрая офлайн-оценка диалогов из jsonl |
| `data/college_ai_knowledge/` | Источник правды: `.md` |

Источник знаний **только** каталог `data/college_ai_knowledge` (см. `bot.py`).

## GitHub

- **Не коммить** `.env` — в репозитории только `.env.example`.
- Кэш `*.rag_meta.json` / `*.rag_vectors.npz` и `_runtime_overrides.json` в `.gitignore` — на новой машине пересоберутся.
- Если токен когда-либо попал в git — перевыпусти токен у BotFather.

## Лицензия

Не указана — добавь при необходимости.
