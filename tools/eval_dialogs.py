"""
Лёгкая офлайн-оценка диалогов бота по jsonl.

Формат входа (одна строка = JSON):
{"user":"где оценки", "bot":"Оценки смотри в SmartNation ...", "expected_keywords":["smartnation"]}
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalRow:
    user: str
    bot: str
    expected_keywords: list[str]


def _load_rows(path: Path) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        rows.append(
            EvalRow(
                user=str(obj.get("user", "")).strip(),
                bot=str(obj.get("bot", "")).strip(),
                expected_keywords=[str(x).lower().strip() for x in obj.get("expected_keywords", []) if str(x).strip()],
            )
        )
    return rows


def _is_listy_answer(bot: str) -> bool:
    return bool(re.search(r"(^|\s)(1[\)\.]|2[\)\.])\s", bot))


def _is_hallucination_risk(bot: str) -> bool:
    # Простейший эвристический флаг для явно неуверенных/странных формулировок.
    red_flags = ("не указано", "нет данных", "предположительно", "возможно")
    return any(x in bot.lower() for x in red_flags) and len(bot) > 160


def evaluate(rows: list[EvalRow]) -> dict[str, float]:
    if not rows:
        return {"rows": 0}
    keyword_hits = 0
    listy = 0
    too_long = 0
    hallucination_risk = 0
    for row in rows:
        b = row.bot.lower()
        if row.expected_keywords:
            if all(k in b for k in row.expected_keywords):
                keyword_hits += 1
        else:
            keyword_hits += 1
        if _is_listy_answer(row.bot):
            listy += 1
        if len(row.bot) > 500:
            too_long += 1
        if _is_hallucination_risk(row.bot):
            hallucination_risk += 1
    total = len(rows)
    return {
        "rows": float(total),
        "keyword_accuracy": keyword_hits / total,
        "listy_answer_rate": listy / total,
        "too_long_rate": too_long / total,
        "hallucination_risk_rate": hallucination_risk / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Путь до jsonl с тест-кейсами")
    args = parser.parse_args()
    rows = _load_rows(Path(args.input))
    metrics = evaluate(rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
