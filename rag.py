"""
RAG: извлечение фрагментов из базы знаний, эмбеддинги Ollama, кэш, семантический поиск.
Расширяемость: добавляйте функции в CHUNK_BUILDERS.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np

logger = logging.getLogger(__name__)

CACHE_VERSION = 1


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    """Уникальный id для дедупликации."""
    embed_text: str
    """Текст, по которому считается embedding."""
    fact_text: str
    """Текст факта для промпта модели."""
    topic_hint: str
    """Краткая метка темы (для логов / отладки)."""


ChunkBuilder = Callable[[dict[str, Any]], list[KnowledgeChunk]]


def _chunks_from_knowledge(data: dict[str, Any]) -> list[KnowledgeChunk]:
    out: list[KnowledgeChunk] = []
    kn = data.get("knowledge")
    if not isinstance(kn, dict):
        return out
    for key, val in kn.items():
        if val is None:
            continue
        sval = str(val).strip()
        if not sval:
            continue
        label = str(key).replace("_", " ")
        embed = f"Тема: {label}. {sval}"
        out.append(
            KnowledgeChunk(
                chunk_id=f"knowledge.{key}",
                embed_text=embed,
                fact_text=sval,
                topic_hint=label,
            )
        )
    return out


def _chunks_from_college(data: dict[str, Any]) -> list[KnowledgeChunk]:
    college = data.get("college")
    if not isinstance(college, dict) or not college:
        return []
    parts: list[str] = []
    for json_key in sorted(college.keys()):
        v = college.get(json_key)
        if v is None or v == "":
            continue
        label = str(json_key).replace("_", " ")
        parts.append(f"{label}: {v}")
    if not parts:
        return []
    blob = "Колледж. " + "; ".join(parts)
    return [
        KnowledgeChunk(
            chunk_id="meta.college",
            embed_text=blob,
            fact_text=blob,
            topic_hint="колледж",
        )
    ]


def _chunks_from_specialties(data: dict[str, Any]) -> list[KnowledgeChunk]:
    specs = data.get("specialties")
    if not isinstance(specs, list) or not specs:
        return []
    lines: list[str] = []
    for s in specs:
        if not isinstance(s, dict):
            continue
        name = s.get("name") or ""
        qual = s.get("qualification") or ""
        code = s.get("code") or ""
        a9 = s.get("after_9")
        a11 = s.get("after_11")
        form = s.get("form") or ""
        pay = s.get("payment")
        pay_s = ", ".join(pay) if isinstance(pay, list) else (pay or "")
        lines.append(
            f"{name} ({qual}), код {code}, срок после 9 класса: {a9}, после 11: {a11}, "
            f"форма обучения: {form}, оплата: {pay_s}"
        )
    if not lines:
        return []
    blob = "Специальности и сроки обучения:\n" + "\n".join(lines)
    return [
        KnowledgeChunk(
            chunk_id="meta.specialties",
            embed_text=blob,
            fact_text=blob,
            topic_hint="специальности",
        )
    ]


# Порядок важен: сначала узкие темы, потом обобщённые блоки.
CHUNK_BUILDERS: list[ChunkBuilder] = [
    _chunks_from_knowledge,
    _chunks_from_college,
    _chunks_from_specialties,
]


def get_chunk_builders() -> list[ChunkBuilder]:
    """По умолчанию индексируется только knowledge. RAG_ONLY_KNOWLEDGE=0 включает все блоки."""
    v = os.environ.get("RAG_ONLY_KNOWLEDGE", "1").strip().lower()
    if v in ("0", "false", "no"):
        return list(CHUNK_BUILDERS)
    return [_chunks_from_knowledge]


def knowledge_content_hash(data: dict[str, Any], builders: list[ChunkBuilder] | None = None) -> str:
    """Хэш содержимого всех чанков — для инвалидации кэша эмбеддингов."""
    builders = builders or CHUNK_BUILDERS
    payload: list[dict[str, str]] = []
    for b in builders:
        for c in b(data):
            payload.append({"id": c.chunk_id, "t": c.embed_text})
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_chunks(data: dict[str, Any], builders: list[ChunkBuilder] | None = None) -> list[KnowledgeChunk]:
    builders = builders or CHUNK_BUILDERS
    chunks: list[KnowledgeChunk] = []
    for b in builders:
        chunks.extend(b(data))
    return chunks


def _cache_paths(knowledge_path: Path) -> tuple[Path, Path]:
    stem = knowledge_path.stem
    parent = knowledge_path.parent
    return parent / f"{stem}.rag_meta.json", parent / f"{stem}.rag_vectors.npz"


def _load_meta(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _save_meta(path: Path, meta: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


async def ollama_embed_batch(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    texts: list[str],
) -> list[list[float]]:
    if not texts:
        return []
    url = f"{base_url.rstrip('/')}/api/embed"
    r = await client.post(url, json={"model": model, "input": texts}, timeout=300.0)
    if r.status_code == 404:
        # Старые версии Ollama
        out: list[list[float]] = []
        legacy = f"{base_url.rstrip('/')}/api/embeddings"
        for t in texts:
            lr = await client.post(legacy, json={"model": model, "prompt": t}, timeout=120.0)
            lr.raise_for_status()
            body = lr.json()
            emb = body.get("embedding")
            if not emb:
                raise ValueError("Пустой embedding (legacy API)")
            out.append(emb)
        return out

    r.raise_for_status()
    body = r.json()
    if "embeddings" in body:
        return body["embeddings"]
    one = body.get("embedding")
    if one is not None:
        return [one]
    raise ValueError("Неожиданный ответ /api/embed")


class RAGIndex:
    def __init__(self, chunks: list[KnowledgeChunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("Число чанков и векторов не совпадает")
        self.chunks = chunks
        norms = np.linalg.norm(vectors.astype(np.float32), axis=1, keepdims=True) + np.float32(1e-12)
        self._normed = (vectors.astype(np.float32) / norms).astype(np.float32)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        min_similarity: float,
    ) -> list[tuple[KnowledgeChunk, float]]:
        q = np.array(query_embedding, dtype=np.float32)
        qn = float(np.linalg.norm(q)) + 1e-12
        q = q / qn
        sims = self._normed @ q
        order = np.argsort(-sims)
        out: list[tuple[KnowledgeChunk, float]] = []
        for i in order:
            s = float(sims[int(i)])
            if s < min_similarity:
                break
            out.append((self.chunks[int(i)], s))
            if len(out) >= top_k:
                break
        return out


async def load_or_build_index(
    client: httpx.AsyncClient,
    base_url: str,
    embed_model: str,
    knowledge_path: Path,
    data: dict[str, Any],
    builders: list[ChunkBuilder] | None = None,
) -> RAGIndex:
    builders = builders or CHUNK_BUILDERS
    chunks = build_chunks(data, builders)
    if not chunks:
        raise ValueError("В базе знаний нет фрагментов для индекса (проверьте markdown-файлы).")

    content_hash = knowledge_content_hash(data, builders)
    meta_path, npz_path = _cache_paths(knowledge_path)
    meta = _load_meta(meta_path)

    if (
        meta
        and meta.get("version") == CACHE_VERSION
        and meta.get("content_hash") == content_hash
        and meta.get("embed_model") == embed_model
        and npz_path.is_file()
    ):
        try:
            try:
                with np.load(npz_path, allow_pickle=False) as loaded:
                    vectors = loaded["vectors"]
                    ids = [str(x) for x in loaded["ids"].tolist()]
            except ValueError as e:
                # Миграция legacy-кэша: раньше ids сохранялись как object.
                if "Object arrays cannot be loaded when allow_pickle=False" not in str(e):
                    raise
                logger.info("RAG: найден legacy-кэш, выполняю миграцию формата ids")
                with np.load(npz_path, allow_pickle=True) as legacy:
                    vectors = legacy["vectors"]
                    ids = [str(x) for x in legacy["ids"].tolist()]
                np.savez_compressed(
                    npz_path,
                    vectors=vectors.astype(np.float32),
                    ids=np.asarray(ids, dtype=np.str_),
                )
            if ids == [c.chunk_id for c in chunks]:
                logger.info("RAG: загружен кэш эмбеддингов (%s чанков)", len(chunks))
                return RAGIndex(chunks, vectors)
        except Exception:
            logger.warning("RAG: кэш повреждён, пересчёт эмбеддингов", exc_info=True)

    logger.info("RAG: считаю эмбеддинги (%s чанков), модель %s", len(chunks), embed_model)
    texts = [c.embed_text for c in chunks]
    embeddings = await ollama_embed_batch(client, base_url, embed_model, texts)
    if len(embeddings) != len(chunks):
        raise ValueError("Число векторов от Ollama не совпадает с числом чанков")

    vectors = np.array(embeddings, dtype=np.float32)
    np.savez_compressed(
        npz_path,
        vectors=vectors,
        ids=np.asarray([c.chunk_id for c in chunks], dtype=np.str_),
    )
    _save_meta(
        meta_path,
        {
            "version": CACHE_VERSION,
            "content_hash": content_hash,
            "embed_model": embed_model,
            "chunk_count": len(chunks),
        },
    )
    logger.info("RAG: кэш сохранён: %s", npz_path)
    return RAGIndex(chunks, vectors)
