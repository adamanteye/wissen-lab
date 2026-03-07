import asyncio
import os
from dataclasses import dataclass

import aiohttp

from chunking import ChunkResult
from db import store_embedding_async
from logutil import get_logger

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "").rstrip("/")
LLAMA_SERVER_API_KEY = os.getenv("LLAMA_SERVER_API_KEY", "")
LLAMA_SERVER_MIN_INTERVAL_SECONDS = float(
    os.getenv("LLAMA_SERVER_MIN_INTERVAL_SECONDS", "0.2")
)
logger = get_logger("llama_server")

content_queue = asyncio.Queue()
consumer_task = None
search_embedding_cache = {}
request_rate_lock = asyncio.Lock()
last_request_at = 0.0


@dataclass
class EmbedItem:
    repo_id: int
    source_kind: str
    source_key: str
    chunk_index: int
    locator_id: str | None
    content: str


@dataclass
class EmbedJob:
    item: EmbedItem
    future: asyncio.Future


class LlamaServerEmbed:
    async def accept(self, source_kind: str, chunk_result: ChunkResult):
        jobs = await producer(source_kind, chunk_result)
        return await asyncio.gather(*(job.future for job in jobs))


async def embed_search_text(content: str):
    cached = search_embedding_cache.get(content)
    if cached is not None:
        return cached

    async with aiohttp.ClientSession() as session:
        embeddings = await fetch_embeddings(session, [content])

    vector = embeddings[0] if embeddings else []
    search_embedding_cache[content] = vector
    return vector


async def producer(source_kind: str, chunk_result: ChunkResult):
    loop = asyncio.get_running_loop()
    jobs = []

    for index, chunk in enumerate(chunk_result.chunks):
        item = EmbedItem(
            repo_id=chunk_result.repo_id,
            source_kind=source_kind,
            source_key=chunk_result.id,
            chunk_index=index,
            locator_id=chunk.metadata.locator_id,
            content=chunk.content,
        )
        job = EmbedJob(item=item, future=loop.create_future())
        await content_queue.put(job)
        jobs.append(job)

    return jobs


async def consumer():
    async with aiohttp.ClientSession() as session:
        while True:
            first = await content_queue.get()
            jobs = [first]

            while True:
                try:
                    job = content_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                jobs.append(job)

            try:
                vectors = await fetch_embeddings(
                    session,
                    [job.item.content for job in jobs],
                )

                for job, vector in zip(jobs, vectors):
                    record = await store_embedding(job.item, vector)
                    if not job.future.done():
                        job.future.set_result(record)
            except Exception as exc:
                logger.exception(
                    "llama-server batch processing failed job_count=%s",
                    len(jobs),
                )
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(exc)
            finally:
                for _ in jobs:
                    content_queue.task_done()


async def fetch_embeddings(session: aiohttp.ClientSession, texts: list[str]):
    if not texts:
        return []

    if not LLAMA_SERVER_URL:
        return [[] for _ in texts]

    embeddings = []
    for text in texts:
        embeddings.append(await fetch_native_embedding(session, text))
    return embeddings


def embedding_url():
    if not LLAMA_SERVER_URL:
        return ""

    if LLAMA_SERVER_URL.endswith("/embedding"):
        return LLAMA_SERVER_URL

    return f"{LLAMA_SERVER_URL}/embedding"


def embedding_headers():
    headers = {}
    if LLAMA_SERVER_API_KEY:
        headers["Authorization"] = f"Bearer {LLAMA_SERVER_API_KEY}"
    return headers


async def fetch_native_embedding(session: aiohttp.ClientSession, text: str):
    url = embedding_url()
    await wait_for_request_slot()

    try:
        async with session.post(
            url,
            json={"content": text},
            headers=embedding_headers(),
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json()
    except aiohttp.ClientError:
        logger.exception("llama-server request failed url=%s", url)
        raise
    except Exception:
        logger.exception("llama-server returned invalid response url=%s", url)
        raise

    embedding = parse_embedding_payload(payload)
    if embedding is not None:
        return embedding

    logger.error(
        "llama-server returned invalid embedding payload url=%s payload_type=%s",
        url,
        type(payload).__name__,
    )
    raise ValueError("invalid llama-server response")


async def wait_for_request_slot():
    global last_request_at

    if LLAMA_SERVER_MIN_INTERVAL_SECONDS <= 0:
        return

    loop = asyncio.get_running_loop()

    async with request_rate_lock:
        now = loop.time()
        wait_seconds = last_request_at + LLAMA_SERVER_MIN_INTERVAL_SECONDS - now
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)
        last_request_at = loop.time()


def parse_embedding_payload(payload):
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            return normalize_embedding(first.get("embedding"))

    if isinstance(payload, dict):
        return normalize_embedding(payload.get("embedding"))

    return None


def normalize_embedding(embedding):
    if not isinstance(embedding, list) or not embedding:
        return None

    first = embedding[0]
    if isinstance(first, (int, float)):
        return embedding

    if (
        len(embedding) == 1
        and isinstance(first, list)
        and first
        and isinstance(first[0], (int, float))
    ):
        return first

    return None


async def store_embedding(item: EmbedItem, vector: list[float]):
    record = {
        "repo_id": item.repo_id,
        "source_kind": item.source_kind,
        "source_key": item.source_key,
        "chunk_index": item.chunk_index,
        "locator_id": item.locator_id,
        "content": item.content,
        "embedding": vector,
    }
    await store_embedding_async(record)
    return record


def start_consumer():
    global consumer_task

    if consumer_task is None or consumer_task.done():
        consumer_task = asyncio.create_task(consumer())

    return consumer_task


async def stop_consumer():
    global consumer_task

    if consumer_task is None:
        return

    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass

    consumer_task = None
