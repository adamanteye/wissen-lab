import asyncio
import os
from dataclasses import dataclass

import aiohttp

from chunking import ChunkResult
from db import replace_embeddings_async
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
consumer_failure_count = 0
unfinished_job_count = 0


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
        vectors = await asyncio.gather(*(job.future for job in jobs))
        records = [
            {
                "repo_id": job.item.repo_id,
                "source_kind": job.item.source_kind,
                "source_key": job.item.source_key,
                "chunk_index": job.item.chunk_index,
                "locator_id": job.item.locator_id,
                "content": job.item.content,
                "embedding": vector,
            }
            for job, vector in zip(jobs, vectors)
        ]
        await replace_embeddings_async(
            chunk_result.repo_id,
            source_kind,
            chunk_result.id,
            records,
        )
        return records


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
        await enqueue_job(job)
        jobs.append(job)

    return jobs


async def consumer():
    async with aiohttp.ClientSession() as session:
        while True:
            job = await content_queue.get()

            try:
                logger.info(
                    "embedding chunk repo_id=%s source_kind=%s source_key=%s chunk_index=%s locator_id=%s content_length=%s",
                    job.item.repo_id,
                    job.item.source_kind,
                    job.item.source_key,
                    job.item.chunk_index,
                    job.item.locator_id,
                    len(job.item.content),
                )

                await wait_for_request_slot()
                vector = await fetch_native_embedding(session, job.item.content)
                register_consumer_success()

                if not job.future.done():
                    job.future.set_result(vector)
            except Exception:
                register_consumer_failure()
                logger.exception(
                    "llama-server request failed repo_id=%s source_kind=%s source_key=%s chunk_index=%s failure_count=%s",
                    job.item.repo_id,
                    job.item.source_kind,
                    job.item.source_key,
                    job.item.chunk_index,
                    consumer_failure_count,
                )
                if consumer_failure_count >= 5:
                    logger.critical(
                        "llama-server accumulated failure limit reached failure_count=%s",
                        consumer_failure_count,
                    )
                    os._exit(1)

                await enqueue_job(job)
            finally:
                mark_job_finished()
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
    wait_seconds = request_wait_seconds()
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)


async def enqueue_job(job: EmbedJob):
    await maybe_wait_for_backlog()
    increment_unfinished_jobs()
    await content_queue.put(job)


async def maybe_wait_for_backlog():
    n = unfinished_job_count
    if n <= 5:
        return

    wait_seconds = max(
        LLAMA_SERVER_MIN_INTERVAL_SECONDS,
        LLAMA_SERVER_MIN_INTERVAL_SECONDS * 2 ** (n - 1),
    )
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)


def request_wait_seconds():
    if LLAMA_SERVER_MIN_INTERVAL_SECONDS <= 0:
        return 0.0

    if consumer_failure_count <= 0:
        return LLAMA_SERVER_MIN_INTERVAL_SECONDS

    return max(
        LLAMA_SERVER_MIN_INTERVAL_SECONDS,
        LLAMA_SERVER_MIN_INTERVAL_SECONDS * 2 ** (consumer_failure_count - 1),
    )


def register_consumer_success():
    global consumer_failure_count
    consumer_failure_count = max(0, consumer_failure_count - 1)


def register_consumer_failure():
    global consumer_failure_count
    consumer_failure_count += 1


def increment_unfinished_jobs():
    global unfinished_job_count
    unfinished_job_count += 1


def mark_job_finished():
    global unfinished_job_count
    unfinished_job_count = max(0, unfinished_job_count - 1)


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
