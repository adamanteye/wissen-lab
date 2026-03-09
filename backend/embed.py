import os

import aiohttp

from chunking import ChunkResult
from db import replace_embeddings_async
from logutil import get_logger


LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "").rstrip("/")
LLAMA_SERVER_API_KEY = os.getenv("LLAMA_SERVER_API_KEY", "")
LLAMA_SERVER_TIMEOUT_SECONDS = float(
    os.getenv("LLAMA_SERVER_TIMEOUT_SECONDS", "30")
)
logger = get_logger("llama_server")

search_embedding_cache = {}


class LlamaServerEmbed:
    async def accept(self, source_kind: str, chunk_result: ChunkResult):
        texts = [chunk.content for chunk in chunk_result.chunks]

        async with create_client_session() as session:
            embeddings = await fetch_embeddings(session, texts)

        records = [
            {
                "repo_id": chunk_result.repo_id,
                "source_kind": source_kind,
                "source_key": chunk_result.id,
                "chunk_index": index,
                "locator_id": chunk.metadata.locator_id,
                "content": chunk.content,
                "embedding": vector,
            }
            for index, (chunk, vector) in enumerate(
                zip(chunk_result.chunks, embeddings)
            )
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

    async with create_client_session() as session:
        embeddings = await fetch_embeddings(session, [content])

    vector = embeddings[0] if embeddings else []
    search_embedding_cache[content] = vector
    return vector


def create_client_session() -> aiohttp.ClientSession:
    kwargs = {}
    timeout = request_timeout()
    if timeout is not None:
        kwargs["timeout"] = timeout
    return aiohttp.ClientSession(**kwargs)


def request_timeout() -> aiohttp.ClientTimeout | None:
    if LLAMA_SERVER_TIMEOUT_SECONDS <= 0:
        return None
    return aiohttp.ClientTimeout(total=LLAMA_SERVER_TIMEOUT_SECONDS)


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
    if not url:
        raise ValueError("LLAMA_SERVER_URL is not set")

    async with session.post(
        url,
        json={"content": text},
        headers=embedding_headers(),
    ) as resp:
        payload = await read_json_response(resp)
        if resp.status >= 400:
            message = (
                error_message_from_payload(payload)
                or f"llama-server returned status {resp.status}"
            )
            logger.error(
                "llama-server request failed url=%s status=%s message=%s",
                url,
                resp.status,
                message,
            )
            raise ValueError(message)

    embedding = parse_embedding_payload(payload)
    if embedding is not None:
        return embedding

    raise ValueError(
        f"invalid llama-server embedding payload type={type(payload).__name__}"
    )


async def read_json_response(resp: aiohttp.ClientResponse):
    try:
        return await resp.json(content_type=None)
    except Exception:
        text = await resp.text()
        logger.error(
            "llama-server returned non-json response url=%s status=%s body=%r",
            str(resp.url),
            resp.status,
            text[:200],
        )
        return None


def error_message_from_payload(payload) -> str | None:
    if not isinstance(payload, dict):
        return None

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message

    message = payload.get("message")
    if isinstance(message, str):
        return message

    return None


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
    return None


async def stop_consumer():
    return None
