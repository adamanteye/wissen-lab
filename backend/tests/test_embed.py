import sys
import types
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

aiohttp_stub = types.ModuleType("aiohttp")


class ClientTimeout:
    def __init__(self, total=None):
        self.total = total


class ClientSession:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


aiohttp_stub.ClientResponse = object
aiohttp_stub.ClientSession = ClientSession
aiohttp_stub.ClientTimeout = ClientTimeout
sys.modules["aiohttp"] = aiohttp_stub

db_stub = types.ModuleType("db")


async def replace_embeddings_async(*args, **kwargs):
    return None


db_stub.replace_embeddings_async = replace_embeddings_async
sys.modules["db"] = db_stub

import embed
from chunking import Chunk, ChunkMetadata, ChunkResult


class EmbedAcceptTests(IsolatedAsyncioTestCase):
    async def test_accept_replaces_embeddings_for_all_chunks(self):
        chunk_result = ChunkResult(
            repo_id=3,
            id="abc123",
            chunks=[
                Chunk("first", ChunkMetadata(locator_id="a")),
                Chunk("second", ChunkMetadata(locator_id="b")),
            ],
        )

        with (
            patch.object(
                embed,
                "fetch_embeddings",
                new=AsyncMock(return_value=[[1.0], [2.0]]),
            ),
            patch.object(
                embed,
                "replace_embeddings_async",
                new=AsyncMock(),
            ) as replace_embeddings_async,
        ):
            records = await embed.LlamaServerEmbed().accept(
                "commit", chunk_result
            )

        self.assertEqual(2, len(records))
        self.assertEqual("first", records[0]["content"])
        self.assertEqual([1.0], records[0]["embedding"])
        self.assertEqual("second", records[1]["content"])
        self.assertEqual([2.0], records[1]["embedding"])
        replace_embeddings_async.assert_awaited_once_with(
            3,
            "commit",
            "abc123",
            records,
        )


class EmbedHelpersTests(TestCase):
    def test_normalize_embedding_flattens_nested_vectors(self):
        self.assertEqual([1.0, 2.0], embed.normalize_embedding([1.0, 2.0]))
        self.assertEqual(
            [1.0, 2.0],
            embed.normalize_embedding([[1.0, 2.0]]),
        )
        self.assertIsNone(embed.normalize_embedding("invalid"))
