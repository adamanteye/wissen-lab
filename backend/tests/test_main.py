import sys
import types
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

fastapi_stub = types.ModuleType("fastapi")


class FastAPI:
    def on_event(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def get(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def post(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class Request:
    async def json(self):
        return {}


fastapi_stub.FastAPI = FastAPI
fastapi_stub.HTTPException = HTTPException
fastapi_stub.Request = Request
sys.modules["fastapi"] = fastapi_stub

db_stub = types.ModuleType("db")


async def noop_async(*args, **kwargs):
    return None


async def search_embeddings_async(*args, **kwargs):
    return []


async def claim_task_async(*args, **kwargs):
    return None


async def commit_needs_index_async(*args, **kwargs):
    return False


async def count_remaining_tasks_async(*args, **kwargs):
    return 0


async def enqueue_branch_missing_commits_async(*args, **kwargs):
    return []


db_stub.claim_task_async = claim_task_async
db_stub.commit_needs_index_async = commit_needs_index_async
db_stub.complete_task_async = noop_async
db_stub.count_remaining_tasks_async = count_remaining_tasks_async
db_stub.enqueue_branch_missing_commits_async = (
    enqueue_branch_missing_commits_async
)
db_stub.enqueue_project_tasks_async = noop_async
db_stub.enqueue_task_async = noop_async
db_stub.ensure_repo_async = noop_async
db_stub.fail_task_async = noop_async
db_stub.get_repo_async = noop_async
db_stub.init_db_async = noop_async
db_stub.mark_branch_deleted_async = noop_async
db_stub.mark_commit_indexed_async = noop_async
db_stub.search_embeddings_async = search_embeddings_async
db_stub.settle_repo_activity_async = noop_async
db_stub.upsert_branch_async = noop_async
db_stub.upsert_commit_async = noop_async
db_stub.upsert_commit_placeholder_async = noop_async
db_stub.upsert_issue_graph_async = noop_async
db_stub.upsert_merge_request_graph_async = noop_async
sys.modules["db"] = db_stub

embed_stub = types.ModuleType("embed")


class LlamaServerEmbed:
    async def accept(self, source_kind: str, chunk_result):
        return []


async def embed_search_text(content: str):
    return []


embed_stub.LlamaServerEmbed = LlamaServerEmbed
embed_stub.embed_search_text = embed_search_text
sys.modules["embed"] = embed_stub

import main
from gitlab import GitLabNotFoundError


async def direct_gitlab_call(func, *args):
    return func(*args)


class ModifiedResourceTests(TestCase):
    def test_build_modified_resources_uses_branch_refs(self):
        events = [
            {
                "push_data": {
                    "ref_type": "branch",
                    "ref": "refs/heads/main",
                }
            },
            {
                "target_type": "Issue",
                "target_iid": 7,
            },
            {
                "note": {
                    "noteable_type": "MergeRequest",
                    "noteable_iid": 11,
                }
            },
        ]

        resources = main.build_modified_resources(events)

        self.assertEqual(["main"], resources["branches"])
        self.assertEqual(["7"], resources["issues"])
        self.assertEqual(["11"], resources["merge_requests"])


class ReconcileResourceTests(IsolatedAsyncioTestCase):
    async def test_reconcile_commit_resource_skips_missing_commit(self):
        gitlab = Mock()
        gitlab.get_commit_object.side_effect = GitLabNotFoundError(
            "https://gitlab.example.com"
        )

        with (
            patch.object(
                main,
                "run_gitlab_call",
                new=direct_gitlab_call,
            ),
            patch.object(
                main,
                "upsert_commit_placeholder_async",
                new=AsyncMock(),
            ) as upsert_commit_placeholder_async,
            patch.object(
                main,
                "mark_commit_indexed_async",
                new=AsyncMock(),
            ) as mark_commit_indexed_async,
        ):
            result = await main.reconcile_commit_resource(
                gitlab,
                "2",
                "deadbeef",
            )

        self.assertIsNone(result)
        upsert_commit_placeholder_async.assert_awaited_once_with(2, "deadbeef")
        mark_commit_indexed_async.assert_awaited_once_with(2, "deadbeef")

    async def test_reconcile_branch_resource_queues_missing_commits(self):
        gitlab = Mock()
        gitlab.get_branch.return_value = {"commit": {"id": "head-sha"}}

        with (
            patch.object(
                main,
                "run_gitlab_call",
                new=direct_gitlab_call,
            ),
            patch.object(
                main,
                "upsert_branch_async",
                new=AsyncMock(),
            ) as upsert_branch_async,
            patch.object(
                main,
                "enqueue_branch_missing_commits_async",
                new=AsyncMock(return_value=["head-sha", "parent-sha"]),
            ) as enqueue_branch_missing_commits_async,
        ):
            result = await main.reconcile_branch_resource(
                gitlab,
                "2",
                "main",
            )

        self.assertEqual(
            {
                "branch": "main",
                "head_sha": "head-sha",
                "queued_commits": ["head-sha", "parent-sha"],
            },
            result,
        )
        upsert_branch_async.assert_awaited_once_with(2, "main", "head-sha")
        enqueue_branch_missing_commits_async.assert_awaited_once_with(2, "main")

    async def test_reconcile_branch_resource_marks_deleted_branch(self):
        gitlab = Mock()
        gitlab.get_branch.side_effect = GitLabNotFoundError(
            "https://gitlab.example.com"
        )

        with (
            patch.object(
                main,
                "run_gitlab_call",
                new=direct_gitlab_call,
            ),
            patch.object(
                main,
                "mark_branch_deleted_async",
                new=AsyncMock(),
            ) as mark_branch_deleted_async,
        ):
            result = await main.reconcile_branch_resource(
                gitlab,
                "2",
                "main",
            )

        self.assertEqual(
            {
                "branch": "main",
                "head_sha": None,
                "queued_commits": [],
            },
            result,
        )
        mark_branch_deleted_async.assert_awaited_once_with(2, "main")


class JsonRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


class SearchEndpointTests(IsolatedAsyncioTestCase):
    async def test_search_requires_page_size(self):
        request = JsonRequest({"content": "mail config", "page": 1})

        with self.assertRaises(main.HTTPException) as exc:
            await main.search(request)

        self.assertEqual(400, exc.exception.status_code)
        self.assertEqual("page_size is required", exc.exception.detail)

    async def test_search_passes_frontend_page_size_to_database(self):
        request = JsonRequest(
            {"content": "mail config", "page": 3, "page_size": 30}
        )

        with (
            patch.object(
                main,
                "embed_search_text",
                new=AsyncMock(return_value=[0.1, 0.2]),
            ) as embed_search_text,
            patch.object(
                main,
                "search_embeddings_async",
                new=AsyncMock(return_value=[{"content": "result"}]),
            ) as search_embeddings_async,
        ):
            result = await main.search(request)

        self.assertEqual([{"content": "result"}], result)
        embed_search_text.assert_awaited_once_with("mail config")
        search_embeddings_async.assert_awaited_once_with([0.1, 0.2], 30, 60)

    async def test_search_caps_page_size_at_maximum(self):
        request = JsonRequest(
            {"content": "mail config", "page": 2, "page_size": 120}
        )

        with (
            patch.object(
                main,
                "embed_search_text",
                new=AsyncMock(return_value=[0.1, 0.2]),
            ),
            patch.object(
                main,
                "search_embeddings_async",
                new=AsyncMock(return_value=[]),
            ) as search_embeddings_async,
        ):
            await main.search(request)

        search_embeddings_async.assert_awaited_once_with([0.1, 0.2], 100, 100)


class TaskConsumerTests(IsolatedAsyncioTestCase):
    async def test_process_claimed_task_uses_supplied_task(self):
        gitlab = Mock()
        task = {
            "id": 11,
            "repo_id": 2,
            "task_kind": "issue",
            "task_key": "7",
            "failure_count": 0,
        }

        with (
            patch.object(
                main,
                "claim_task_async",
                new=AsyncMock(
                    side_effect=AssertionError(
                        "process_claimed_task must not claim again"
                    )
                ),
            ),
            patch.object(
                main,
                "process_task",
                new=AsyncMock(return_value="7"),
            ) as process_task,
            patch.object(
                main,
                "complete_task_async",
                new=AsyncMock(),
            ) as complete_task_async,
            patch.object(
                main,
                "settle_repo_activity_async",
                new=AsyncMock(return_value=True),
            ) as settle_repo_activity_async,
        ):
            result = await main.process_claimed_task(gitlab, task)

        process_task.assert_awaited_once_with(gitlab, task)
        complete_task_async.assert_awaited_once_with(11)
        settle_repo_activity_async.assert_awaited_once_with(
            2, main.TASK_MAX_FAILURES
        )
        self.assertEqual(
            {
                "task_id": 11,
                "project_id": 2,
                "task_kind": "issue",
                "task_key": "7",
                "status": "done",
                "result": "7",
            },
            result,
        )

    async def test_process_task_batch_claims_up_to_limit_then_waits(self):
        gitlab = Mock()
        claimed_tasks = [
            {
                "id": 1,
                "repo_id": 2,
                "task_kind": "issue",
                "task_key": "7",
                "failure_count": 0,
            },
            {
                "id": 2,
                "repo_id": 2,
                "task_kind": "commit",
                "task_key": "abc",
                "failure_count": 1,
            },
        ]

        with (
            patch.object(
                main,
                "claim_task_async",
                new=AsyncMock(side_effect=[*claimed_tasks, None]),
            ) as claim_task_async,
            patch.object(
                main,
                "process_claimed_task",
                new=AsyncMock(
                    side_effect=[
                        {"task_id": 1, "status": "done"},
                        {"task_id": 2, "status": "done"},
                    ]
                ),
            ) as process_claimed_task,
        ):
            result = await main.process_task_batch(gitlab, limit=8)

        self.assertEqual(
            [
                {"task_id": 1, "status": "done"},
                {"task_id": 2, "status": "done"},
            ],
            result,
        )
        self.assertEqual(3, claim_task_async.await_count)
        process_claimed_task.assert_any_await(gitlab, claimed_tasks[0])
        process_claimed_task.assert_any_await(gitlab, claimed_tasks[1])
