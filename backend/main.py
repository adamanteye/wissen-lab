import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from chunking import (
    CommitChunkBuilder,
    IssueChunkBuilder,
    MergeRequestChunkBuilder,
)
from db import (
    claim_task_async,
    commit_needs_index_async,
    complete_task_async,
    count_remaining_tasks_async,
    enqueue_branch_missing_commits_async,
    enqueue_project_tasks_async,
    enqueue_task_async,
    ensure_repo_async,
    fail_task_async,
    get_repo_async,
    init_db_async,
    mark_branch_deleted_async,
    mark_commit_indexed_async,
    search_embeddings_async,
    settle_repo_activity_async,
    upsert_branch_async,
    upsert_commit_async,
    upsert_commit_placeholder_async,
    upsert_issue_graph_async,
    upsert_merge_request_graph_async,
)
from embed import LlamaServerEmbed, embed_search_text
from gitlab import GitLabClient, GitLabNotFoundError
from logutil import get_logger

logger = get_logger("api")

EPOCH = "1970-01-01T00:00:00Z"
RECONCILE_INTERVAL_SECONDS = 300
TASK_IDLE_SECONDS = 1.0
TASK_BATCH_SIZE = 8
TASK_MAX_FAILURES = max(1, int(os.getenv("TASK_MAX_FAILURES", "3")))


app = FastAPI()
project_scan_task = None
task_consumer_task = None


async def run_gitlab_call(func, *args):
    return await asyncio.to_thread(func, *args)


async def index_chunk_result(source_kind: str, chunk_result):
    embedder = LlamaServerEmbed()
    return await embedder.accept(source_kind, chunk_result)


def parse_ts(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def format_ts(value):
    if value is None:
        return None
    dt = parse_ts(value)
    return dt.isoformat().replace("+00:00", "Z")


def normalize_branch_name(ref: str | None) -> str | None:
    if not ref:
        return None

    branch_name = str(ref).strip()
    if not branch_name:
        return None

    prefix = "refs/heads/"
    if branch_name.startswith(prefix):
        branch_name = branch_name[len(prefix) :]

    return branch_name or None


async def ensure_project_repo(gitlab, project_id: str):
    repo_state = await run_gitlab_call(gitlab.get_repo_state, project_id)
    repo = await get_repo_async(repo_state["id"])
    if repo is None:
        last_activity_at = repo_state.get("last_activity_at") or EPOCH
        repo = {
            "id": repo_state["id"],
            "path": repo_state["path"],
            "self_link": repo_state.get("self_link"),
            "issues_link": repo_state.get("issues_link"),
            "merge_requests_link": repo_state.get("merge_requests_link"),
            "left_activity_at": last_activity_at,
            "right_activity_at": last_activity_at,
            "queued_left_activity_at": last_activity_at,
            "queued_right_activity_at": last_activity_at,
        }
    else:
        repo = {
            "id": repo_state["id"],
            "path": repo_state["path"],
            "self_link": repo_state.get("self_link"),
            "issues_link": repo_state.get("issues_link"),
            "merge_requests_link": repo_state.get("merge_requests_link"),
            "left_activity_at": repo["left_activity_at"],
            "right_activity_at": repo["right_activity_at"],
            "queued_left_activity_at": repo["queued_left_activity_at"],
            "queued_right_activity_at": repo["queued_right_activity_at"],
        }

    await ensure_repo_async(repo)
    current_repo = await get_repo_async(repo_state["id"])
    return repo_state, current_repo or repo


def build_modified_resources(events: list[dict[str, Any]]):
    branches = set()
    issues = set()
    merge_requests = set()

    for event in events:
        push_data = event.get("push_data") or {}
        if push_data.get("ref_type") == "branch":
            branch_name = normalize_branch_name(push_data.get("ref"))
            if branch_name is not None:
                branches.add(branch_name)

        target_type = event.get("target_type")
        target_iid = event.get("target_iid")

        if target_type == "Issue" and target_iid is not None:
            issues.add(str(target_iid))
            continue

        if target_type == "MergeRequest" and target_iid is not None:
            merge_requests.add(str(target_iid))
            continue

        note = event.get("note") or {}
        noteable_type = note.get("noteable_type")
        noteable_iid = note.get("noteable_iid")

        if noteable_type == "Issue" and noteable_iid is not None:
            issues.add(str(noteable_iid))
            continue

        if noteable_type == "MergeRequest" and noteable_iid is not None:
            merge_requests.add(str(noteable_iid))

    return {
        "branches": sorted(branches),
        "issues": sorted(issues, key=int),
        "merge_requests": sorted(merge_requests, key=int),
    }


async def reconcile_issue_resource(gitlab, project_id: str, issue_iid: str):
    issue = await run_gitlab_call(gitlab.get_issue_object, project_id, issue_iid)
    await upsert_issue_graph_async(issue)
    chunk_result = IssueChunkBuilder().build(issue)
    await index_chunk_result("issue", chunk_result)
    logger.info(
        "issue reconciled project_id=%s issue_iid=%s",
        project_id,
        issue.iid,
    )
    return str(issue.iid)


async def reconcile_merge_request_resource(
    gitlab, project_id: str, merge_request_iid: str
):
    merge_request = await run_gitlab_call(
        gitlab.get_merge_request_object,
        project_id,
        merge_request_iid,
    )
    await upsert_merge_request_graph_async(merge_request)
    chunk_result = MergeRequestChunkBuilder().build(merge_request)
    await index_chunk_result("merge_request", chunk_result)
    logger.info(
        "merge request reconciled project_id=%s merge_request_iid=%s",
        project_id,
        merge_request.iid,
    )
    return str(merge_request.iid)


async def reconcile_branch_resource(gitlab, project_id: str, branch_name: str):
    repo_id = int(project_id)

    try:
        branch = await run_gitlab_call(gitlab.get_branch, project_id, branch_name)
    except GitLabNotFoundError:
        logger.warning(
            "branch reconcile skipped project_id=%s branch=%s reason=not_found",
            project_id,
            branch_name,
        )
        await mark_branch_deleted_async(repo_id, branch_name)
        return {
            "branch": branch_name,
            "head_sha": None,
            "queued_commits": [],
        }

    head = branch.get("commit") or {}
    head_sha = str(head.get("id") or "").strip()
    if not head_sha:
        raise ValueError(
            f"branch {branch_name} in project {project_id} is missing head commit"
        )

    await upsert_branch_async(repo_id, branch_name, head_sha)
    queued_commits = await enqueue_branch_missing_commits_async(
        repo_id, branch_name
    )
    logger.info(
        "branch reconciled project_id=%s branch=%s head_sha=%s queued_commit_count=%s",
        project_id,
        branch_name,
        head_sha,
        len(queued_commits),
    )
    return {
        "branch": branch_name,
        "head_sha": head_sha,
        "queued_commits": queued_commits,
    }


async def reconcile_commit_resource(gitlab, project_id: str, sha: str):
    repo_id = int(project_id)

    try:
        commit = await run_gitlab_call(gitlab.get_commit_object, project_id, sha)
    except GitLabNotFoundError:
        logger.warning(
            "commit reconcile skipped project_id=%s sha=%s reason=not_found",
            project_id,
            sha,
        )
        await upsert_commit_placeholder_async(repo_id, sha)
        await mark_commit_indexed_async(repo_id, sha)
        return None

    await upsert_commit_async(commit)
    chunk_result = CommitChunkBuilder().build(commit)
    await index_chunk_result("commit", chunk_result)
    await mark_commit_indexed_async(commit.repo_id, commit.sha)
    logger.info(
        "commit reconciled project_id=%s sha=%s parent_count=%s",
        project_id,
        commit.sha,
        len(commit.parent_shas),
    )

    for parent_sha in commit.parent_shas:
        if await commit_needs_index_async(commit.repo_id, parent_sha):
            await enqueue_task_async(commit.repo_id, "commit", parent_sha)

    return commit.sha


async def backfill_project_events(gitlab, project_id: str, left_activity_at):
    left_dt = parse_ts(left_activity_at)
    epoch_dt = parse_ts(EPOCH)
    if left_dt is None or left_dt <= epoch_dt:
        return [], EPOCH

    window_days = 2

    while True:
        after_dt = left_dt - timedelta(days=window_days)
        if after_dt < epoch_dt:
            after_dt = epoch_dt

        events = await run_gitlab_call(
            gitlab.list_project_events_window,
            project_id,
            format_ts(after_dt),
            format_ts(left_dt),
        )
        if events:
            oldest = min(parse_ts(event["created_at"]) for event in events)
            return events, format_ts(oldest)

        if after_dt == epoch_dt:
            return [], EPOCH

        window_days *= 2


async def reconcile_project_id(gitlab, project_id: str):
    repo_state, previous_repo = await ensure_project_repo(gitlab, project_id)

    new_events = await run_gitlab_call(
        gitlab.list_project_events_window,
        project_id,
        format_ts(previous_repo["queued_right_activity_at"]),
    )
    queued_right_activity_at = previous_repo["queued_right_activity_at"]
    if new_events:
        newest = max(parse_ts(event["created_at"]) for event in new_events)
        queued_right_activity_at = format_ts(newest)

    backfill_events, queued_left_activity_at = await backfill_project_events(
        gitlab,
        project_id,
        previous_repo["queued_left_activity_at"],
    )

    events = new_events + backfill_events
    modified_resources = build_modified_resources(events)
    if events:
        remaining_task_count = await count_remaining_tasks_async(
            repo_state["id"], TASK_MAX_FAILURES
        )
        logger.info(
            "project task submission started project_id=%s remaining_task_count=%s branch_count=%s issue_count=%s merge_request_count=%s",
            project_id,
            remaining_task_count,
            len(modified_resources["branches"]),
            len(modified_resources["issues"]),
            len(modified_resources["merge_requests"]),
        )
        await enqueue_project_tasks_async(
            {
                "id": repo_state["id"],
                "path": repo_state["path"],
                "self_link": repo_state.get("self_link"),
                "issues_link": repo_state.get("issues_link"),
                "merge_requests_link": repo_state.get("merge_requests_link"),
                "left_activity_at": previous_repo["left_activity_at"],
                "right_activity_at": previous_repo["right_activity_at"],
                "queued_left_activity_at": queued_left_activity_at,
                "queued_right_activity_at": queued_right_activity_at,
            },
            modified_resources,
        )

    activity_settled = await settle_repo_activity_async(
        repo_state["id"], TASK_MAX_FAILURES
    )

    return {
        "project_id": project_id,
        "event_count": len(events),
        "queued_left_activity_at": queued_left_activity_at,
        "queued_right_activity_at": queued_right_activity_at,
        "modified_resources": modified_resources,
        "activity_settled": activity_settled,
    }


async def reconcile_all_projects(
    gitlab: GitLabClient | None = None,
):
    gitlab = gitlab or GitLabClient()
    projects = await run_gitlab_call(gitlab.list_projects)
    logger.info("project scan started project_count=%s", len(projects))
    reconciled_projects = []
    failed_projects = []

    for project in projects:
        project_id = str(project["id"])
        try:
            reconciled_projects.append(
                await reconcile_project_id(gitlab, project_id)
            )
        except Exception as exc:
            logger.exception(
                "project reconcile failed project_id=%s",
                project_id,
            )
            failed_projects.append(
                {
                    "project_id": project_id,
                    "error": str(exc),
                }
            )

    result = {
        "project_count": len(projects),
        "projects": reconciled_projects,
        "failed_projects": failed_projects,
    }
    logger.info(
        "project scan finished project_count=%s reconciled_count=%s failed_count=%s",
        result["project_count"],
        len(result["projects"]),
        len(result["failed_projects"]),
    )
    return result


async def process_task(gitlab, task: dict):
    project_id = str(task["repo_id"])
    task_kind = task["task_kind"]
    task_key = task["task_key"]

    if task_kind == "issue":
        return await reconcile_issue_resource(gitlab, project_id, task_key)

    if task_kind == "merge_request":
        return await reconcile_merge_request_resource(
            gitlab, project_id, task_key
        )

    if task_kind == "branch":
        return await reconcile_branch_resource(gitlab, project_id, task_key)

    if task_kind == "commit":
        return await reconcile_commit_resource(gitlab, project_id, task_key)

    raise ValueError(f"unsupported task kind: {task_kind}")


async def claim_task_batch(limit: int = TASK_BATCH_SIZE):
    claimed_tasks = []

    for _ in range(limit):
        task = await claim_task_async(TASK_MAX_FAILURES)
        if task is None:
            break

        logger.info(
            "task claimed task_id=%s project_id=%s task_kind=%s task_key=%s failure_count=%s",
            task["id"],
            task["repo_id"],
            task["task_kind"],
            task["task_key"],
            task["failure_count"],
        )
        claimed_tasks.append(task)

    return claimed_tasks


async def process_claimed_task(gitlab, task: dict):
    try:
        result = await process_task(gitlab, task)
    except Exception as exc:
        logger.exception(
            "task reconcile failed task_id=%s project_id=%s task_kind=%s task_key=%s",
            task["id"],
            task["repo_id"],
            task["task_kind"],
            task["task_key"],
        )
        failure = await fail_task_async(
            task["id"],
            str(exc),
            TASK_MAX_FAILURES,
        )
        await settle_repo_activity_async(task["repo_id"], TASK_MAX_FAILURES)
        return {
            "task_id": task["id"],
            "project_id": task["repo_id"],
            "task_kind": task["task_kind"],
            "task_key": task["task_key"],
            "status": failure["status"] if failure is not None else "pending",
            "failure_count": failure["failure_count"]
            if failure is not None
            else None,
            "error": str(exc),
        }

    await complete_task_async(task["id"])
    await settle_repo_activity_async(task["repo_id"], TASK_MAX_FAILURES)
    logger.info(
        "task completed task_id=%s project_id=%s task_kind=%s task_key=%s",
        task["id"],
        task["repo_id"],
        task["task_kind"],
        task["task_key"],
    )
    return {
        "task_id": task["id"],
        "project_id": task["repo_id"],
        "task_kind": task["task_kind"],
        "task_key": task["task_key"],
        "status": "done",
        "result": result,
    }


async def process_task_batch(gitlab, limit: int = TASK_BATCH_SIZE):
    claimed_tasks = await claim_task_batch(limit)
    if not claimed_tasks:
        return []

    logger.info("task batch started task_count=%s", len(claimed_tasks))
    processed = await asyncio.gather(
        *(process_claimed_task(gitlab, task) for task in claimed_tasks)
    )
    logger.info("task batch finished task_count=%s", len(processed))
    return list(processed)


async def project_scan_loop():
    gitlab = GitLabClient()

    while True:
        try:
            await reconcile_all_projects(gitlab)
        except Exception:
            logger.exception("project scan iteration failed")

        await asyncio.sleep(RECONCILE_INTERVAL_SECONDS)


async def task_consumer_loop():
    gitlab = GitLabClient()

    while True:
        try:
            processed_tasks = await process_task_batch(gitlab)
        except Exception:
            logger.exception("task batch iteration failed")
            processed_tasks = []

        if processed_tasks:
            continue
        await asyncio.sleep(TASK_IDLE_SECONDS)


@app.on_event("startup")
async def startup():
    global project_scan_task, task_consumer_task

    await init_db_async()
    if project_scan_task is None or project_scan_task.done():
        project_scan_task = asyncio.create_task(project_scan_loop())
    if task_consumer_task is None or task_consumer_task.done():
        task_consumer_task = asyncio.create_task(task_consumer_loop())


@app.on_event("shutdown")
async def shutdown():
    global project_scan_task, task_consumer_task

    if project_scan_task is not None:
        project_scan_task.cancel()
        try:
            await project_scan_task
        except asyncio.CancelledError:
            pass
        project_scan_task = None

    if task_consumer_task is not None:
        task_consumer_task.cancel()
        try:
            await task_consumer_task
        except asyncio.CancelledError:
            pass
        task_consumer_task = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reconcile/projects")
async def reconcile_projects():
    gitlab = GitLabClient()
    queued = await reconcile_all_projects(gitlab)
    queued["processed_tasks"] = await process_task_batch(gitlab)
    queued["ok"] = True
    return queued


@app.post("/search")
async def search(request: Request):
    try:
        payload = await request.json()
    except Exception as exc:
        logger.warning("search returned status=400 reason=invalid_json")
        raise HTTPException(
            status_code=400, detail="invalid json payload"
        ) from exc

    content = str(payload.get("content", "")).strip()
    if not content:
        logger.warning("search returned status=400 reason=missing_content")
        raise HTTPException(status_code=400, detail="content is required")

    try:
        page = int(payload.get("page", 1))
    except (TypeError, ValueError) as exc:
        logger.warning("search returned status=400 reason=invalid_page")
        raise HTTPException(
            status_code=400,
            detail="page must be an integer",
        ) from exc

    if page < 1:
        logger.warning("search returned status=400 reason=page_lt_1")
        raise HTTPException(status_code=400, detail="page must be >= 1")

    try:
        page_size = int(payload.get("page_size", 10))
    except (TypeError, ValueError) as exc:
        logger.warning("search returned status=400 reason=invalid_page_size")
        raise HTTPException(
            status_code=400,
            detail="page_size must be an integer",
        ) from exc

    if page_size < 1:
        logger.warning("search returned status=400 reason=page_size_lt_1")
        raise HTTPException(
            status_code=400,
            detail="page_size must be >= 1",
        )

    page_size = min(page_size, 50)
    offset = (page - 1) * page_size

    try:
        embedding = await embed_search_text(content)
        results = await search_embeddings_async(
            embedding,
            page_size,
            offset,
        )
    except Exception as exc:
        logger.exception(
            "search returned status=500 reason=backend_failure page=%s page_size=%s",
            page,
            page_size,
        )
        raise HTTPException(status_code=500, detail="search failed") from exc

    return results
