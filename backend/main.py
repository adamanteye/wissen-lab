import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from chunking import (
    CommitChunkBuilder,
    IssueChunkBuilder,
    MergeRequestChunkBuilder,
)
from db import (
    delete_embeddings_async,
    ensure_repo_async,
    get_repo_async,
    init_db_async,
    search_embeddings_async,
    update_repo_state_async,
    upsert_commit_async,
    upsert_issue_graph_async,
    upsert_merge_request_graph_async,
)
from embed import (
    LlamaServerEmbed,
    embed_search_text,
    start_consumer,
    stop_consumer,
)
from gitlab import GitLabClient
from logutil import get_logger

logger = get_logger("api")

EPOCH = "1970-01-01T00:00:00Z"
ZERO_SHA = "0" * 40
RECONCILE_INTERVAL_SECONDS = 300


app = FastAPI()
reconcile_task = None


async def index_chunk_result(source_kind: str, chunk_result):
    await delete_embeddings_async(
        chunk_result.repo_id,
        source_kind,
        chunk_result.id,
    )

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


async def ensure_project_repo(gitlab, project_id: str):
    repo_state = gitlab.get_repo_state(project_id)
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
        }
        await ensure_repo_async(repo)
    else:
        await ensure_repo_async(
            {
                "id": repo_state["id"],
                "path": repo_state["path"],
                "self_link": repo_state.get("self_link"),
                "issues_link": repo_state.get("issues_link"),
                "merge_requests_link": repo_state.get("merge_requests_link"),
            }
        )
    return repo_state, repo


def build_modified_resources(events: list[dict[str, Any]]):
    commit_pushes = []
    commit_push_keys = set()
    issues = set()
    merge_requests = set()

    for event in events:
        push_data = event.get("push_data") or {}
        commit_from = push_data.get("commit_from")
        commit_to = push_data.get("commit_to")
        if commit_to and commit_to != ZERO_SHA:
            if commit_from == ZERO_SHA:
                commit_from = None
            key = (commit_from or "", commit_to)
            if key not in commit_push_keys:
                commit_push_keys.add(key)
                commit_pushes.append(
                    {
                        "from_sha": commit_from,
                        "to_sha": commit_to,
                    }
                )

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
        "commit_pushes": commit_pushes,
        "issues": sorted(issues, key=int),
        "merge_requests": sorted(merge_requests, key=int),
    }


async def reconcile_issue_resource(gitlab, project_id: str, issue_iid: str):
    issue = gitlab.get_issue_object(project_id, issue_iid)
    await upsert_issue_graph_async(issue)
    chunk_result = IssueChunkBuilder().build(issue)
    await index_chunk_result("issue", chunk_result)
    return str(issue.iid)


async def reconcile_merge_request_resource(
    gitlab, project_id: str, merge_request_iid: str
):
    merge_request = gitlab.get_merge_request_object(
        project_id, merge_request_iid
    )
    await upsert_merge_request_graph_async(merge_request)
    chunk_result = MergeRequestChunkBuilder().build(merge_request)
    await index_chunk_result("merge_request", chunk_result)
    return str(merge_request.iid)


async def reconcile_commit_resource(gitlab, project_id: str, sha: str):
    commit = gitlab.get_commit_object(project_id, sha)
    await upsert_commit_async(commit)
    chunk_result = CommitChunkBuilder().build(commit)
    await index_chunk_result("commit", chunk_result)
    return commit.sha


async def reconcile_commit_push_resources(
    gitlab, project_id: str, commit_pushes
):
    seen_commits = set()
    reconciled_commits = []

    for commit_push in commit_pushes:
        from_sha = commit_push["from_sha"]
        to_sha = commit_push["to_sha"]

        shas = []
        if from_sha:
            compare = gitlab.compare_commits(project_id, from_sha, to_sha)
            for commit in compare.get("commits") or []:
                sha = str(commit.get("id") or "")
                if sha:
                    shas.append(sha)
        elif to_sha:
            shas.append(to_sha)

        for sha in shas:
            if sha in seen_commits:
                continue
            seen_commits.add(sha)
            reconciled_commits.append(
                await reconcile_commit_resource(gitlab, project_id, sha)
            )

    return reconciled_commits


async def reconcile_issue_resources(gitlab, project_id: str, issues):
    reconciled_issues = []

    for issue_iid in issues:
        reconciled_issues.append(
            await reconcile_issue_resource(gitlab, project_id, issue_iid)
        )

    return reconciled_issues


async def reconcile_merge_request_resources(
    gitlab, project_id: str, merge_requests
):
    reconciled_merge_requests = []

    for merge_request_iid in merge_requests:
        reconciled_merge_requests.append(
            await reconcile_merge_request_resource(
                gitlab, project_id, merge_request_iid
            )
        )

    return reconciled_merge_requests


async def reconcile_modified_resources(
    gitlab, project_id: str, modified_resources
):
    return {
        "commit_pushes": await reconcile_commit_push_resources(
            gitlab, project_id, modified_resources["commit_pushes"]
        ),
        "issues": await reconcile_issue_resources(
            gitlab, project_id, modified_resources["issues"]
        ),
        "merge_requests": await reconcile_merge_request_resources(
            gitlab, project_id, modified_resources["merge_requests"]
        ),
    }


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

        events = gitlab.list_project_events_window(
            project_id,
            after=format_ts(after_dt),
            before=format_ts(left_dt),
        )
        if events:
            oldest = min(parse_ts(event["created_at"]) for event in events)
            return events, format_ts(oldest)

        if after_dt == epoch_dt:
            return [], EPOCH

        window_days *= 2


async def reconcile_project_id(gitlab, project_id: str):
    repo_state, previous_repo = await ensure_project_repo(gitlab, project_id)

    new_events = gitlab.list_project_events_window(
        project_id,
        after=format_ts(previous_repo["right_activity_at"]),
    )
    right_activity_at = previous_repo["right_activity_at"]
    if new_events:
        newest = max(parse_ts(event["created_at"]) for event in new_events)
        right_activity_at = format_ts(newest)

    backfill_events, left_activity_at = await backfill_project_events(
        gitlab,
        project_id,
        previous_repo["left_activity_at"],
    )

    events = new_events + backfill_events
    modified_resources = build_modified_resources(events)
    reconciled = await reconcile_modified_resources(
        gitlab, project_id, modified_resources
    )
    await update_repo_state_async(
        {
            "id": repo_state["id"],
            "path": repo_state["path"],
            "self_link": repo_state.get("self_link"),
            "issues_link": repo_state.get("issues_link"),
            "merge_requests_link": repo_state.get("merge_requests_link"),
            "left_activity_at": left_activity_at,
            "right_activity_at": right_activity_at,
        }
    )

    return {
        "project_id": project_id,
        "repo_id": repo_state["id"],
        "left_activity_at": left_activity_at,
        "right_activity_at": right_activity_at,
        "event_count": len(events),
        "modified_resources": modified_resources,
        "reconciled": reconciled,
    }


async def reconcile_all_projects():
    gitlab = GitLabClient()
    projects = gitlab.list_projects()
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

    return {
        "project_count": len(projects),
        "projects": reconciled_projects,
        "failed_projects": failed_projects,
    }


async def reconcile_loop():
    while True:
        try:
            await reconcile_all_projects()
        except Exception:
            logger.exception("reconcile loop iteration failed")

        await asyncio.sleep(RECONCILE_INTERVAL_SECONDS)


@app.on_event("startup")
async def startup():
    global reconcile_task

    await init_db_async()
    start_consumer()
    if reconcile_task is None or reconcile_task.done():
        reconcile_task = asyncio.create_task(reconcile_loop())


@app.on_event("shutdown")
async def shutdown():
    global reconcile_task

    await stop_consumer()
    if reconcile_task is not None:
        reconcile_task.cancel()
        try:
            await reconcile_task
        except asyncio.CancelledError:
            pass
        reconcile_task = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reconcile/projects")
async def reconcile_projects():
    result = await reconcile_all_projects()
    result["ok"] = True
    return result


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
        top_k = int(payload.get("top_k", 10))
    except (TypeError, ValueError) as exc:
        logger.warning("search returned status=400 reason=invalid_top_k")
        raise HTTPException(
            status_code=400, detail="top_k must be an integer"
        ) from exc

    if top_k < 1:
        logger.warning("search returned status=400 reason=top_k_lt_1")
        raise HTTPException(status_code=400, detail="top_k must be >= 1")

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
        page_size = int(payload.get("page_size", top_k))
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

    top_k = min(top_k, 50)
    page_size = min(page_size, 50)
    offset = (page - 1) * page_size

    try:
        embedding = await embed_search_text(content)
        results = await search_embeddings_async(
            embedding,
            min(top_k, page_size),
            offset,
        )
    except Exception as exc:
        logger.exception(
            "search returned status=500 reason=backend_failure top_k=%s page=%s page_size=%s",
            top_k,
            page,
            page_size,
        )
        raise HTTPException(status_code=500, detail="search failed") from exc

    return results
