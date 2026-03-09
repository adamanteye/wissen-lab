from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row

from chunking import Commit, Issue, MergeRequest
from logutil import get_logger


EMBEDDING_DIM = 4096
PG_HOST = os.getenv("PG_HOST", "")
PG_PASS = os.getenv("PG_PASS", "")
EPOCH = "1970-01-01T00:00:00Z"
TASK_CLAIM_TIMEOUT_SECONDS = 900
logger = get_logger("pg")

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS repo (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL,
  self_link TEXT,
  issues_link TEXT,
  merge_requests_link TEXT,
  left_activity_at TIMESTAMPTZ NOT NULL DEFAULT '1970-01-01T00:00:00Z',
  right_activity_at TIMESTAMPTZ NOT NULL DEFAULT '1970-01-01T00:00:00Z',
  queued_left_activity_at TIMESTAMPTZ NOT NULL DEFAULT '1970-01-01T00:00:00Z',
  queued_right_activity_at TIMESTAMPTZ NOT NULL DEFAULT '1970-01-01T00:00:00Z',
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS issue (
  repo_id INTEGER NOT NULL REFERENCES repo(id),
  iid INTEGER NOT NULL,
  title TEXT NOT NULL DEFAULT '',
  description TEXT NOT NULL DEFAULT '',
  url TEXT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (repo_id, iid)
);

CREATE TABLE IF NOT EXISTS issue_note (
  repo_id INTEGER NOT NULL,
  issue_iid INTEGER NOT NULL,
  id BIGINT NOT NULL,
  body TEXT NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (repo_id, issue_iid, id),
  FOREIGN KEY (repo_id, issue_iid) REFERENCES issue(repo_id, iid)
);

CREATE TABLE IF NOT EXISTS merge_request (
  repo_id INTEGER NOT NULL REFERENCES repo(id),
  iid INTEGER NOT NULL,
  title TEXT NOT NULL DEFAULT '',
  description TEXT NOT NULL DEFAULT '',
  url TEXT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (repo_id, iid)
);

CREATE TABLE IF NOT EXISTS merge_request_note (
  repo_id INTEGER NOT NULL,
  merge_request_iid INTEGER NOT NULL,
  id BIGINT NOT NULL,
  body TEXT NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (repo_id, merge_request_iid, id),
  FOREIGN KEY (repo_id, merge_request_iid)
    REFERENCES merge_request(repo_id, iid)
);

CREATE TABLE IF NOT EXISTS git_commit (
  repo_id INTEGER NOT NULL REFERENCES repo(id),
  sha TEXT NOT NULL,
  message TEXT NOT NULL DEFAULT '',
  url TEXT,
  indexed_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (repo_id, sha)
);

CREATE TABLE IF NOT EXISTS repo_branch (
  repo_id INTEGER NOT NULL REFERENCES repo(id),
  name TEXT NOT NULL,
  head_sha TEXT,
  is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (repo_id, name),
  FOREIGN KEY (repo_id, head_sha) REFERENCES git_commit(repo_id, sha)
);

CREATE TABLE IF NOT EXISTS git_commit_edge (
  repo_id INTEGER NOT NULL REFERENCES repo(id),
  child_sha TEXT NOT NULL,
  parent_sha TEXT NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (repo_id, child_sha, parent_sha),
  FOREIGN KEY (repo_id, child_sha) REFERENCES git_commit(repo_id, sha),
  FOREIGN KEY (repo_id, parent_sha) REFERENCES git_commit(repo_id, sha)
);

CREATE TABLE IF NOT EXISTS gitlab_task (
  id BIGSERIAL PRIMARY KEY,
  repo_id INTEGER NOT NULL REFERENCES repo(id),
  task_kind TEXT NOT NULL,
  task_key TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  failure_count INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  UNIQUE (repo_id, task_kind, task_key)
);

CREATE INDEX IF NOT EXISTS idx_gitlab_task_claim
  ON gitlab_task(status, failure_count, created_at);

CREATE INDEX IF NOT EXISTS idx_gitlab_task_repo
  ON gitlab_task(repo_id, status, failure_count);

CREATE TABLE IF NOT EXISTS gitlab_embedding (
  repo_id INTEGER NOT NULL REFERENCES repo(id),
  source_kind TEXT NOT NULL,
  source_key TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  locator_id TEXT,
  content TEXT NOT NULL,
  embedding vector(4096),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (repo_id, source_kind, source_key, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_gitlab_embedding_source
  ON gitlab_embedding(repo_id, source_kind, source_key);
"""


class Database:
    def __init__(self, host: str, password: str) -> None:
        self._host = host
        self._password = password

    def init_schema(self) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
            conn.commit()

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection]:
        if not self._host:
            raise ValueError("PG_HOST is not set")

        if not self._password:
            raise ValueError("PG_PASS is not set")

        with psycopg.connect(
            host=self._host,
            user="postgres",
            password=self._password,
            dbname="postgres",
        ) as conn:
            yield conn

    def ensure_repo(self, repo: dict) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO repo (
                        id,
                        path,
                        self_link,
                        issues_link,
                        merge_requests_link,
                        left_activity_at,
                        right_activity_at,
                        queued_left_activity_at,
                        queued_right_activity_at,
                        updated_at
                    )
                    VALUES (
                        %(id)s,
                        %(path)s,
                        %(self_link)s,
                        %(issues_link)s,
                        %(merge_requests_link)s,
                        %(left_activity_at)s,
                        %(right_activity_at)s,
                        %(queued_left_activity_at)s,
                        %(queued_right_activity_at)s,
                        now()
                    )
                    ON CONFLICT (id)
                    DO UPDATE SET
                        path = EXCLUDED.path,
                        self_link = EXCLUDED.self_link,
                        issues_link = EXCLUDED.issues_link,
                        merge_requests_link = EXCLUDED.merge_requests_link,
                        left_activity_at = COALESCE(
                            EXCLUDED.left_activity_at,
                            repo.left_activity_at
                        ),
                        right_activity_at = COALESCE(
                            EXCLUDED.right_activity_at,
                            repo.right_activity_at
                        ),
                        queued_left_activity_at = COALESCE(
                            EXCLUDED.queued_left_activity_at,
                            repo.queued_left_activity_at
                        ),
                        queued_right_activity_at = COALESCE(
                            EXCLUDED.queued_right_activity_at,
                            repo.queued_right_activity_at
                        ),
                        updated_at = now()
                    """,
                    {
                        "id": repo["id"],
                        "path": repo["path"],
                        "self_link": repo.get("self_link"),
                        "issues_link": repo.get("issues_link"),
                        "merge_requests_link": repo.get("merge_requests_link"),
                        "left_activity_at": repo.get("left_activity_at", EPOCH),
                        "right_activity_at": repo.get("right_activity_at", EPOCH),
                        "queued_left_activity_at": repo.get(
                            "queued_left_activity_at",
                            repo.get("left_activity_at", EPOCH),
                        ),
                        "queued_right_activity_at": repo.get(
                            "queued_right_activity_at",
                            repo.get("right_activity_at", EPOCH),
                        ),
                    },
                )
            conn.commit()

    def get_repo(self, repo_id: int) -> dict | None:
        with self.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        id,
                        path,
                        self_link,
                        issues_link,
                        merge_requests_link,
                        left_activity_at,
                        right_activity_at,
                        queued_left_activity_at,
                        queued_right_activity_at,
                        updated_at
                    FROM repo
                    WHERE id = %s
                    """,
                    (repo_id,),
                )
                row = cur.fetchone()

        if row is None:
            return None

        return dict(row)

    def enqueue_project_tasks(self, repo: dict, tasks: dict[str, list[str]]) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                self.ensure_repo_in_cursor(cur, repo)

                for issue_iid in tasks.get("issues", []):
                    self._enqueue_task(cur, repo["id"], "issue", issue_iid)

                for merge_request_iid in tasks.get("merge_requests", []):
                    self._enqueue_task(
                        cur,
                        repo["id"],
                        "merge_request",
                        merge_request_iid,
                    )

                for branch_name in tasks.get("branches", []):
                    self._enqueue_task(cur, repo["id"], "branch", branch_name)

            conn.commit()

    def enqueue_task(self, repo_id: int, task_kind: str, task_key: str) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                self._enqueue_task(cur, repo_id, task_kind, task_key)
            conn.commit()

    def claim_task(self, max_failures: int) -> dict | None:
        with self.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    WITH next_task AS (
                        SELECT id
                        FROM gitlab_task
                        WHERE failure_count < %s
                          AND (
                            status = 'pending'
                            OR (
                              status = 'running'
                              AND started_at < now() - (%s * interval '1 second')
                            )
                          )
                        ORDER BY failure_count ASC, created_at ASC
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    )
                    UPDATE gitlab_task AS task
                    SET
                        status = 'running',
                        started_at = now(),
                        updated_at = now(),
                        last_error = NULL
                    FROM next_task
                    WHERE task.id = next_task.id
                    RETURNING
                        task.id,
                        task.repo_id,
                        task.task_kind,
                        task.task_key,
                        task.failure_count,
                        task.created_at
                    """,
                    (max_failures, TASK_CLAIM_TIMEOUT_SECONDS),
                )
                row = cur.fetchone()
            conn.commit()

        if row is None:
            return None

        return dict(row)

    def complete_task(self, task_id: int) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE gitlab_task
                    SET
                        status = 'done',
                        last_error = NULL,
                        updated_at = now(),
                        finished_at = now()
                    WHERE id = %s
                    """,
                    (task_id,),
                )
            conn.commit()

    def fail_task(
        self, task_id: int, error: str, max_failures: int
    ) -> dict | None:
        with self.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT id, repo_id, failure_count
                    FROM gitlab_task
                    WHERE id = %s
                    FOR UPDATE
                    """,
                    (task_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None

                failure_count = int(row["failure_count"]) + 1
                status = "failed" if failure_count >= max_failures else "pending"
                cur.execute(
                    """
                    UPDATE gitlab_task
                    SET
                        status = %s,
                        failure_count = %s,
                        last_error = %s,
                        updated_at = now(),
                        started_at = NULL,
                        finished_at = CASE
                            WHEN %s = 'failed' THEN now()
                            ELSE finished_at
                        END
                    WHERE id = %s
                    """,
                    (status, failure_count, error, status, task_id),
                )
            conn.commit()

        return {
            "id": task_id,
            "repo_id": int(row["repo_id"]),
            "status": status,
            "failure_count": failure_count,
        }

    def settle_repo_activity(self, repo_id: int, max_failures: int) -> bool:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE repo
                    SET
                        left_activity_at = queued_left_activity_at,
                        right_activity_at = queued_right_activity_at,
                        updated_at = now()
                    WHERE id = %s
                      AND NOT EXISTS (
                        SELECT 1
                        FROM gitlab_task
                        WHERE repo_id = repo.id
                          AND status IN ('pending', 'running')
                          AND failure_count < %s
                      )
                    """,
                    (repo_id, max_failures),
                )
                updated = cur.rowcount > 0
            conn.commit()

        return updated

    def count_remaining_tasks(self, repo_id: int, max_failures: int) -> int:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM gitlab_task
                    WHERE repo_id = %s
                      AND status IN ('pending', 'running')
                      AND failure_count < %s
                    """,
                    (repo_id, max_failures),
                )
                row = cur.fetchone()

        return int(row[0]) if row is not None else 0

    def upsert_issue_graph(self, issue: Issue) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO issue (
                        repo_id,
                        iid,
                        title,
                        description,
                        url,
                        updated_at
                    )
                    VALUES (
                        %(repo_id)s,
                        %(iid)s,
                        %(title)s,
                        %(description)s,
                        %(url)s,
                        now()
                    )
                    ON CONFLICT (repo_id, iid)
                    DO UPDATE SET
                        title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        url = EXCLUDED.url,
                        updated_at = now()
                    """,
                    {
                        "repo_id": issue.repo_id,
                        "iid": issue.iid,
                        "title": issue.title,
                        "description": issue.description,
                        "url": issue.url,
                    },
                )
                cur.execute(
                    """
                    DELETE FROM issue_note
                    WHERE repo_id = %s AND issue_iid = %s
                    """,
                    (issue.repo_id, issue.iid),
                )
                for note in issue.notes:
                    cur.execute(
                        """
                        INSERT INTO issue_note (
                            repo_id,
                            issue_iid,
                            id,
                            body,
                            updated_at
                        )
                        VALUES (%s, %s, %s, %s, now())
                        """,
                        (issue.repo_id, issue.iid, note.id, note.body),
                    )
            conn.commit()

    def upsert_merge_request_graph(self, merge_request: MergeRequest) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO merge_request (
                        repo_id,
                        iid,
                        title,
                        description,
                        url,
                        updated_at
                    )
                    VALUES (
                        %(repo_id)s,
                        %(iid)s,
                        %(title)s,
                        %(description)s,
                        %(url)s,
                        now()
                    )
                    ON CONFLICT (repo_id, iid)
                    DO UPDATE SET
                        title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        url = EXCLUDED.url,
                        updated_at = now()
                    """,
                    {
                        "repo_id": merge_request.repo_id,
                        "iid": merge_request.iid,
                        "title": merge_request.title,
                        "description": merge_request.description,
                        "url": merge_request.url,
                    },
                )
                cur.execute(
                    """
                    DELETE FROM merge_request_note
                    WHERE repo_id = %s AND merge_request_iid = %s
                    """,
                    (merge_request.repo_id, merge_request.iid),
                )
                for note in merge_request.notes:
                    cur.execute(
                        """
                        INSERT INTO merge_request_note (
                            repo_id,
                            merge_request_iid,
                            id,
                            body,
                            updated_at
                        )
                        VALUES (%s, %s, %s, %s, now())
                        """,
                        (
                            merge_request.repo_id,
                            merge_request.iid,
                            note.id,
                            note.body,
                        ),
                    )
            conn.commit()

    def upsert_commit(self, commit: Commit) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                self._upsert_commit(cur, commit)
                self._replace_commit_edges(cur, commit)
            conn.commit()

    def upsert_commit_placeholder(
        self, repo_id: int, sha: str, url: str | None = None
    ) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                self._upsert_commit_placeholder(cur, repo_id, sha, url)
            conn.commit()

    def mark_commit_indexed(self, repo_id: int, sha: str) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                self._upsert_commit_placeholder(cur, repo_id, sha)
                cur.execute(
                    """
                    UPDATE git_commit
                    SET indexed_at = now(), updated_at = now()
                    WHERE repo_id = %s AND sha = %s
                    """,
                    (repo_id, sha),
                )
            conn.commit()

    def commit_needs_index(self, repo_id: int, sha: str) -> bool:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT indexed_at
                    FROM git_commit
                    WHERE repo_id = %s AND sha = %s
                    """,
                    (repo_id, sha),
                )
                row = cur.fetchone()

        if row is None:
            return True

        return row[0] is None

    def upsert_branch(self, repo_id: int, branch_name: str, head_sha: str) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                self._upsert_commit_placeholder(cur, repo_id, head_sha)
                cur.execute(
                    """
                    INSERT INTO repo_branch (
                        repo_id,
                        name,
                        head_sha,
                        is_deleted,
                        updated_at
                    )
                    VALUES (%s, %s, %s, FALSE, now())
                    ON CONFLICT (repo_id, name)
                    DO UPDATE SET
                        head_sha = EXCLUDED.head_sha,
                        is_deleted = FALSE,
                        updated_at = now()
                    """,
                    (repo_id, branch_name, head_sha),
                )
            conn.commit()

    def mark_branch_deleted(self, repo_id: int, branch_name: str) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO repo_branch (
                        repo_id,
                        name,
                        head_sha,
                        is_deleted,
                        updated_at
                    )
                    VALUES (%s, %s, NULL, TRUE, now())
                    ON CONFLICT (repo_id, name)
                    DO UPDATE SET
                        head_sha = NULL,
                        is_deleted = TRUE,
                        updated_at = now()
                    """,
                    (repo_id, branch_name),
                )
            conn.commit()

    def enqueue_branch_missing_commits(
        self, repo_id: int, branch_name: str
    ) -> list[str]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH RECURSIVE reachable(sha) AS (
                        SELECT head_sha
                        FROM repo_branch
                        WHERE repo_id = %s
                          AND name = %s
                          AND is_deleted = FALSE
                          AND head_sha IS NOT NULL
                        UNION
                        SELECT edge.parent_sha
                        FROM git_commit_edge AS edge
                        JOIN reachable
                          ON edge.repo_id = %s
                         AND edge.child_sha = reachable.sha
                    )
                    SELECT DISTINCT commit.sha
                    FROM reachable
                    JOIN git_commit AS commit
                      ON commit.repo_id = %s
                     AND commit.sha = reachable.sha
                    WHERE commit.indexed_at IS NULL
                    ORDER BY commit.sha
                    """,
                    (repo_id, branch_name, repo_id, repo_id),
                )
                rows = [row[0] for row in cur.fetchall()]
                for sha in rows:
                    self._enqueue_task(cur, repo_id, "commit", sha)
            conn.commit()

        return rows

    def replace_embeddings(
        self,
        repo_id: int,
        source_kind: str,
        source_key: str,
        records: list[dict],
    ) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM gitlab_embedding
                    WHERE repo_id = %s
                      AND source_kind = %s
                      AND source_key = %s
                    """,
                    (repo_id, source_kind, source_key),
                )

                for record in records:
                    vector_value = self._vector_value(record["embedding"])
                    cur.execute(
                        """
                        INSERT INTO gitlab_embedding (
                            repo_id,
                            source_kind,
                            source_key,
                            chunk_index,
                            locator_id,
                            content,
                            embedding,
                            updated_at
                        )
                        VALUES (
                            %(repo_id)s,
                            %(source_kind)s,
                            %(source_key)s,
                            %(chunk_index)s,
                            %(locator_id)s,
                            %(content)s,
                            %(embedding)s::vector,
                            now()
                        )
                        """,
                        {
                            "repo_id": record["repo_id"],
                            "source_kind": record["source_kind"],
                            "source_key": record["source_key"],
                            "chunk_index": record["chunk_index"],
                            "locator_id": record["locator_id"],
                            "content": record["content"],
                            "embedding": vector_value,
                        },
                    )
            conn.commit()

    def search_embeddings(
        self,
        query_embedding: list[float],
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict]:
        if not query_embedding:
            return []

        vector_value = self._vector_value(query_embedding)

        with self.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        repo.id AS repo_id,
                        repo.path AS repo_path,
                        repo.self_link,
                        repo.issues_link,
                        repo.merge_requests_link,
                        embedding.source_kind,
                        embedding.source_key,
                        embedding.chunk_index,
                        embedding.locator_id,
                        embedding.content,
                        issue.url AS issue_url,
                        merge_request.url AS merge_request_url,
                        git_commit.url AS commit_url,
                        1 - (embedding.embedding <=> %(embedding)s::vector) AS score
                    FROM gitlab_embedding AS embedding
                    JOIN repo ON repo.id = embedding.repo_id
                    LEFT JOIN issue
                      ON embedding.source_kind = 'issue'
                     AND issue.repo_id = embedding.repo_id
                     AND issue.iid::text = embedding.source_key
                    LEFT JOIN merge_request
                      ON embedding.source_kind = 'merge_request'
                     AND merge_request.repo_id = embedding.repo_id
                     AND merge_request.iid::text = embedding.source_key
                    LEFT JOIN git_commit
                      ON embedding.source_kind = 'commit'
                     AND git_commit.repo_id = embedding.repo_id
                     AND git_commit.sha = embedding.source_key
                    WHERE embedding.embedding IS NOT NULL
                    ORDER BY embedding.embedding <=> %(embedding)s::vector
                    LIMIT %(limit)s
                    OFFSET %(offset)s
                    """,
                    {
                        "embedding": vector_value,
                        "limit": limit,
                        "offset": offset,
                    },
                )
                rows = cur.fetchall()

        return [
            {
                "project_id": row["repo_id"],
                "project_path": row["repo_path"],
                "source_kind": row["source_kind"],
                "source_key": row["source_key"],
                "chunk_index": row["chunk_index"],
                "locator_id": row["locator_id"],
                "content": row["content"],
                "url": self._resource_url(dict(row)),
                "score": float(row["score"])
                if row["score"] is not None
                else None,
            }
            for row in rows
        ]

    def ensure_repo_in_cursor(self, cur, repo: dict) -> None:
        cur.execute(
            """
            INSERT INTO repo (
                id,
                path,
                self_link,
                issues_link,
                merge_requests_link,
                left_activity_at,
                right_activity_at,
                queued_left_activity_at,
                queued_right_activity_at,
                updated_at
            )
            VALUES (
                %(id)s,
                %(path)s,
                %(self_link)s,
                %(issues_link)s,
                %(merge_requests_link)s,
                %(left_activity_at)s,
                %(right_activity_at)s,
                %(queued_left_activity_at)s,
                %(queued_right_activity_at)s,
                now()
            )
            ON CONFLICT (id)
            DO UPDATE SET
                path = EXCLUDED.path,
                self_link = EXCLUDED.self_link,
                issues_link = EXCLUDED.issues_link,
                merge_requests_link = EXCLUDED.merge_requests_link,
                left_activity_at = EXCLUDED.left_activity_at,
                right_activity_at = EXCLUDED.right_activity_at,
                queued_left_activity_at = EXCLUDED.queued_left_activity_at,
                queued_right_activity_at = EXCLUDED.queued_right_activity_at,
                updated_at = now()
            """,
            {
                "id": repo["id"],
                "path": repo["path"],
                "self_link": repo.get("self_link"),
                "issues_link": repo.get("issues_link"),
                "merge_requests_link": repo.get("merge_requests_link"),
                "left_activity_at": repo["left_activity_at"],
                "right_activity_at": repo["right_activity_at"],
                "queued_left_activity_at": repo["queued_left_activity_at"],
                "queued_right_activity_at": repo["queued_right_activity_at"],
            },
        )

    def _enqueue_task(
        self,
        cur,
        repo_id: int,
        task_kind: str,
        task_key: str,
    ) -> None:
        cur.execute(
            """
            INSERT INTO gitlab_task (
                repo_id,
                task_kind,
                task_key,
                status,
                failure_count,
                last_error,
                created_at,
                updated_at,
                started_at,
                finished_at
            )
            VALUES (%s, %s, %s, 'pending', 0, NULL, now(), now(), NULL, NULL)
            ON CONFLICT (repo_id, task_kind, task_key)
            DO UPDATE SET
                status = CASE
                    WHEN gitlab_task.status IN ('done', 'failed')
                    THEN 'pending'
                    ELSE gitlab_task.status
                END,
                failure_count = CASE
                    WHEN gitlab_task.status IN ('done', 'failed')
                    THEN 0
                    ELSE gitlab_task.failure_count
                END,
                last_error = CASE
                    WHEN gitlab_task.status IN ('done', 'failed')
                    THEN NULL
                    ELSE gitlab_task.last_error
                END,
                created_at = CASE
                    WHEN gitlab_task.status IN ('done', 'failed')
                    THEN now()
                    ELSE gitlab_task.created_at
                END,
                updated_at = now(),
                started_at = CASE
                    WHEN gitlab_task.status IN ('done', 'failed')
                    THEN NULL
                    ELSE gitlab_task.started_at
                END,
                finished_at = CASE
                    WHEN gitlab_task.status IN ('done', 'failed')
                    THEN NULL
                    ELSE gitlab_task.finished_at
                END
            """,
            (repo_id, task_kind, task_key),
        )

    def _upsert_commit(self, cur, commit: Commit) -> None:
        cur.execute(
            """
            INSERT INTO git_commit (
                repo_id,
                sha,
                message,
                url,
                updated_at
            )
            VALUES (%s, %s, %s, %s, now())
            ON CONFLICT (repo_id, sha)
            DO UPDATE SET
                message = EXCLUDED.message,
                url = EXCLUDED.url,
                updated_at = now()
            """,
            (commit.repo_id, commit.sha, commit.message, commit.url),
        )

    def _upsert_commit_placeholder(
        self, cur, repo_id: int, sha: str, url: str | None = None
    ) -> None:
        cur.execute(
            """
            INSERT INTO git_commit (
                repo_id,
                sha,
                message,
                url,
                updated_at
            )
            VALUES (%s, %s, '', %s, now())
            ON CONFLICT (repo_id, sha)
            DO UPDATE SET
                url = COALESCE(EXCLUDED.url, git_commit.url),
                updated_at = now()
            """,
            (repo_id, sha, url),
        )

    def _replace_commit_edges(self, cur, commit: Commit) -> None:
        cur.execute(
            """
            DELETE FROM git_commit_edge
            WHERE repo_id = %s AND child_sha = %s
            """,
            (commit.repo_id, commit.sha),
        )

        for parent_sha in commit.parent_shas:
            self._upsert_commit_placeholder(cur, commit.repo_id, parent_sha)
            cur.execute(
                """
                INSERT INTO git_commit_edge (
                    repo_id,
                    child_sha,
                    parent_sha,
                    updated_at
                )
                VALUES (%s, %s, %s, now())
                ON CONFLICT (repo_id, child_sha, parent_sha)
                DO UPDATE SET updated_at = now()
                """,
                (commit.repo_id, commit.sha, parent_sha),
            )

    def _resource_url(self, row: dict) -> str | None:
        source_kind = row.get("source_kind")
        locator_id = row.get("locator_id")

        if source_kind == "issue":
            base = row.get("issue_url") or self._issue_url(row)
            if base and locator_id:
                return f"{base}#note_{locator_id}"
            return base

        if source_kind == "merge_request":
            base = row.get("merge_request_url") or self._merge_request_url(row)
            if base and locator_id:
                return f"{base}#note_{locator_id}"
            return base

        if source_kind == "commit":
            return row.get("commit_url") or self._commit_url(row)

        return None

    def _issue_url(self, row: dict) -> str | None:
        issues_link = row.get("issues_link")
        source_key = row.get("source_key")
        if not issues_link or not source_key:
            return None
        return f"{issues_link.rstrip('/')}/{source_key}"

    def _merge_request_url(self, row: dict) -> str | None:
        merge_requests_link = row.get("merge_requests_link")
        source_key = row.get("source_key")
        if not merge_requests_link or not source_key:
            return None
        return f"{merge_requests_link.rstrip('/')}/{source_key}"

    def _commit_url(self, row: dict) -> str | None:
        self_link = row.get("self_link")
        source_key = row.get("source_key")
        if not self_link or not source_key:
            return None
        return f"{self_link.rstrip('/')}/repository/commits/{source_key}"

    def _vector_value(self, embedding: list[float] | None) -> str | None:
        if not embedding:
            return None

        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(embedding)}"
            )

        return "[" + ",".join(str(value) for value in embedding) + "]"


database = Database(PG_HOST, PG_PASS)


def init_db() -> None:
    try:
        database.init_schema()
    except Exception:
        logger.exception("postgres init failed")
        raise


async def init_db_async() -> None:
    await asyncio.to_thread(init_db)


def ensure_repo(repo: dict) -> None:
    try:
        database.ensure_repo(repo)
    except Exception:
        logger.exception(
            "postgres repo ensure failed repo_id=%s path=%s",
            repo["id"],
            repo["path"],
        )
        raise


async def ensure_repo_async(repo: dict) -> None:
    await asyncio.to_thread(ensure_repo, repo)


def get_repo(repo_id: int) -> dict | None:
    try:
        return database.get_repo(repo_id)
    except Exception:
        logger.exception("postgres repo get failed repo_id=%s", repo_id)
        raise


async def get_repo_async(repo_id: int) -> dict | None:
    return await asyncio.to_thread(get_repo, repo_id)


def enqueue_project_tasks(repo: dict, tasks: dict[str, list[str]]) -> None:
    try:
        database.enqueue_project_tasks(repo, tasks)
    except Exception:
        logger.exception(
            "postgres task enqueue failed repo_id=%s path=%s",
            repo["id"],
            repo["path"],
        )
        raise


async def enqueue_project_tasks_async(
    repo: dict, tasks: dict[str, list[str]]
) -> None:
    await asyncio.to_thread(enqueue_project_tasks, repo, tasks)


def enqueue_task(repo_id: int, task_kind: str, task_key: str) -> None:
    try:
        database.enqueue_task(repo_id, task_kind, task_key)
    except Exception:
        logger.exception(
            "postgres task enqueue failed repo_id=%s task_kind=%s task_key=%s",
            repo_id,
            task_kind,
            task_key,
        )
        raise


async def enqueue_task_async(
    repo_id: int, task_kind: str, task_key: str
) -> None:
    await asyncio.to_thread(enqueue_task, repo_id, task_kind, task_key)


def claim_task(max_failures: int) -> dict | None:
    try:
        return database.claim_task(max_failures)
    except Exception:
        logger.exception("postgres task claim failed")
        raise


async def claim_task_async(max_failures: int) -> dict | None:
    return await asyncio.to_thread(claim_task, max_failures)


def complete_task(task_id: int) -> None:
    try:
        database.complete_task(task_id)
    except Exception:
        logger.exception("postgres task complete failed task_id=%s", task_id)
        raise


async def complete_task_async(task_id: int) -> None:
    await asyncio.to_thread(complete_task, task_id)


def fail_task(task_id: int, error: str, max_failures: int) -> dict | None:
    try:
        return database.fail_task(task_id, error, max_failures)
    except Exception:
        logger.exception("postgres task fail failed task_id=%s", task_id)
        raise


async def fail_task_async(
    task_id: int, error: str, max_failures: int
) -> dict | None:
    return await asyncio.to_thread(fail_task, task_id, error, max_failures)


def settle_repo_activity(repo_id: int, max_failures: int) -> bool:
    try:
        return database.settle_repo_activity(repo_id, max_failures)
    except Exception:
        logger.exception("postgres repo settle failed repo_id=%s", repo_id)
        raise


async def settle_repo_activity_async(
    repo_id: int, max_failures: int
) -> bool:
    return await asyncio.to_thread(settle_repo_activity, repo_id, max_failures)


def count_remaining_tasks(repo_id: int, max_failures: int) -> int:
    try:
        return database.count_remaining_tasks(repo_id, max_failures)
    except Exception:
        logger.exception(
            "postgres remaining task count failed repo_id=%s",
            repo_id,
        )
        raise


async def count_remaining_tasks_async(
    repo_id: int, max_failures: int
) -> int:
    return await asyncio.to_thread(count_remaining_tasks, repo_id, max_failures)


def upsert_issue_graph(issue: Issue) -> None:
    try:
        database.upsert_issue_graph(issue)
    except Exception:
        logger.exception(
            "postgres issue upsert failed repo_id=%s issue_iid=%s",
            issue.repo_id,
            issue.iid,
        )
        raise


async def upsert_issue_graph_async(issue: Issue) -> None:
    await asyncio.to_thread(upsert_issue_graph, issue)


def upsert_merge_request_graph(merge_request: MergeRequest) -> None:
    try:
        database.upsert_merge_request_graph(merge_request)
    except Exception:
        logger.exception(
            "postgres merge request upsert failed repo_id=%s merge_request_iid=%s",
            merge_request.repo_id,
            merge_request.iid,
        )
        raise


async def upsert_merge_request_graph_async(
    merge_request: MergeRequest,
) -> None:
    await asyncio.to_thread(upsert_merge_request_graph, merge_request)


def upsert_commit(commit: Commit) -> None:
    try:
        database.upsert_commit(commit)
    except Exception:
        logger.exception(
            "postgres commit upsert failed repo_id=%s sha=%s",
            commit.repo_id,
            commit.sha,
        )
        raise


async def upsert_commit_async(commit: Commit) -> None:
    await asyncio.to_thread(upsert_commit, commit)


def upsert_commit_placeholder(
    repo_id: int, sha: str, url: str | None = None
) -> None:
    try:
        database.upsert_commit_placeholder(repo_id, sha, url)
    except Exception:
        logger.exception(
            "postgres commit placeholder upsert failed repo_id=%s sha=%s",
            repo_id,
            sha,
        )
        raise


async def upsert_commit_placeholder_async(
    repo_id: int, sha: str, url: str | None = None
) -> None:
    await asyncio.to_thread(upsert_commit_placeholder, repo_id, sha, url)


def mark_commit_indexed(repo_id: int, sha: str) -> None:
    try:
        database.mark_commit_indexed(repo_id, sha)
    except Exception:
        logger.exception(
            "postgres commit index mark failed repo_id=%s sha=%s",
            repo_id,
            sha,
        )
        raise


async def mark_commit_indexed_async(repo_id: int, sha: str) -> None:
    await asyncio.to_thread(mark_commit_indexed, repo_id, sha)


def commit_needs_index(repo_id: int, sha: str) -> bool:
    try:
        return database.commit_needs_index(repo_id, sha)
    except Exception:
        logger.exception(
            "postgres commit index check failed repo_id=%s sha=%s",
            repo_id,
            sha,
        )
        raise


async def commit_needs_index_async(repo_id: int, sha: str) -> bool:
    return await asyncio.to_thread(commit_needs_index, repo_id, sha)


def upsert_branch(repo_id: int, branch_name: str, head_sha: str) -> None:
    try:
        database.upsert_branch(repo_id, branch_name, head_sha)
    except Exception:
        logger.exception(
            "postgres branch upsert failed repo_id=%s branch=%s",
            repo_id,
            branch_name,
        )
        raise


async def upsert_branch_async(
    repo_id: int, branch_name: str, head_sha: str
) -> None:
    await asyncio.to_thread(upsert_branch, repo_id, branch_name, head_sha)


def mark_branch_deleted(repo_id: int, branch_name: str) -> None:
    try:
        database.mark_branch_deleted(repo_id, branch_name)
    except Exception:
        logger.exception(
            "postgres branch delete mark failed repo_id=%s branch=%s",
            repo_id,
            branch_name,
        )
        raise


async def mark_branch_deleted_async(repo_id: int, branch_name: str) -> None:
    await asyncio.to_thread(mark_branch_deleted, repo_id, branch_name)


def enqueue_branch_missing_commits(repo_id: int, branch_name: str) -> list[str]:
    try:
        return database.enqueue_branch_missing_commits(repo_id, branch_name)
    except Exception:
        logger.exception(
            "postgres branch commit enqueue failed repo_id=%s branch=%s",
            repo_id,
            branch_name,
        )
        raise


async def enqueue_branch_missing_commits_async(
    repo_id: int, branch_name: str
) -> list[str]:
    return await asyncio.to_thread(
        enqueue_branch_missing_commits,
        repo_id,
        branch_name,
    )


def replace_embeddings(
    repo_id: int,
    source_kind: str,
    source_key: str,
    records: list[dict],
) -> None:
    try:
        database.replace_embeddings(repo_id, source_kind, source_key, records)
    except Exception:
        logger.exception(
            "postgres replace failed repo_id=%s source_kind=%s source_key=%s record_count=%s",
            repo_id,
            source_kind,
            source_key,
            len(records),
        )
        raise


async def replace_embeddings_async(
    repo_id: int,
    source_kind: str,
    source_key: str,
    records: list[dict],
) -> None:
    await asyncio.to_thread(
        replace_embeddings,
        repo_id,
        source_kind,
        source_key,
        records,
    )


def search_embeddings(
    query_embedding: list[float],
    limit: int = 10,
    offset: int = 0,
) -> list[dict]:
    try:
        return database.search_embeddings(query_embedding, limit, offset)
    except Exception:
        logger.exception(
            "postgres search failed limit=%s offset=%s",
            limit,
            offset,
        )
        raise


async def search_embeddings_async(
    query_embedding: list[float],
    limit: int = 10,
    offset: int = 0,
) -> list[dict]:
    return await asyncio.to_thread(
        search_embeddings,
        query_embedding,
        limit,
        offset,
    )
