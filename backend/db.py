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
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (repo_id, sha)
);

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
                        now()
                    )
                    ON CONFLICT (id)
                    DO UPDATE SET
                        path = EXCLUDED.path,
                        self_link = EXCLUDED.self_link,
                        issues_link = EXCLUDED.issues_link,
                        merge_requests_link = EXCLUDED.merge_requests_link,
                        updated_at = now()
                    """,
                    {
                        "id": repo["id"],
                        "path": repo["path"],
                        "self_link": repo.get("self_link"),
                        "issues_link": repo.get("issues_link"),
                        "merge_requests_link": repo.get("merge_requests_link"),
                        "left_activity_at": repo.get("left_activity_at", EPOCH),
                        "right_activity_at": repo.get(
                            "right_activity_at", EPOCH
                        ),
                    },
                )
            conn.commit()

    def update_repo_state(self, repo: dict) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE repo
                    SET
                        path = %(path)s,
                        self_link = %(self_link)s,
                        issues_link = %(issues_link)s,
                        merge_requests_link = %(merge_requests_link)s,
                        left_activity_at = %(left_activity_at)s,
                        right_activity_at = %(right_activity_at)s,
                        updated_at = now()
                    WHERE id = %(id)s
                    """,
                    {
                        "id": repo["id"],
                        "path": repo["path"],
                        "self_link": repo.get("self_link"),
                        "issues_link": repo.get("issues_link"),
                        "merge_requests_link": repo.get("merge_requests_link"),
                        "left_activity_at": repo["left_activity_at"],
                        "right_activity_at": repo["right_activity_at"],
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
            conn.commit()

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
                "repo_id": row["repo_id"],
                "repo_path": row["repo_path"],
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


def update_repo_state(repo: dict) -> None:
    try:
        database.update_repo_state(repo)
    except Exception:
        logger.exception(
            "postgres repo state update failed repo_id=%s path=%s",
            repo["id"],
            repo["path"],
        )
        raise


async def update_repo_state_async(repo: dict) -> None:
    await asyncio.to_thread(update_repo_state, repo)


def get_repo(repo_id: int) -> dict | None:
    try:
        return database.get_repo(repo_id)
    except Exception:
        logger.exception("postgres repo get failed repo_id=%s", repo_id)
        raise


async def get_repo_async(repo_id: int) -> dict | None:
    return await asyncio.to_thread(get_repo, repo_id)


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
