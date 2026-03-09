# Wissen Lab

GitLab fact indexer.

## What It Does

- Reconcile GitLab projects.
- Read issues, merge requests, commits, and notes.
- Split long text into chunks.
- Embed chunks with llama-server.
- Store vectors in Postgres with pgvector.
- Search by text.
- Return content with source links.
- Serve a small search frontend.

## Environment

- `PG_HOST`: Postgres host.
- `PG_PASS`: Postgres password.
- `GITLAB_URL`: GitLab base URL.
- `GITLAB_API_TOKEN`: GitLab API token.
- `LLAMA_SERVER_URL`: llama-server URL.
- `LLAMA_SERVER_API_KEY`: llama-server API key.
- `LLAMA_SERVER_TIMEOUT_SECONDS`: Timeout for each embedding request. Default `30`.
- `TASK_MAX_FAILURES`: Max task failures before the reconciler treats the task as settled and advances project activity windows. Default `3`.
- `BATCH_SIZE_LIMIT`: Max chars per chunk before split. Default `8192`.

## `/search` Example Response

Example request:

```sh
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  --data '{
    "content": "mail notification config",
    "page": 1,
    "page_size": 10
  }'
```

Each result has `url` and `content`.

```json
[
  {
    "project_id": 6,
    "project_path": "infra/wissen-lab",
    "source_kind": "issue",
    "source_key": "42",
    "chunk_index": 0,
    "locator_id": null,
    "content": "Mail notification configuration is currently driven by the SMTP relay host and the sender address configured in deployment.",
    "url": "https://gitlab.example.com/infra/wissen-lab/-/issues/42",
    "score": 0.9134
  },
  {
    "project_id": 6,
    "project_path": "infra/wissen-lab",
    "source_kind": "merge_request",
    "source_key": "17",
    "chunk_index": 2,
    "locator_id": "9812",
    "content": "Please keep the mail sender configurable through env so staging and production can use different identities.",
    "url": "https://gitlab.example.com/infra/wissen-lab/-/merge_requests/17#note_9812",
    "score": 0.8871
  },
  {
    "project_id": 6,
    "project_path": "infra/wissen-lab",
    "source_kind": "commit",
    "source_key": "367610302658f4288181a346a4b509e2eda69ede",
    "chunk_index": 0,
    "locator_id": null,
    "content": "配置邮件提醒.md: 更新",
    "url": "https://gitlab.example.com/infra/wissen-lab/-/commit/367610302658f4288181a346a4b509e2eda69ede",
    "score": 0.8619
  }
]
```
