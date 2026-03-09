import os
from urllib.parse import quote

import requests

from chunking import Commit, Issue, MergeRequest
from logutil import get_logger


GITLAB_URL = os.getenv("GITLAB_URL", "").rstrip("/")
GITLAB_API_TOKEN = os.getenv("GITLAB_API_TOKEN", "")
logger = get_logger("gitlab")


class GitLabNotFoundError(Exception):
    def __init__(self, url: str, params: dict | None = None) -> None:
        self.url = url
        self.params = params
        super().__init__(
            f"gitlab resource not found url={url} params={params}"
        )


class GitLabClient:
    def __init__(
        self, base_url: str = GITLAB_URL, api_token: str = GITLAB_API_TOKEN
    ):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token

    def get_project(self, project_id: str | int) -> dict:
        path = f"/api/v4/projects/{quote(str(project_id), safe='')}"
        return self._get(path)

    def list_projects(self) -> list[dict]:
        return self._get_all(
            "/api/v4/projects",
            params={
                "order_by": "id",
                "sort": "asc",
                "per_page": 100,
            },
        )

    def get_issue(self, project_id: str | int, issue_iid: str | int) -> dict:
        path = (
            f"/api/v4/projects/{quote(str(project_id), safe='')}"
            f"/issues/{issue_iid}"
        )
        return self._get(path)

    def list_issue_notes(
        self, project_id: str | int, issue_iid: int
    ) -> list[dict]:
        path = (
            f"/api/v4/projects/{quote(str(project_id), safe='')}"
            f"/issues/{issue_iid}/notes"
        )
        return self._get_all(
            path,
            params={
                "activity_filter": "only_comments",
                "order_by": "created_at",
                "sort": "asc",
                "per_page": 100,
            },
        )

    def get_issue_object(
        self, project_id: str | int, issue_iid: str | int
    ) -> Issue:
        project = self.get_project(project_id)
        issue = self.get_issue(project["id"], issue_iid)
        notes = self.list_issue_notes(project["id"], int(issue["iid"]))
        return Issue.from_gitlab(int(project["id"]), issue, notes)

    def get_merge_request(
        self, project_id: str | int, merge_request_iid: str | int
    ) -> dict:
        path = (
            f"/api/v4/projects/{quote(str(project_id), safe='')}"
            f"/merge_requests/{merge_request_iid}"
        )
        return self._get(path)

    def list_merge_request_notes(
        self, project_id: str | int, merge_request_iid: int
    ) -> list[dict]:
        path = (
            f"/api/v4/projects/{quote(str(project_id), safe='')}"
            f"/merge_requests/{merge_request_iid}/notes"
        )
        return self._get_all(
            path,
            params={
                "activity_filter": "only_comments",
                "order_by": "created_at",
                "sort": "asc",
                "per_page": 100,
            },
        )

    def get_merge_request_object(
        self, project_id: str | int, merge_request_iid: str | int
    ) -> MergeRequest:
        project = self.get_project(project_id)
        merge_request = self.get_merge_request(project["id"], merge_request_iid)
        notes = self.list_merge_request_notes(
            project["id"], int(merge_request["iid"])
        )
        return MergeRequest.from_gitlab(
            int(project["id"]),
            merge_request,
            notes,
        )

    def get_commit(self, project_id: str | int, sha: str) -> dict:
        path = (
            f"/api/v4/projects/{quote(str(project_id), safe='')}"
            f"/repository/commits/{quote(sha, safe='')}"
        )
        return self._get(path)

    def get_branch(self, project_id: str | int, branch_name: str) -> dict:
        path = (
            f"/api/v4/projects/{quote(str(project_id), safe='')}"
            f"/repository/branches/{quote(branch_name, safe='')}"
        )
        return self._get(path)

    def list_commit_diffs(self, project_id: str | int, sha: str) -> list[dict]:
        path = (
            f"/api/v4/projects/{quote(str(project_id), safe='')}"
            f"/repository/commits/{quote(sha, safe='')}/diff"
        )
        diffs = self._get(path)
        if isinstance(diffs, list):
            return diffs
        return []

    def get_commit_object(self, project_id: str | int, sha: str) -> Commit:
        project = self.get_project(project_id)
        commit = self.get_commit(project["id"], sha)
        diffs = self.list_commit_diffs(project["id"], sha)
        return Commit.from_gitlab(
            int(project["id"]),
            commit,
            diffs,
            url=self._commit_url(project, sha),
        )

    def compare_commits(
        self, project_id: str | int, from_sha: str, to_sha: str
    ) -> dict:
        path = (
            f"/api/v4/projects/{quote(str(project_id), safe='')}"
            "/repository/compare"
        )
        return self._get(
            path,
            params={
                "from": from_sha,
                "to": to_sha,
                "straight": "true",
            },
        )

    def list_project_events_window(
        self,
        project_id: str | int,
        after: str | None = None,
        before: str | None = None,
    ) -> list[dict]:
        project = self.get_project(project_id)
        return self._get_all(
            f"/api/v4/projects/{quote(str(project['id']), safe='')}/events",
            params={
                "after": after,
                "before": before,
                "sort": "asc",
                "per_page": 100,
            },
        )

    def get_repo_state(self, project_id: str | int) -> dict:
        project = self.get_project(project_id)
        links = project.get("_links") or {}
        return {
            "id": int(project["id"]),
            "path": project["path_with_namespace"],
            "self_link": links.get("self"),
            "issues_link": links.get("issues"),
            "merge_requests_link": links.get("merge_requests"),
            "last_activity_at": project["last_activity_at"],
        }

    def _commit_url(self, project: dict, sha: str) -> str | None:
        web_url = (project.get("web_url") or "").rstrip("/")
        if not web_url:
            return None
        return f"{web_url}/-/commit/{sha}"

    def _get(self, path: str, params: dict | None = None):
        url = f"{self.base_url}{path}"
        try:
            response = requests.get(
                url,
                headers=self._headers(),
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.warning(
                    "gitlab resource not found url=%s params=%s",
                    url,
                    params,
                )
                raise GitLabNotFoundError(url, params) from exc

            logger.exception(
                "gitlab request failed url=%s params=%s", url, params
            )
            raise
        except requests.RequestException:
            logger.exception(
                "gitlab request failed url=%s params=%s", url, params
            )
            raise
        except ValueError:
            logger.exception("gitlab returned invalid json url=%s", url)
            raise

    def _get_all(self, path: str, params: dict | None = None) -> list[dict]:
        page = 1
        items = []
        params = dict(params or {})

        while True:
            page_params = dict(params)
            page_params["page"] = page
            batch = self._get(path, params=page_params)
            if not batch:
                break

            items.extend(batch)
            if len(batch) < page_params.get("per_page", 100):
                break

            page += 1

        return items

    def _headers(self) -> dict:
        headers = {"Accept": "application/json"}
        if self.api_token:
            headers["PRIVATE-TOKEN"] = self.api_token
        return headers
