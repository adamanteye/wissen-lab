import os
from dataclasses import dataclass, field


BATCH_SIZE_LIMIT = int(os.getenv("BATCH_SIZE_LIMIT", "8192"))


@dataclass
class ChunkMetadata:
    locator_id: str | None


@dataclass
class Chunk:
    content: str
    metadata: ChunkMetadata


@dataclass
class ChunkResult:
    repo_id: int
    id: str
    chunks: list[Chunk]


@dataclass
class IssueNote:
    id: int
    body: str


@dataclass
class Issue:
    repo_id: int
    iid: int
    title: str = ""
    description: str = ""
    url: str | None = None
    notes: list[IssueNote] = field(default_factory=list)

    @classmethod
    def from_gitlab(
        cls, repo_id: int, issue: dict, notes: list[dict] | None = None
    ):
        issue_notes = []

        for note in notes or []:
            if note.get("system"):
                continue

            body = (note.get("body") or "").strip()
            if not body:
                continue

            issue_notes.append(IssueNote(id=int(note["id"]), body=body))

        return cls(
            repo_id=repo_id,
            iid=int(issue["iid"]),
            title=(issue.get("title") or "").strip(),
            description=(issue.get("description") or "").strip(),
            url=issue.get("web_url"),
            notes=issue_notes,
        )


@dataclass
class MergeRequestNote:
    id: int
    body: str


@dataclass
class MergeRequest:
    repo_id: int
    iid: int
    title: str = ""
    description: str = ""
    url: str | None = None
    notes: list[MergeRequestNote] = field(default_factory=list)

    @classmethod
    def from_gitlab(
        cls, repo_id: int, merge_request: dict, notes: list[dict] | None = None
    ):
        merge_request_notes = []

        for note in notes or []:
            if note.get("system"):
                continue

            body = (note.get("body") or "").strip()
            if not body:
                continue

            merge_request_notes.append(
                MergeRequestNote(id=int(note["id"]), body=body)
            )

        return cls(
            repo_id=repo_id,
            iid=int(merge_request["iid"]),
            title=(merge_request.get("title") or "").strip(),
            description=(merge_request.get("description") or "").strip(),
            url=merge_request.get("web_url"),
            notes=merge_request_notes,
        )


@dataclass
class CommitDiff:
    path: str
    diff: str


@dataclass
class Commit:
    repo_id: int
    sha: str
    message: str = ""
    url: str | None = None
    parent_shas: list[str] = field(default_factory=list)
    diffs: list[CommitDiff] = field(default_factory=list)

    @classmethod
    def from_gitlab(
        cls,
        repo_id: int,
        commit: dict,
        diffs: list[dict] | None = None,
        url: str | None = None,
    ):
        commit_diffs = []
        seen = set()
        for diff in diffs or []:
            changed_file = diff.get("new_path") or diff.get("old_path")
            if not changed_file or changed_file in seen:
                continue
            seen.add(changed_file)
            raw_diff = (diff.get("diff") or "").strip()
            if not raw_diff:
                continue
            commit_diffs.append(CommitDiff(path=changed_file, diff=raw_diff))

        return cls(
            repo_id=repo_id,
            sha=str(commit["id"]),
            message=(commit.get("message") or "").strip(),
            url=url,
            parent_shas=[
                str(parent_sha)
                for parent_sha in commit.get("parent_ids") or []
                if parent_sha
            ],
            diffs=commit_diffs,
        )


class ChunkBuilder:
    def _chunk(
        self,
        content: str,
        locator_id: str | None = None,
        split_locator_prefix: str | None = None,
    ) -> list[Chunk]:
        content = content.strip()
        if not content:
            return []

        if len(content) <= BATCH_SIZE_LIMIT:
            return [
                Chunk(
                    content=content,
                    metadata=ChunkMetadata(locator_id=locator_id),
                )
            ]

        parts = []
        base_locator = locator_id or split_locator_prefix or "chunk"
        for index, start in enumerate(range(0, len(content), BATCH_SIZE_LIMIT)):
            parts.append(
                Chunk(
                    content=content[start : start + BATCH_SIZE_LIMIT],
                    metadata=ChunkMetadata(
                        locator_id=f"{base_locator}-batch{index}"
                    ),
                )
            )
        return parts


class IssueChunkBuilder(ChunkBuilder):
    def build(self, issue: Issue) -> ChunkResult:
        chunks = []

        chunks.extend(
            self._chunk(
                self._root_content(issue),
                split_locator_prefix="root",
            )
        )

        for note in issue.notes:
            chunks.extend(
                self._chunk(
                    note.body,
                    str(note.id),
                    split_locator_prefix=str(note.id),
                )
            )

        return ChunkResult(
            repo_id=issue.repo_id,
            id=str(issue.iid),
            chunks=chunks,
        )

    def _root_content(self, issue: Issue) -> str:
        if issue.title and issue.description:
            return f"{issue.title}\n\n{issue.description}"
        return issue.title or issue.description


class CommitChunkBuilder(ChunkBuilder):
    def build(self, commit: Commit) -> ChunkResult:
        chunks = []

        chunks.extend(
            self._chunk(
                commit.message,
                split_locator_prefix="message",
            )
        )

        for diff in commit.diffs:
            chunks.extend(
                self._chunk(
                    diff.diff,
                    diff.path,
                    split_locator_prefix=diff.path,
                )
            )

        return ChunkResult(
            repo_id=commit.repo_id,
            id=commit.sha,
            chunks=chunks,
        )


class MergeRequestChunkBuilder(ChunkBuilder):
    def build(self, merge_request: MergeRequest) -> ChunkResult:
        chunks = []

        chunks.extend(
            self._chunk(
                self._root_content(merge_request),
                split_locator_prefix="root",
            )
        )

        for note in merge_request.notes:
            chunks.extend(
                self._chunk(
                    note.body,
                    str(note.id),
                    split_locator_prefix=str(note.id),
                )
            )

        return ChunkResult(
            repo_id=merge_request.repo_id,
            id=str(merge_request.iid),
            chunks=chunks,
        )

    def _root_content(self, merge_request: MergeRequest) -> str:
        if merge_request.title and merge_request.description:
            return f"{merge_request.title}\n\n{merge_request.description}"
        return merge_request.title or merge_request.description
