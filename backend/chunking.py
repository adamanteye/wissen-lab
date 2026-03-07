from dataclasses import dataclass, field


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
            diffs=commit_diffs,
        )


class ChunkBuilder:
    def _chunk(
        self, content: str, locator_id: str | None = None
    ) -> Chunk | None:
        content = content.strip()
        if not content:
            return None

        return Chunk(
            content=content,
            metadata=ChunkMetadata(locator_id=locator_id),
        )


class IssueChunkBuilder(ChunkBuilder):
    def build(self, issue: Issue) -> ChunkResult:
        chunks = []

        root = self._chunk(self._root_content(issue))
        if root:
            chunks.append(root)

        for note in issue.notes:
            chunk = self._chunk(note.body, str(note.id))
            if chunk:
                chunks.append(chunk)

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

        message_chunk = self._chunk(commit.message)
        if message_chunk:
            chunks.append(message_chunk)

        for diff in commit.diffs:
            chunk = self._chunk(diff.diff, diff.path)
            if chunk:
                chunks.append(chunk)

        return ChunkResult(
            repo_id=commit.repo_id,
            id=commit.sha,
            chunks=chunks,
        )


class MergeRequestChunkBuilder(ChunkBuilder):
    def build(self, merge_request: MergeRequest) -> ChunkResult:
        chunks = []

        root = self._chunk(self._root_content(merge_request))
        if root:
            chunks.append(root)

        for note in merge_request.notes:
            chunk = self._chunk(note.body, str(note.id))
            if chunk:
                chunks.append(chunk)

        return ChunkResult(
            repo_id=merge_request.repo_id,
            id=str(merge_request.iid),
            chunks=chunks,
        )

    def _root_content(self, merge_request: MergeRequest) -> str:
        if merge_request.title and merge_request.description:
            return f"{merge_request.title}\n\n{merge_request.description}"
        return merge_request.title or merge_request.description
