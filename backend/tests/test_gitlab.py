import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gitlab import GitLabClient


class GitLabProjectTests(TestCase):
    def test_list_projects_prioritizes_recent_activity(self):
        client = GitLabClient("https://gitlab.example.com", "token")
        client._get_all = Mock(return_value=[])

        client.list_projects()

        client._get_all.assert_called_once_with(
            "/api/v4/projects",
            params={
                "order_by": "last_activity_at",
                "sort": "desc",
                "per_page": 100,
            },
        )

    def test_inventory_returns_sorted_resource_keys(self):
        client = GitLabClient("https://gitlab.example.com", "token")
        client.get_project = Mock(return_value={"id": 451})
        client._get_all = Mock(
            side_effect=[
                [{"name": "topic"}, {"name": "main"}],
                [{"iid": 12}, {"iid": 2}],
                [{"iid": 9}, {"iid": 3}],
            ]
        )

        inventory = client.get_project_inventory("451")

        self.assertEqual(
            {
                "branches": ["main", "topic"],
                "issues": ["2", "12"],
                "merge_requests": ["3", "9"],
            },
            inventory,
        )

    def test_list_branch_commit_shas_returns_complete_history(self):
        client = GitLabClient("https://gitlab.example.com", "token")
        client._get_all = Mock(
            return_value=[{"id": "head-sha"}, {"id": "parent-sha"}]
        )

        shas = client.list_branch_commit_shas("451", "main")

        self.assertEqual(["head-sha", "parent-sha"], shas)
        client._get_all.assert_called_once_with(
            "/api/v4/projects/451/repository/commits",
            params={"ref_name": "main", "per_page": 100},
        )


class GitLabEventWindowTests(TestCase):
    def test_same_day_events_are_retrieved_with_date_overlap(self):
        client = GitLabClient("https://gitlab.example.com", "token")
        client.get_project = Mock(return_value={"id": 451})
        client._get_all = Mock(
            return_value=[
                {
                    "id": 1,
                    "created_at": "2026-08-16T08:48:33.333Z",
                },
                {
                    "id": 2,
                    "created_at": "2026-08-16T10:53:45.170Z",
                },
            ]
        )

        events = client.list_project_events_window(
            "451",
            after="2026-08-16T08:48:33.333Z",
        )

        self.assertEqual([2], [event["id"] for event in events])
        client._get_all.assert_called_once_with(
            "/api/v4/projects/451/events",
            params={
                "after": "2026-08-15",
                "before": None,
                "sort": "asc",
                "per_page": 100,
            },
        )

    def test_exact_before_and_after_bounds_are_applied_locally(self):
        client = GitLabClient("https://gitlab.example.com", "token")
        client.get_project = Mock(return_value={"id": 451})
        client._get_all = Mock(
            return_value=[
                {"id": 1, "created_at": "2026-08-16T08:00:00Z"},
                {"id": 2, "created_at": "2026-08-16T09:00:00Z"},
                {"id": 3, "created_at": "2026-08-16T10:00:00Z"},
            ]
        )

        events = client.list_project_events_window(
            "451",
            after="2026-08-16T08:00:00Z",
            before="2026-08-16T10:00:00Z",
        )

        self.assertEqual([2], [event["id"] for event in events])
        self.assertEqual(
            {
                "after": "2026-08-15",
                "before": "2026-08-17",
                "sort": "asc",
                "per_page": 100,
            },
            client._get_all.call_args.kwargs["params"],
        )
