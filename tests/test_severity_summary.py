import unittest

from mcp_code_review.review import ReviewResult
from mcp_code_review.server import (
    count_tagged_issue_runs,
    format_completion_summary,
    format_elapsed_time,
    format_severity_summary,
)


class TestSeveritySummary(unittest.TestCase):
    def test_format_severity_summary_aggregates_counts(self):
        results = [
            ReviewResult(
                verdict="comment",
                summary="first",
                comment_markdown="- [P1] one\n- [P1] two\n- [P1] three\n- [P2] four",
            ),
            ReviewResult(
                verdict="comment",
                summary="second",
                comment_markdown="notes [p3]",
            ),
            ReviewResult(
                verdict="pass",
                summary="clean",
                comment_markdown="- [P0] ignored",
            ),
        ]
        self.assertEqual(format_severity_summary(results), " (P0=1, P1=3, P2=1, P3=1)")

    def test_format_severity_summary_empty(self):
        results = [ReviewResult(verdict="comment", summary="x", comment_markdown="no tags")]
        self.assertEqual(format_severity_summary(results), "")

    def test_count_tagged_issue_runs_by_tags(self):
        results = [
            ReviewResult(verdict="comment", summary="none", comment_markdown="no tags"),
            ReviewResult(verdict="pass", summary="tagged", comment_markdown="- [P1] found"),
            ReviewResult(verdict="comment", summary="multi", comment_markdown="- [P2]\n- [P3]"),
            ReviewResult(verdict="comment", summary="dup", comment_markdown="- [P2]\n- [P2]"),
        ]
        self.assertEqual(count_tagged_issue_runs(results), 3)

    def test_format_elapsed_time_seconds_only(self):
        self.assertEqual(format_elapsed_time(12), "12s")

    def test_format_elapsed_time_minutes_seconds(self):
        self.assertEqual(format_elapsed_time(72), "1m 12s")

    def test_format_completion_summary_includes_severity_and_time(self):
        summary = format_completion_summary(3, 1, " (P1=1)", 72)
        self.assertEqual(
            summary,
            "completed 3 review(s); 1 run(s) reported tagged issues (P1=1); took 1m 12s",
        )

    def test_format_completion_summary_without_severity(self):
        summary = format_completion_summary(1, 0, "", 12)
        self.assertEqual(
            summary,
            "completed 1 review(s); 0 run(s) reported tagged issues; took 12s",
        )

    def test_format_completion_summary_appends_archive_failure_suffix(self):
        summary = format_completion_summary(1, 0, "", 12, archive_failed=True)
        self.assertEqual(
            summary,
            "completed 1 review(s); 0 run(s) reported tagged issues; took 12s. "
            "thread archive failed",
        )


if __name__ == "__main__":
    unittest.main()
