import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock
from unittest.mock import patch

from mcp_code_review.app_server import (
    AppServerError,
    BANNED_APP_SERVER_FEATURES,
    AppServerClient,
    CodexReviewProfile,
    MCP_DISABLED_ERROR_TEXT,
    MCP_SERVER_LIST_ATTEMPTS,
    _list_mcp_server_names_once,
    best_effort_archive_thread,
    build_app_server_command,
    is_mcp_tool_call_notification,
    list_mcp_server_names_with_retry,
    load_codex_review_profile,
    parse_mcp_server_names,
    profile_config_assignments,
    run_reviews,
    run_reviews_threads,
    run_single_review,
)
from mcp_code_review.spawn_guard import SPAWN_GUARD_CONFIG_ASSIGNMENT


class TestAppServerSpawnConfig(unittest.TestCase):
    def test_parse_mcp_server_names_deduplicates(self):
        payload = """
        [
          {"name": "review"},
          {"name": "serena"},
          {"name": "review"},
          {"name": 123},
          {}
        ]
        """
        self.assertEqual(parse_mcp_server_names(payload), ["review", "serena"])

    def test_parse_mcp_server_names_rejects_invalid_json(self):
        with self.assertRaises(AppServerError):
            parse_mcp_server_names("{not json")

    def test_build_app_server_command_disables_mcp_servers_and_sets_guard(self):
        command = build_app_server_command(
            codex_bin="codex",
            mcp_server_names=["review", "serena"],
            config_assignments=[
                'model_reasoning_effort="xhigh"',
                'model_reasoning_summary="auto"',
            ],
        )

        self.assertEqual(command[0], "codex")
        self.assertEqual(command[-1], "app-server")
        self.assertNotIn("profile=review", command)
        self.assertIn('model_reasoning_effort="xhigh"', command)
        self.assertIn('model_reasoning_summary="auto"', command)
        self.assertEqual(
            BANNED_APP_SERVER_FEATURES,
            (
                "apps",
                "collab",
                "multi_agent",
                "multi_agent_v2",
                "memory_tool",
                "plugins",
                "spawn_csv",
                "enable_fanout",
            ),
        )
        for feature in BANNED_APP_SERVER_FEATURES:
            self.assertIn(f"features.{feature}=false", command)
        self.assertIn(SPAWN_GUARD_CONFIG_ASSIGNMENT, command)
        self.assertIn("mcp_servers.review.enabled=false", command)
        self.assertIn("mcp_servers.serena.enabled=false", command)

    def test_is_mcp_tool_call_notification_variants(self):
        self.assertTrue(
            is_mcp_tool_call_notification({"method": "item/mcpToolCall/progress", "params": {}})
        )
        self.assertTrue(
            is_mcp_tool_call_notification(
                {"method": "event/msg", "params": {"msg": {"type": "mcp_tool_call_begin"}}}
            )
        )
        self.assertTrue(
            is_mcp_tool_call_notification(
                {"method": "item/completed", "params": {"item": {"type": "mcpToolCall"}}}
            )
        )
        self.assertFalse(
            is_mcp_tool_call_notification(
                {"method": "item/completed", "params": {"item": {"type": "exitedReviewMode"}}}
            )
        )


class TestMcpServerNameRetry(unittest.IsolatedAsyncioTestCase):
    async def test_list_mcp_server_names_once_builds_configured_command(self):
        captured_command = None

        class Process:
            returncode = 0

            async def communicate(self):
                return b'[{"name": "review"}]', b""

        async def create_subprocess_exec(*command, **_kwargs):
            nonlocal captured_command
            captured_command = command
            return Process()

        with patch("asyncio.create_subprocess_exec", create_subprocess_exec):
            names = await _list_mcp_server_names_once(
                "codex",
                config_assignments=["features.plugins=false"],
            )

        self.assertEqual(names, ["review"])
        self.assertEqual(
            captured_command,
            (
                "codex",
                "-c",
                "features.plugins=false",
                "mcp",
                "list",
                "--json",
            ),
        )

    async def test_list_mcp_server_names_applies_config_assignments(self):
        list_once = AsyncMock(return_value=["review"])
        config_assignments = [
            'model_reasoning_effort="xhigh"',
            "features.plugins=false",
        ]

        with patch("mcp_code_review.app_server._list_mcp_server_names_once", list_once):
            names = await list_mcp_server_names_with_retry(
                "codex",
                config_assignments=config_assignments,
            )

        self.assertEqual(names, ["review"])
        list_once.assert_awaited_once_with(
            "codex",
            env=None,
            config_assignments=config_assignments,
        )

    async def test_list_mcp_server_names_retries_once(self):
        list_once = AsyncMock(side_effect=[AppServerError("transient"), ["review"]])
        with patch("mcp_code_review.app_server._list_mcp_server_names_once", list_once):
            names = await list_mcp_server_names_with_retry("codex")

        self.assertEqual(names, ["review"])
        self.assertEqual(list_once.await_count, 2)

    async def test_list_mcp_server_names_fails_after_retry(self):
        list_once = AsyncMock(
            side_effect=[AppServerError("first"), AppServerError("second")]
        )
        with patch("mcp_code_review.app_server._list_mcp_server_names_once", list_once):
            with self.assertRaises(AppServerError):
                await list_mcp_server_names_with_retry("codex")

        self.assertEqual(list_once.await_count, MCP_SERVER_LIST_ATTEMPTS)


class TestCodexProfileConfig(unittest.TestCase):
    def test_load_codex_review_profile_from_codex_home(self):
        with TemporaryDirectory() as codex_home:
            config_path = Path(codex_home) / "config.toml"
            config_path.write_text(
                """
                [profiles.review]
                model = "gpt-5.5"
                model_provider = "openai"
                service_tier = "priority"
                model_reasoning_effort = "xhigh"
                model_reasoning_summary = "auto"
                """,
                encoding="utf-8",
            )

            profile = load_codex_review_profile(
                "review",
                env={"CODEX_HOME": codex_home},
            )

        self.assertEqual(profile.model, "gpt-5.5")
        self.assertEqual(profile.model_provider, "openai")
        self.assertEqual(profile.service_tier, "priority")
        self.assertEqual(profile.model_reasoning_effort, "xhigh")
        self.assertEqual(profile.model_reasoning_summary, "auto")
        self.assertEqual(
            profile_config_assignments(profile),
            [
                'model_reasoning_effort="xhigh"',
                'model_reasoning_summary="auto"',
            ],
        )

    def test_load_codex_review_profile_fails_for_missing_profile(self):
        with TemporaryDirectory() as codex_home:
            config_path = Path(codex_home) / "config.toml"
            config_path.write_text(
                """
                [profiles.other]
                model = "gpt-5"
                """,
                encoding="utf-8",
            )

            with self.assertRaisesRegex(AppServerError, "Codex profile 'review' not found"):
                load_codex_review_profile("review", env={"CODEX_HOME": codex_home})

    def test_load_codex_review_profile_rejects_non_string_supported_fields(self):
        with TemporaryDirectory() as codex_home:
            config_path = Path(codex_home) / "config.toml"
            config_path.write_text(
                """
                [profiles.review]
                model = 123
                """,
                encoding="utf-8",
            )

            with self.assertRaisesRegex(AppServerError, "field 'model'.*must be a string"):
                load_codex_review_profile("review", env={"CODEX_HOME": codex_home})


class TestRunReviewsFallback(unittest.IsolatedAsyncioTestCase):
    async def test_profile_defaults_apply_to_thread_reviews_with_explicit_arg_precedence(self):
        client = AsyncMock()
        review = AsyncMock(return_value="thread-result")
        settings = CodexReviewProfile(
            model="profile-model",
            model_provider="profile-provider",
            service_tier="priority",
            model_reasoning_effort="xhigh",
        )

        with (
            patch(
                "mcp_code_review.app_server.AppServerClient.start",
                new=AsyncMock(return_value=client),
            ) as start,
            patch("mcp_code_review.app_server.run_single_review", new=review),
        ):
            result = await run_reviews_threads(
                "codex",
                "/tmp/repo",
                1,
                30,
                model="explicit-model",
                model_provider="explicit-provider",
                profile_settings=settings,
            )

        self.assertEqual(result, ["thread-result"])
        start.assert_awaited_once_with(
            "codex",
            config_assignments=['model_reasoning_effort="xhigh"'],
        )
        review.assert_awaited_once_with(
            client,
            "/tmp/repo",
            30,
            model="explicit-model",
            model_provider="explicit-provider",
            service_tier="priority",
            additional_developer_instructions=None,
        )

    async def test_auto_falls_back_to_process_mode_on_thread_failure(self):
        with (
            patch("mcp_code_review.app_server._PREFER_PROCESS_MODE", False),
            patch(
                "mcp_code_review.app_server.run_reviews_threads",
                new=AsyncMock(side_effect=AppServerError("codex app-server closed stdout")),
            ) as run_threads,
            patch(
                "mcp_code_review.app_server.run_reviews_processes",
                new=AsyncMock(return_value=["process-result"]),
            ) as run_processes,
        ):
            result = await run_reviews("codex", "/tmp/repo", 2, 30, "auto")

        self.assertEqual(result, ["process-result"])
        self.assertEqual(run_threads.await_count, 1)
        self.assertEqual(run_processes.await_count, 1)

    async def test_auto_remembers_thread_failure_and_starts_next_review_in_process_mode(self):
        with (
            patch("mcp_code_review.app_server._PREFER_PROCESS_MODE", False),
            patch(
                "mcp_code_review.app_server.run_reviews_threads",
                new=AsyncMock(side_effect=AppServerError("thread failure")),
            ) as run_threads,
            patch(
                "mcp_code_review.app_server.run_reviews_processes",
                new=AsyncMock(side_effect=[["first"], ["second"]]),
            ) as run_processes,
        ):
            first = await run_reviews("codex", "/tmp/repo", 2, 30, "auto")
            second = await run_reviews("codex", "/tmp/repo", 2, 30, "auto")

        self.assertEqual(first, ["first"])
        self.assertEqual(second, ["second"])
        self.assertEqual(run_threads.await_count, 1)
        self.assertEqual(run_processes.await_count, 2)

    async def test_threads_mode_falls_back_to_process_mode_on_failure(self):
        with (
            patch("mcp_code_review.app_server._PREFER_PROCESS_MODE", False),
            patch(
                "mcp_code_review.app_server.run_reviews_threads",
                new=AsyncMock(side_effect=AppServerError("thread failure")),
            ) as run_threads,
            patch(
                "mcp_code_review.app_server.run_reviews_processes",
                new=AsyncMock(return_value=["process-result"]),
            ) as run_processes,
        ):
            result = await run_reviews("codex", "/tmp/repo", 2, 30, "threads")

        self.assertEqual(result, ["process-result"])
        self.assertEqual(run_threads.await_count, 1)
        self.assertEqual(run_processes.await_count, 1)

    async def test_threads_mode_uses_process_mode_when_memory_prefers_processes(self):
        with (
            patch("mcp_code_review.app_server._PREFER_PROCESS_MODE", True),
            patch(
                "mcp_code_review.app_server.run_reviews_threads",
                new=AsyncMock(return_value=["thread-result"]),
            ) as run_threads,
            patch(
                "mcp_code_review.app_server.run_reviews_processes",
                new=AsyncMock(return_value=["process-result"]),
            ) as run_processes,
        ):
            result = await run_reviews("codex", "/tmp/repo", 2, 30, "threads")

        self.assertEqual(result, ["process-result"])
        self.assertEqual(run_threads.await_count, 0)
        self.assertEqual(run_processes.await_count, 1)

    async def test_auto_does_not_remember_process_mode_when_fallback_fails(self):
        with (
            patch("mcp_code_review.app_server._PREFER_PROCESS_MODE", False),
            patch(
                "mcp_code_review.app_server.run_reviews_threads",
                new=AsyncMock(side_effect=[AppServerError("thread failure"), ["thread-result"]]),
            ) as run_threads,
            patch(
                "mcp_code_review.app_server.run_reviews_processes",
                new=AsyncMock(side_effect=[AppServerError("process failure")]),
            ) as run_processes,
        ):
            with self.assertRaises(AppServerError):
                await run_reviews("codex", "/tmp/repo", 2, 30, "auto")
            second = await run_reviews("codex", "/tmp/repo", 2, 30, "auto")

        self.assertEqual(second, ["thread-result"])
        self.assertEqual(run_threads.await_count, 2)
        self.assertEqual(run_processes.await_count, 1)

    async def test_auto_does_not_fallback_on_mcp_policy_violation(self):
        with (
            patch("mcp_code_review.app_server._PREFER_PROCESS_MODE", False),
            patch(
                "mcp_code_review.app_server.run_reviews_threads",
                new=AsyncMock(side_effect=AppServerError(MCP_DISABLED_ERROR_TEXT)),
            ) as run_threads,
            patch(
                "mcp_code_review.app_server.run_reviews_processes",
                new=AsyncMock(return_value=["process-result"]),
            ) as run_processes,
        ):
            with self.assertRaises(AppServerError) as ctx:
                await run_reviews("codex", "/tmp/repo", 2, 30, "auto")

        self.assertEqual(str(ctx.exception), MCP_DISABLED_ERROR_TEXT)
        self.assertEqual(run_threads.await_count, 1)
        self.assertEqual(run_processes.await_count, 0)

    async def test_auto_does_not_fallback_on_wrapped_mcp_policy_violation(self):
        wrapped_error = (
            "{'code': -32000, 'message': "
            f"'{MCP_DISABLED_ERROR_TEXT}'"
            "}"
        )
        with (
            patch("mcp_code_review.app_server._PREFER_PROCESS_MODE", False),
            patch(
                "mcp_code_review.app_server.run_reviews_threads",
                new=AsyncMock(side_effect=AppServerError(wrapped_error)),
            ) as run_threads,
            patch(
                "mcp_code_review.app_server.run_reviews_processes",
                new=AsyncMock(return_value=["process-result"]),
            ) as run_processes,
        ):
            with self.assertRaises(AppServerError) as ctx:
                await run_reviews("codex", "/tmp/repo", 2, 30, "auto")

        self.assertIn(MCP_DISABLED_ERROR_TEXT, str(ctx.exception))
        self.assertEqual(run_threads.await_count, 1)
        self.assertEqual(run_processes.await_count, 0)

    async def test_auto_does_not_fallback_on_timeout_error(self):
        with (
            patch("mcp_code_review.app_server._PREFER_PROCESS_MODE", False),
            patch(
                "mcp_code_review.app_server.run_reviews_threads",
                new=AsyncMock(side_effect=AppServerError("codex review timed out after 30s")),
            ) as run_threads,
            patch(
                "mcp_code_review.app_server.run_reviews_processes",
                new=AsyncMock(return_value=["process-result"]),
            ) as run_processes,
        ):
            with self.assertRaises(AppServerError) as ctx:
                await run_reviews("codex", "/tmp/repo", 2, 30, "auto")

        self.assertIn("timed out", str(ctx.exception).lower())
        self.assertEqual(run_threads.await_count, 1)
        self.assertEqual(run_processes.await_count, 0)

    async def test_auto_recovers_to_threads_when_preferred_process_mode_fails(self):
        with (
            patch("mcp_code_review.app_server._PREFER_PROCESS_MODE", True),
            patch(
                "mcp_code_review.app_server.run_reviews_threads",
                new=AsyncMock(return_value=["thread-result"]),
            ) as run_threads,
            patch(
                "mcp_code_review.app_server.run_reviews_processes",
                new=AsyncMock(side_effect=AppServerError("process failure")),
            ) as run_processes,
        ):
            result = await run_reviews("codex", "/tmp/repo", 2, 30, "auto")

        self.assertEqual(result, ["thread-result"])
        self.assertEqual(run_processes.await_count, 1)
        self.assertEqual(run_threads.await_count, 1)

    async def test_auto_preferred_process_mode_does_not_retry_threads_on_mcp_policy_violation(self):
        with (
            patch("mcp_code_review.app_server._PREFER_PROCESS_MODE", True),
            patch(
                "mcp_code_review.app_server.run_reviews_threads",
                new=AsyncMock(return_value=["thread-result"]),
            ) as run_threads,
            patch(
                "mcp_code_review.app_server.run_reviews_processes",
                new=AsyncMock(side_effect=AppServerError(MCP_DISABLED_ERROR_TEXT)),
            ) as run_processes,
        ):
            with self.assertRaises(AppServerError) as ctx:
                await run_reviews("codex", "/tmp/repo", 2, 30, "auto")

        self.assertEqual(str(ctx.exception), MCP_DISABLED_ERROR_TEXT)
        self.assertEqual(run_processes.await_count, 1)
        self.assertEqual(run_threads.await_count, 0)

    async def test_auto_preferred_process_mode_clears_memory_on_hard_process_error(self):
        with (
            patch("mcp_code_review.app_server._PREFER_PROCESS_MODE", True),
            patch(
                "mcp_code_review.app_server.run_reviews_threads",
                new=AsyncMock(return_value=["thread-result"]),
            ) as run_threads,
            patch(
                "mcp_code_review.app_server.run_reviews_processes",
                new=AsyncMock(side_effect=[AppServerError("codex review timed out after 30s")]),
            ) as run_processes,
        ):
            with self.assertRaises(AppServerError) as ctx:
                await run_reviews("codex", "/tmp/repo", 2, 30, "auto")
            second = await run_reviews("codex", "/tmp/repo", 2, 30, "auto")

        self.assertIn("timed out", str(ctx.exception).lower())
        self.assertEqual(second, ["thread-result"])
        self.assertEqual(run_processes.await_count, 1)
        self.assertEqual(run_threads.await_count, 1)


class _RecordingClient:
    def __init__(
        self,
        *,
        review_thread_id: str = "thread-1",
        fail_archive: bool = False,
        complete_review: bool = True,
    ) -> None:
        self.requests = []
        self.review_thread_id = review_thread_id
        self.fail_archive = fail_archive
        self.complete_review = complete_review

    async def request(self, method, params=None):
        self.requests.append((method, params))
        if method == "thread/start":
            return {"thread": {"id": "thread-1"}}
        if method == "review/start":
            return {"turn": {"id": "turn-1"}, "reviewThreadId": self.review_thread_id}
        if method == "thread/archive":
            if self.fail_archive:
                raise AppServerError("archive failed")
            return {}
        raise AssertionError(f"unexpected method: {method}")

    def add_tracker(self, tracker):
        if self.complete_review:
            tracker.future.set_result(
                '{"verdict":"pass","summary":"ok","comment_markdown":""}'
            )


class _StallingArchiveClient:
    async def request(self, method, params=None):
        await asyncio.sleep(3600)


class _ClosedStdout:
    async def read(self, _size):
        return b""


class TestRunSingleReview(unittest.IsolatedAsyncioTestCase):
    async def test_thread_start_payload_omits_base_instructions(self):
        client = _RecordingClient()

        result = await run_single_review(
            client,
            cwd="/tmp/project",
            timeout_seconds=5,
            model="gpt-5",
            model_provider="openai",
            service_tier="priority",
        )

        self.assertEqual(result.verdict, "pass")
        self.assertEqual(client.requests[0][0], "thread/start")
        thread_start_payload = client.requests[0][1]
        self.assertEqual(thread_start_payload["cwd"], "/tmp/project")
        self.assertEqual(thread_start_payload["approvalPolicy"], "never")
        self.assertEqual(thread_start_payload["sandbox"], "read-only")
        self.assertEqual(thread_start_payload["model"], "gpt-5")
        self.assertEqual(thread_start_payload["modelProvider"], "openai")
        self.assertEqual(thread_start_payload["serviceTier"], "priority")
        self.assertNotIn("baseInstructions", thread_start_payload)

    async def test_review_start_uses_uncommitted_changes_without_extra_instructions(self):
        client = _RecordingClient()

        await run_single_review(
            client,
            cwd="/tmp/project",
            timeout_seconds=5,
        )

        self.assertEqual(client.requests[1][0], "review/start")
        review_start_payload = client.requests[1][1]
        self.assertEqual(review_start_payload["delivery"], "inline")
        self.assertEqual(review_start_payload["target"], {"type": "uncommittedChanges"})
        self.assertEqual(client.requests[2][0], "thread/archive")
        self.assertEqual(client.requests[2][1], {"threadId": "thread-1"})

    async def test_archive_uses_review_thread_id(self):
        client = _RecordingClient(review_thread_id="review-thread-1")

        await run_single_review(
            client,
            cwd="/tmp/project",
            timeout_seconds=5,
        )

        self.assertEqual(client.requests[2], ("thread/archive", {"threadId": "review-thread-1"}))

    async def test_archive_failure_marks_result_without_failing_review(self):
        client = _RecordingClient(fail_archive=True)

        result = await run_single_review(
            client,
            cwd="/tmp/project",
            timeout_seconds=5,
        )

        self.assertEqual(result.verdict, "pass")
        self.assertTrue(result.archive_failed)

    async def test_timeout_attempts_archive_and_preserves_timeout_error(self):
        client = _RecordingClient(complete_review=False)

        with self.assertRaisesRegex(AppServerError, "timed out"):
            await run_single_review(
                client,
                cwd="/tmp/project",
                timeout_seconds=0,
            )

        self.assertEqual(client.requests[2], ("thread/archive", {"threadId": "thread-1"}))

    async def test_archive_helper_times_out_stalled_archive_request(self):
        archived = await best_effort_archive_thread(
            _StallingArchiveClient(),
            "thread-1",
            timeout_seconds=0,
        )

        self.assertFalse(archived)

    async def test_handle_response_ignores_cancelled_pending_request(self):
        client = object.__new__(AppServerClient)
        future = asyncio.get_running_loop().create_future()
        future.cancel()
        client._pending = {"request-1": future}

        await AppServerClient._handle_response(
            client,
            {"id": "request-1", "result": {}},
        )

        self.assertEqual(client._pending, {})

    async def test_reader_loop_reports_stderr_when_stdout_closes(self):
        client = object.__new__(AppServerClient)
        future = asyncio.get_running_loop().create_future()
        client._stdout = _ClosedStdout()
        client._pending = {"request-1": future}
        client._trackers = []
        client._buffer = bytearray()
        client._stderr_buffer = bytearray(b"Error: invalid transport\n")
        client._stderr_task = None

        await AppServerClient._reader_loop(client)

        with self.assertRaisesRegex(AppServerError, "invalid transport"):
            future.result()

    async def test_review_start_uses_custom_prompt_with_extra_instructions(self):
        client = _RecordingClient()

        await run_single_review(
            client,
            cwd="/tmp/project",
            timeout_seconds=5,
            additional_developer_instructions="  Check generated files carefully.  ",
        )

        self.assertEqual(client.requests[1][0], "review/start")
        review_start_payload = client.requests[1][1]
        self.assertEqual(review_start_payload["delivery"], "inline")
        self.assertEqual(review_start_payload["target"]["type"], "custom")
        self.assertIn(
            "Review the current code changes (staged, unstaged, and untracked files)",
            review_start_payload["target"]["instructions"],
        )
        self.assertIn(
            "Additional review instructions:\nCheck generated files carefully.",
            review_start_payload["target"]["instructions"],
        )


if __name__ == "__main__":
    unittest.main()
