import os
import subprocess
import tempfile
import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch

from mcp_code_review.app_server import (
    AppServerError,
    BANNED_APP_SERVER_FEATURES,
    GIT_ISOLATION_ENV,
    MCP_DISABLED_ERROR_TEXT,
    MCP_SERVER_LIST_ATTEMPTS,
    REVIEW_PERMISSION_PROFILE,
    build_app_server_command,
    build_app_server_env,
    build_review_filesystem_policy,
    build_review_permission_assignments,
    detect_tool_prefixes,
    ensure_review_permission_profile,
    is_mcp_tool_call_notification,
    resolve_git_common_dir,
    resolve_global_excludes_file,
    list_mcp_server_names_with_retry,
    parse_mcp_server_names,
    run_reviews,
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
            profile="review",
            mcp_server_names=["review", "serena"],
        )

        self.assertEqual(command[0], "codex")
        self.assertEqual(command[-1], "app-server")
        self.assertIn("profile=review", command)
        self.assertEqual(
            BANNED_APP_SERVER_FEATURES,
            (
                "apps",
                "collab",
                "multi_agent",
                "multi_agent_v2",
                "memory_tool",
                "spawn_csv",
                "enable_fanout",
            ),
        )
        for feature in BANNED_APP_SERVER_FEATURES:
            self.assertIn(f"features.{feature}=false", command)
        self.assertIn(SPAWN_GUARD_CONFIG_ASSIGNMENT, command)
        self.assertIn("mcp_servers.review.enabled=false", command)
        self.assertIn("mcp_servers.serena.enabled=false", command)

    def test_build_app_server_command_restricts_reads_after_every_override(self):
        command = build_app_server_command(
            codex_bin="codex",
            profile=None,
            mcp_server_names=[],
            extra_config_overrides=['default_permissions=":danger-no-sandbox"'],
        )
        expected = build_review_permission_assignments()
        # The restriction is the last group of `-c` assignments before `app-server`,
        # so a caller-supplied override cannot widen it.
        tail = command[-(2 * len(expected) + 1):-1]
        self.assertEqual(tail[1::2], expected)
        self.assertEqual(set(tail[0::2]), {"-c"})
        self.assertEqual(expected[0], f'default_permissions="{REVIEW_PERMISSION_PROFILE}"')
        self.assertIn(f"permissions.{REVIEW_PERMISSION_PROFILE}.network.enabled=false", expected)

    def test_review_filesystem_policy_reads_only_platform_project_and_tools(self):
        policy = build_review_filesystem_policy(["/opt/homebrew"])
        self.assertEqual(
            policy,
            {
                ":minimal": "read",
                ":project_roots": "read",
                "/opt/homebrew": "read",
                "/opt/homebrew/etc": "deny",
                "/opt/homebrew/var": "deny",
            },
        )
        self.assertNotIn(":root", policy)
        self.assertFalse(any(path.startswith("/Users") or path.startswith("/home") for path in policy))

    def test_review_permission_assignment_is_a_toml_inline_table(self):
        assignment = build_review_permission_assignments(["/opt/homebrew"])[1]
        self.assertEqual(
            assignment,
            f"permissions.{REVIEW_PERMISSION_PROFILE}.filesystem="
            '{":minimal"="read",":project_roots"="read","/opt/homebrew"="read",'
            '"/opt/homebrew/etc"="deny","/opt/homebrew/var"="deny"}',
        )

    def test_detect_tool_prefixes_keeps_existing_prefixes_with_bin(self):
        with patch("mcp_code_review.app_server.os.path.isdir", side_effect=lambda p: p == "/opt/homebrew/bin"):
            self.assertEqual(detect_tool_prefixes(("/opt/homebrew", "/usr/local", "/opt/homebrew")), ["/opt/homebrew"])

    def test_app_server_env_isolates_git_config_and_keeps_base(self):
        env = build_app_server_env(
            {
                "PATH": "/usr/bin",
                "GIT_CONFIG_GLOBAL": "/Users/x/.gitconfig",
                "GIT_CONFIG_COUNT": "2",
                "GIT_CONFIG_KEY_1": "url.https://tok@x/.insteadOf",
                "GIT_CONFIG_VALUE_1": "https://x/",
            }
        )
        self.assertEqual(env["PATH"], "/usr/bin")
        for key, value in GIT_ISOLATION_ENV.items():
            self.assertEqual(env[key], value)
        self.assertNotIn("GIT_CONFIG_COUNT", env)
        self.assertNotIn("GIT_CONFIG_KEY_1", env)
        self.assertNotIn("GIT_CONFIG_VALUE_1", env)

    def test_app_server_env_keeps_the_global_excludes_file(self):
        env = build_app_server_env({"PATH": "/usr/bin"}, excludes_file="/home/u/.config/git/ignore")
        self.assertEqual(env["GIT_CONFIG_COUNT"], "1")
        self.assertEqual(env["GIT_CONFIG_KEY_0"], "core.excludesFile")
        self.assertEqual(env["GIT_CONFIG_VALUE_0"], "/home/u/.config/git/ignore")

    def test_review_filesystem_policy_adds_extra_readable_paths(self):
        policy = build_review_filesystem_policy([], ["/repo/.git", "/home/u/.config/git/ignore"])
        self.assertEqual(policy["/repo/.git"], "read")
        self.assertEqual(policy["/home/u/.config/git/ignore"], "read")

    def test_ensure_review_permission_profile(self):
        ensure_review_permission_profile({"activePermissionProfile": {"id": REVIEW_PERMISSION_PROFILE}})
        for response in (None, {}, {"activePermissionProfile": {"id": ":read-only"}}, {"activePermissionProfile": "x"}):
            with self.assertRaises(AppServerError):
                ensure_review_permission_profile(response)


def _git(*args, cwd):
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args],
        cwd=cwd, check=True, capture_output=True,
    )


class TestGitPathsForReview(unittest.TestCase):
    def test_global_excludes_file_defaults_to_xdg_location(self):
        with tempfile.TemporaryDirectory() as home:
            env = {"HOME": home, "PATH": os.environ.get("PATH", "")}
            self.assertIsNone(resolve_global_excludes_file(env))
            ignore = os.path.join(home, ".config", "git", "ignore")
            os.makedirs(os.path.dirname(ignore))
            with open(ignore, "w") as fh:
                fh.write("*.env\n")
            self.assertEqual(resolve_global_excludes_file(env), os.path.abspath(ignore))

    def test_git_common_dir_only_for_linked_worktrees(self):
        with tempfile.TemporaryDirectory() as root:
            main = os.path.join(root, "main")
            linked = os.path.join(root, "linked")
            os.makedirs(main)
            _git("init", "-q", cwd=main)
            with open(os.path.join(main, "f"), "w") as fh:
                fh.write("x\n")
            _git("add", "f", cwd=main)
            _git("commit", "-q", "-m", "init", cwd=main)
            _git("worktree", "add", "-q", linked, cwd=main)
            env = dict(os.environ)
            self.assertIsNone(resolve_git_common_dir(main, env))
            self.assertEqual(
                resolve_git_common_dir(linked, env), os.path.realpath(os.path.join(main, ".git"))
            )
            self.assertIsNone(resolve_git_common_dir(root, env))
            self.assertIsNone(resolve_git_common_dir(None, env))

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
    async def test_list_mcp_server_names_retries_once(self):
        list_once = AsyncMock(side_effect=[AppServerError("transient"), ["review"]])
        with patch("mcp_code_review.app_server._list_mcp_server_names_once", list_once):
            names = await list_mcp_server_names_with_retry("codex", profile="review")

        self.assertEqual(names, ["review"])
        self.assertEqual(list_once.await_count, 2)

    async def test_list_mcp_server_names_fails_after_retry(self):
        list_once = AsyncMock(
            side_effect=[AppServerError("first"), AppServerError("second")]
        )
        with patch("mcp_code_review.app_server._list_mcp_server_names_once", list_once):
            with self.assertRaises(AppServerError):
                await list_mcp_server_names_with_retry("codex", profile=None)

        self.assertEqual(list_once.await_count, MCP_SERVER_LIST_ATTEMPTS)


class TestRunReviewsFallback(unittest.IsolatedAsyncioTestCase):
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
    def __init__(self) -> None:
        self.requests = []

    async def request(self, method, params=None):
        self.requests.append((method, params))
        if method == "thread/start":
            return {
                "thread": {"id": "thread-1"},
                "activePermissionProfile": {"id": REVIEW_PERMISSION_PROFILE, "extends": None},
            }
        if method == "review/start":
            return {"turn": {"id": "turn-1"}, "reviewThreadId": "thread-1"}
        raise AssertionError(f"unexpected method: {method}")

    def add_tracker(self, tracker):
        tracker.future.set_result(
            '{"verdict":"pass","summary":"ok","comment_markdown":""}'
        )


class TestRunSingleReview(unittest.IsolatedAsyncioTestCase):
    async def test_refuses_to_review_without_the_permission_profile(self):
        client = _RecordingClient()

        async def request(method, params=None):
            client.requests.append((method, params))
            return {"thread": {"id": "thread-1"}}

        client.request = request
        with self.assertRaises(AppServerError):
            await run_single_review(client, cwd="/tmp/project", timeout_seconds=5)
        self.assertEqual([m for m, _ in client.requests], ["thread/start"])

    async def test_thread_start_payload_omits_base_instructions(self):
        client = _RecordingClient()

        result = await run_single_review(
            client,
            cwd="/tmp/project",
            timeout_seconds=5,
            model="gpt-5",
            model_provider="openai",
        )

        self.assertEqual(result.verdict, "pass")
        self.assertEqual(client.requests[0][0], "thread/start")
        thread_start_payload = client.requests[0][1]
        self.assertEqual(thread_start_payload["cwd"], "/tmp/project")
        self.assertEqual(thread_start_payload["approvalPolicy"], "never")
        # An explicit sandbox mode would replace the read-restricting permissions
        # profile set on the app-server command line.
        self.assertNotIn("sandbox", thread_start_payload)
        self.assertNotIn("sandboxPolicy", thread_start_payload)
        self.assertEqual(thread_start_payload["model"], "gpt-5")
        self.assertEqual(thread_start_payload["modelProvider"], "openai")
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
