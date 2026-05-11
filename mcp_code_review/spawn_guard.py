import os
import subprocess
from typing import Iterable, Iterator, Optional


SPAWN_GUARD_CONFIG_KEY = "codex_mcp_code_review_spawn_guard"
SPAWN_GUARD_CONFIG_VALUE = "true"
SPAWN_GUARD_CONFIG_ASSIGNMENT = f"{SPAWN_GUARD_CONFIG_KEY}={SPAWN_GUARD_CONFIG_VALUE}"


def is_guarded_app_server_cmdline(cmdline: str) -> bool:
    if not cmdline:
        return False
    lowered = cmdline.lower()
    return "app-server" in lowered and SPAWN_GUARD_CONFIG_ASSIGNMENT in lowered


def find_guarded_app_server_in_cmdlines(cmdlines: Iterable[str]) -> Optional[str]:
    for cmdline in cmdlines:
        if is_guarded_app_server_cmdline(cmdline):
            return cmdline
    return None


def _read_via_ps(pid: int, field: str) -> Optional[str]:
    # Fallback for systems where /proc isn't available (macOS, *BSD). Uses
    # `ps -p PID -o <field>=` which is POSIX-standard. `field` is always a
    # literal ("command" or "ppid") from this module — no injection surface.
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", f"{field}="],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output if output else None


def read_proc_cmdline(pid: int) -> Optional[str]:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            data = handle.read()
    except OSError:
        # /proc isn't available on macOS/BSD — fall back to `ps`.
        return _read_via_ps(pid, "command")
    if not data:
        return None
    return data.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()


def read_proc_parent_pid(pid: int) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("PPid:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1])
    except OSError:
        # /proc isn't available on macOS/BSD — fall back to `ps`.
        ppid = _read_via_ps(pid, "ppid")
        if ppid is None:
            return None
        return int(ppid) if ppid.isdigit() else None
    return None


def iter_ancestor_cmdlines(max_depth: int = 16) -> Iterator[str]:
    pid = os.getpid()
    for _ in range(max_depth):
        cmdline = read_proc_cmdline(pid)
        if cmdline:
            yield cmdline

        parent_pid = read_proc_parent_pid(pid)
        if parent_pid is None or parent_pid <= 0 or parent_pid == pid:
            break
        pid = parent_pid


def find_guarded_app_server_ancestor_cmdline(max_depth: int = 16) -> Optional[str]:
    return find_guarded_app_server_in_cmdlines(iter_ancestor_cmdlines(max_depth=max_depth))


__all__ = [
    "SPAWN_GUARD_CONFIG_ASSIGNMENT",
    "SPAWN_GUARD_CONFIG_KEY",
    "SPAWN_GUARD_CONFIG_VALUE",
    "is_guarded_app_server_cmdline",
    "find_guarded_app_server_in_cmdlines",
    "find_guarded_app_server_ancestor_cmdline",
]
