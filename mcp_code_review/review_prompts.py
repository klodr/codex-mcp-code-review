# Drift note:
# This mirrors only the native uncommitted-changes review target prompt from:
# `.agent/codex/codex-rs/core/src/review_prompts.rs` at commit
# `6a673e7339161cf5aa6711324bfe873846234b6b`.
#
# Local alteration:
# - append `Additional review instructions` when this MCP server is called with
#   per-run `additional_developer_instructions`
# - add `FULL_REPO_REVIEW_PROMPT` for whole-codebase audits (custom target,
#   not mirrored upstream since Codex has no native `--full-repo` review mode)
UNCOMMITTED_CHANGES_REVIEW_PROMPT = (
    "Review the current code changes (staged, unstaged, and untracked files) "
    "and provide prioritized findings."
)


FULL_REPO_REVIEW_PROMPT = (
    "Audit the entire repository for issues. Walk the project tree starting "
    "from the obvious entry points (README, package manifest such as "
    "package.json / pyproject.toml / Cargo.toml / go.mod, main module or "
    "binary, plus CI workflows under .github/) and explore every source file "
    "reachable from there. Skip generated artefacts (dist/, build/, node_modules/, "
    ".venv/, vendored deps, lockfiles, binary assets, anything in .gitignore).\n\n"
    "Produce prioritized findings:\n"
    "- P0: critical security or correctness defects exploitable today.\n"
    "- P1: major correctness, security, architecture, or maintainability issues.\n"
    "- P2 and below: skip — do not emit noise.\n\n"
    "For every finding, cite `path/to/file.ext:LINE` (or a clear range) and "
    "quote the smallest piece of code that demonstrates the issue. Tag each "
    "finding with `[P0]` or `[P1]` so they can be aggregated across parallel "
    "runs. If the repository is genuinely clean, say so explicitly — do not "
    "manufacture findings."
)


def build_uncommitted_changes_review_prompt() -> str:
    return UNCOMMITTED_CHANGES_REVIEW_PROMPT


def build_full_repo_review_prompt() -> str:
    return FULL_REPO_REVIEW_PROMPT


def append_additional_review_instructions(
    prompt: str,
    additional_instructions: str,
) -> str:
    return f"{prompt}\n\nAdditional review instructions:\n{additional_instructions.strip()}"
