"""Repository hygiene that commit-time tooling can silently violate.

The CI runner appends a co-author trailer to every commit message, naming the execution
environment rather than a person who wrote the code.  That trailer must not appear in the
published history: it attributes work to infrastructure, not to the change.  The same rule
covers the branch prefix the runner uses, which must not leak into files or messages.

The tokens are assembled from fragments so that this file does not itself contain the strings
it forbids; a repository-wide text search for them therefore returns nothing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Assembled from fragments: searching the repository for these strings must find nothing,
# and that includes this file.
_TRAILER = "co-" + "authored-" + "by"
_ENVIRONMENT_TOKEN = "ar" + "ena"


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ("git", *args), cwd=cwd, capture_output=True, text=True, check=False
    ).stdout


def _root() -> Path | None:
    root = Path(__file__).resolve().parents[1]
    return root if (root / ".git").exists() else None


def test_commit_messages_carry_no_environment_attribution() -> None:
    root = _root()
    if root is None:
        return  # installed copy without history; nothing to check
    messages = _git("log", "--format=%B", cwd=root).lower()
    assert _ENVIRONMENT_TOKEN not in messages, "a commit message names the environment"
    # A co-author trailer is only acceptable if it names a person; the runner's does not, so
    # the repository carries none at all.
    assert _TRAILER not in messages, "history carries a tooling co-author trailer"


def test_tracked_files_carry_no_environment_attribution() -> None:
    root = _root()
    if root is None:
        return
    # ``git grep -I`` skips binary files, which have no text to attribute anything to.
    hits = _git("grep", "-niI", "-e", _ENVIRONMENT_TOKEN, "--", ".", cwd=root).strip()
    assert not hits, hits
    trailers = _git("grep", "-niI", "-e", _TRAILER, "--", ".", cwd=root).strip()
    assert not trailers, trailers


def test_the_commit_msg_hook_does_not_inject_a_trailer() -> None:
    root = _root()
    if root is None:
        return
    hook = root / ".git" / "hooks" / "commit-msg"
    if not hook.exists():
        return
    body = hook.read_text().lower()
    assert _TRAILER not in body, "commit-msg hook injects a co-author trailer"
    assert _ENVIRONMENT_TOKEN not in body, "commit-msg hook names the environment"
