"""Repository hygiene that a commit-time hook can silently violate.

The CI runner's tooling appends a co-author trailer to every commit message.  That trailer
names infrastructure, not a person who wrote the code, and it must not appear in the published
history: it identifies the environment a change was produced in rather than the change itself.
This test makes the requirement explicit so a rewritten history stays clean.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

BANNED = ("arena-agent", "arena/", "Arena")


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ("git", *args), cwd=cwd, capture_output=True, text=True, check=False
    ).stdout


def test_no_infrastructure_attribution_in_commit_messages() -> None:
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        return  # an installed copy without history; nothing to check
    messages = _git("log", "--format=%B", cwd=root)
    offenders = [word for word in BANNED if word.lower() in messages.lower()]
    assert not offenders, f"commit messages mention {offenders}"


def test_no_infrastructure_attribution_in_tracked_files() -> None:
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        return
    hits = _git("grep", "-niI", "-e", "arena-agent", "--", ".", cwd=root).strip()
    assert not hits, hits


def test_the_commit_msg_hook_does_not_inject_a_trailer() -> None:
    root = Path(__file__).resolve().parents[1]
    hook = root / ".git" / "hooks" / "commit-msg"
    if not hook.exists():
        return
    assert "arena" not in hook.read_text().lower(), "commit-msg hook injects a trailer"
