import codecs
import re
from typing import Literal

from pydantic import BaseModel, Field

FileChangeType = Literal["added", "modified", "deleted", "renamed"]


class FileDiff(BaseModel):
    """Diff delta for a single file."""

    path: str
    change_type: FileChangeType = "modified"
    added_lines: list[str] = Field(default_factory=list)
    deleted_lines: list[str] = Field(default_factory=list)
    raw_patch: str = ""


class GitDiff(BaseModel):
    """Aggregated Git diff across multiple files."""

    files: list[FileDiff] = Field(default_factory=list)
    raw_diff: str = ""

    @property
    def touched_files(self) -> list[str]:
        return [f.path for f in self.files]

    @property
    def total_added(self) -> int:
        return sum(len(f.added_lines) for f in self.files)

    @property
    def total_deleted(self) -> int:
        return sum(len(f.deleted_lines) for f in self.files)


def _clean_git_path(p: str) -> str:
    p = p.strip()
    if p.startswith('"') and p.endswith('"'):
        unquoted = p[1:-1]
        try:
            raw_bytes, _ = codecs.escape_decode(unquoted.encode("latin1"))
            p = raw_bytes.decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            p = unquoted
    if p.startswith(("a/", "b/")):
        p = p[2:]
    return p


def parse_git_diff(raw_diff: str) -> GitDiff:
    """Parse unified git diff output into structured GitDiff model."""
    if not raw_diff or not raw_diff.strip():
        return GitDiff(files=[], raw_diff="")

    files: list[FileDiff] = []
    # Split diff by file header: "diff --git"
    raw_file_blocks = re.split(r"(?=^diff --git )", raw_diff.strip(), flags=re.MULTILINE)

    for block in raw_file_blocks:
        if not block.strip():
            continue

        lines = block.splitlines()
        first_line = lines[0]

        # Extract file paths from "diff --git a/... b/..." or "diff --git "a/..." "b/...""
        match = re.match(r'^diff --git (?:("a/.*?"|a/\S+))\s+(?:("b/.*?"|b/\S+))$', first_line)
        if match:
            old_raw, new_raw = match.groups()
            old_path = _clean_git_path(old_raw)
            new_path = _clean_git_path(new_raw)
        else:
            # Fallback using +++ and --- lines
            plus_line = next((l for l in lines if l.startswith("+++ ")), None)
            minus_line = next((l for l in lines if l.startswith("--- ")), None)
            if plus_line and not plus_line.startswith("+++ /dev/null"):
                new_path = _clean_git_path(plus_line[4:])
                old_path = new_path
            elif minus_line and not minus_line.startswith("--- /dev/null"):
                old_path = _clean_git_path(minus_line[4:])
                new_path = old_path
            else:
                continue

        # Header records exist only before the first @@ hunk marker
        hunk_start_idx = next((i for i, l in enumerate(lines) if l.startswith("@@")), len(lines))
        header_lines = lines[:hunk_start_idx]

        change_type: FileChangeType = "modified"
        target_path = new_path

        if any(l.startswith("new file mode") for l in header_lines):
            change_type = "added"
            target_path = new_path
        elif any(l.startswith("deleted file mode") for l in header_lines):
            change_type = "deleted"
            target_path = old_path
        elif any(l.startswith("rename from") for l in header_lines):
            change_type = "renamed"
            target_path = new_path

        added_lines: list[str] = []
        deleted_lines: list[str] = []

        in_hunk = False
        for line in lines:
            if line.startswith("@@"):
                in_hunk = True
                continue
            if in_hunk:
                if line.startswith("+") and not line.startswith("+++"):
                    added_lines.append(line[1:])
                elif line.startswith("-") and not line.startswith("---"):
                    deleted_lines.append(line[1:])

        files.append(
            FileDiff(
                path=target_path,
                change_type=change_type,
                added_lines=added_lines,
                deleted_lines=deleted_lines,
                raw_patch=block,
            )
        )

    return GitDiff(files=files, raw_diff=raw_diff)


if __name__ == "__main__":
    # Framework-free self-check (Rule 2903681)
    diff_sample = """diff --git a/test.py b/test.py
index 1111111..2222222 100644
--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
-    def old_code():
+    def new_code():
+        # includes "new file mode" inside comment
"""
    parsed = parse_git_diff(diff_sample)
    assert len(parsed.files) == 1
    assert parsed.files[0].change_type == "modified", "Hunk text must not change file type"
    assert parsed.files[0].added_lines[0] == "    def new_code():", "Indentation must be preserved"
    assert parsed.files[0].deleted_lines[0] == "    def old_code():", (
        "Indentation must be preserved"
    )

    # Octal unicode path self-check
    octal_sample = r"""diff --git "a/src/r\303\251sum\303\251.ts" "b/src/r\303\251sum\303\251.ts"
index 1111111..2222222 100644
--- "a/src/r\303\251sum\303\251.ts"
+++ "b/src/r\303\251sum\303\251.ts"
@@ -1,1 +1,1 @@
+export const ok = true;
"""
    parsed_octal = parse_git_diff(octal_sample)
    assert parsed_octal.files[0].path == "src/r\u00e9sum\u00e9.ts"

    print("diff.py standalone self-check passed successfully.")
