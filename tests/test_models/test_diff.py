from sentinel.models.diff import parse_git_diff

SAMPLE_DIFF = """diff --git a/src/auth/session.ts b/src/auth/session.ts
index e69de29..d95f3ad 100644
--- a/src/auth/session.ts
+++ b/src/auth/session.ts
@@ -10,3 +10,4 @@ export function getSession() {
-  return window.localStorage.getItem("token");
+  return SecureEncryptedStore.getItem("token");
 }
+export const AUTH_VERSION = "2.0";
diff --git a/src/utils/config.ts b/src/utils/config.ts
new file mode 100644
index 0000000..83db48f
--- /dev/null
+++ b/src/utils/config.ts
@@ -0,0 +1,2 @@
+export const API_URL = "https://api.example.com";
+export const TIMEOUT = 5000;
"""


def test_parse_git_diff_multiple_files():
    git_diff = parse_git_diff(SAMPLE_DIFF)
    assert len(git_diff.files) == 2
    assert git_diff.touched_files == ["src/auth/session.ts", "src/utils/config.ts"]

    # Check first file
    file1 = git_diff.files[0]
    assert file1.path == "src/auth/session.ts"
    assert file1.change_type == "modified"
    assert len(file1.deleted_lines) == 1
    assert 'return window.localStorage.getItem("token");' in file1.deleted_lines[0]
    assert len(file1.added_lines) == 2
    assert 'return SecureEncryptedStore.getItem("token");' in file1.added_lines[0]
    assert 'export const AUTH_VERSION = "2.0";' in file1.added_lines[1]

    # Check second file
    file2 = git_diff.files[1]
    assert file2.path == "src/utils/config.ts"
    assert file2.change_type == "added"
    assert len(file2.added_lines) == 2
    assert len(file2.deleted_lines) == 0

    assert git_diff.total_added == 4
    assert git_diff.total_deleted == 1


def test_parse_git_diff_deleted_file():
    diff_text = """diff --git a/legacy.ts b/legacy.ts
deleted file mode 100644
--- a/legacy.ts
+++ /dev/null
@@ -1,2 +0,0 @@
-const oldCode = true;
-export default oldCode;
"""
    git_diff = parse_git_diff(diff_text)
    assert len(git_diff.files) == 1
    assert git_diff.touched_files == ["legacy.ts"]
    assert git_diff.files[0].change_type == "deleted"
    assert len(git_diff.files[0].deleted_lines) == 2
    assert git_diff.total_deleted == 2
    assert git_diff.total_added == 0


def test_parse_empty_diff():
    git_diff = parse_git_diff("")
    assert git_diff.files == []
    assert git_diff.touched_files == []
    assert git_diff.total_added == 0
    assert git_diff.total_deleted == 0


def test_parse_git_diff_quoted_paths():
    quoted_diff = """diff --git "a/path with spaces/file.ts" "b/path with spaces/file.ts"
index 1111111..2222222 100644
--- "a/path with spaces/file.ts"
+++ "b/path with spaces/file.ts"
@@ -1,1 +1,2 @@
+const added = true;
"""
    git_diff = parse_git_diff(quoted_diff)
    assert len(git_diff.files) == 1
    assert git_diff.touched_files == ["path with spaces/file.ts"]
    assert git_diff.files[0].path == "path with spaces/file.ts"
    assert git_diff.total_added == 1


def test_parse_git_diff_unicode_octal_escapes():
    # Git outputs octal escapes \303\251 for 'é'
    octal_diff = r"""diff --git "a/src/t\303\251st.ts" "b/src/t\303\251st.ts"
index 1111111..2222222 100644
--- "a/src/t\303\251st.ts"
+++ "b/src/t\303\251st.ts"
@@ -1,1 +1,2 @@
+export const UTF8 = true;
"""
    git_diff = parse_git_diff(octal_diff)
    assert len(git_diff.files) == 1
    assert git_diff.touched_files == ["src/t\u00e9st.ts"]
    assert git_diff.files[0].path == "src/t\u00e9st.ts"


def test_parse_git_diff_whitespace_and_hunk_metadata():
    diff_text = """diff --git a/src/app.py b/src/app.py
index 1234567..89abcdef 100644
--- a/src/app.py
+++ b/src/app.py
@@ -10,3 +10,4 @@ def run():
-    # old indentation
+    # new indentation with trailing spaces   
+    print("deleted file mode in string")
"""
    git_diff = parse_git_diff(diff_text)
    assert len(git_diff.files) == 1
    file_diff = git_diff.files[0]
    assert file_diff.change_type == "modified", (
        "Must remain modified despite metadata words in hunk"
    )
    assert file_diff.added_lines[0] == "    # new indentation with trailing spaces   "
    assert file_diff.deleted_lines[0] == "    # old indentation"


if __name__ == "__main__":
    test_parse_git_diff_whitespace_and_hunk_metadata()
    test_parse_git_diff_unicode_octal_escapes()
    print("test_diff.py standalone checks passed.")
