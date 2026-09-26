import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_release", ROOT / "tools/build_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def test_draft_contains_exact_allowlist_and_notice(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = release.build(ROOT, Path(directory), draft=True)
            expected = set(release.release_files(ROOT)) | {"DRAFT_NOTICE.txt"}
            with zipfile.ZipFile(archive) as data:
                actual = {str(Path(name).relative_to(release.PROJECT)) for name in data.namelist()}
                self.assertEqual(actual, expected)
                self.assertIsNone(data.testzip())
            # A second independent build must have identical contents and metadata.
            other = release.build(ROOT, Path(directory) / "second", draft=True)
            self.assertEqual(archive.read_bytes(), other.read_bytes())

    def test_pending_publication_rights_prevents_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "release-status.json").write_text(json.dumps({"rights_confirmed": False, "license_spdx": None}))
            with self.assertRaisesRegex(ValueError, "confirmation pending"):
                release.require_release_ready(root, [], {}, {"rights_confirmed": False, "license_spdx": None})

    def test_release_without_license_contains_exact_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = release.build(ROOT, Path(directory))
            with zipfile.ZipFile(archive) as data:
                actual = {str(Path(name).relative_to(release.PROJECT)) for name in data.namelist()}
                self.assertEqual(actual, set(release.release_files(ROOT)))
                self.assertNotIn("LICENSE", actual)
                self.assertNotIn(str(release.SKILL / "LICENSE"), actual)
                self.assertNotIn("DRAFT_NOTICE.txt", actual)
                self.assertIsNone(data.testzip())

    def test_inconsistent_license_declarations_prevent_release(self):
        cases = [
            ([], {"license": "MIT"}, None, "declared license_spdx"),
            (["LICENSE"], {}, None, "declared license_spdx"),
            ([], {"license": "MIT"}, "MIT", "LICENSE must be listed"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for files, meta, license_id, message in cases:
                with self.subTest(files=files, meta=meta, license_id=license_id):
                    with self.assertRaisesRegex(ValueError, message):
                        release.require_release_ready(
                            Path(directory), files, meta,
                            {"rights_confirmed": True, "license_spdx": license_id},
                        )

    def test_manifest_rejects_traversal_metadata_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "safe.txt").write_text("synthetic")
            (root / "alias.txt").symlink_to(root / "safe.txt")
            for bad in ("../outside.txt", ".DS_Store", "__MACOSX/._data", "secret.npz", "alias.txt"):
                with self.subTest(path=bad):
                    (root / "release-files.txt").write_text(bad + "\n")
                    with self.assertRaises(ValueError):
                        release.release_files(root)

    def test_unlisted_files_never_enter_release_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "public.txt").write_text("public")
            (root / "private-notes.md").write_text("local-only synthetic fixture")
            (root / ".DS_Store").write_text("metadata fixture")
            (root / "release-files.txt").write_text("public.txt\n")
            self.assertEqual(release.release_files(root), ["public.txt"])


if __name__ == "__main__":
    unittest.main()
