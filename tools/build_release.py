#!/usr/bin/env python3
"""Validate and package the explicit release file list; never publish remotely."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "rl-train-infer-diagnosis-tool"
SKILL = PurePosixPath("skills/rl-train-infer-diagnosis")
FORBIDDEN_PARTS = {".git", ".venv", "__pycache__", "__MACOSX", "outputs", "dist"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".npz", ".pt", ".pth", ".safetensors"}


def release_files(root):
    files = []
    for line in (root / "release-files.txt").read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
            raise ValueError(f"Invalid release path: {name}")
        if (set(path.parts) & FORBIDDEN_PARTS or path.name == ".DS_Store"
                or any(part.startswith("._") for part in path.parts)
                or path.suffix in FORBIDDEN_SUFFIXES or path.name.startswith(".env")):
            raise ValueError(f"Excluded artifact listed for release: {name}")
        source = root / name
        if any((root / Path(*path.parts[:i])).is_symlink() for i in range(1, len(path.parts) + 1)):
            raise ValueError(f"Symlink is not a release file: {name}")
        if not source.is_file():
            raise ValueError(f"Missing release file: {name}")
        if name in files:
            raise ValueError(f"Duplicate release file: {name}")
        files.append(name)
    if not files:
        raise ValueError("Empty release file list")
    return sorted(files)


def validate(root, files):
    required = {"README.md", "PROVENANCE.md", "release-status.json", "release-files.txt",
                "requirements.txt", "requirements-dev.txt", "tools/build_release.py",
                str(SKILL / "SKILL.md"), str(SKILL / "requirements.txt")}
    if not required.issubset(files):
        raise ValueError(f"Missing required release entries: {sorted(required - set(files))}")
    text = (root / SKILL / "SKILL.md").read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---(?:\n|$)", text, re.DOTALL)
    if not match:
        raise ValueError("SKILL.md must start with YAML frontmatter")
    meta = yaml.safe_load(match.group(1))
    if not isinstance(meta, dict) or meta.get("name") != SKILL.name:
        raise ValueError("Skill directory and frontmatter name must match")
    if not isinstance(meta.get("description"), str) or not 1 <= len(meta["description"]) <= 1024:
        raise ValueError("Skill description must contain 1–1024 characters")
    ui = yaml.safe_load((root / SKILL / "agents/openai.yaml").read_text(encoding="utf-8"))
    if "$" + SKILL.name not in ui["interface"]["default_prompt"]:
        raise ValueError("UI prompt must refer to the installed skill name")

    # Check links against the archive contents, not only files present on disk.
    root_resolved = root.resolve()
    for name in files:
        if not name.endswith(".md"):
            continue
        source = root / name
        for target in re.findall(r"\]\(([^)]+)\)", source.read_text(encoding="utf-8")):
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0]
            if Path(target).is_absolute():
                raise ValueError(f"Nonportable Markdown link in {name}")
            try:
                dest = (source.parent / target).resolve().relative_to(root_resolved).as_posix()
            except ValueError:
                raise ValueError(f"Markdown link leaves the package: {name}: {target}")
            if dest not in files:
                raise ValueError(f"Markdown link missing from release: {name}: {target}")
    status = json.loads((root / "release-status.json").read_text(encoding="utf-8"))
    if type(status.get("rights_confirmed")) is not bool:
        raise ValueError("rights_confirmed must be a boolean")
    license_id = status.get("license_spdx")
    if license_id is not None and (not isinstance(license_id, str) or not license_id.strip()):
        raise ValueError("license_spdx must be a nonempty string or null")
    return meta, status


def require_license(root, files, meta, status):
    if status["rights_confirmed"] is not True or not status.get("license_spdx"):
        raise ValueError("Rights/license confirmation pending. Use --draft for a review archive.")
    paths = ["LICENSE", str(SKILL / "LICENSE")]
    if not set(paths).issubset(files):
        raise ValueError("Both repository and standalone skill LICENSE must be listed")
    licenses = [(root / name).read_bytes() for name in paths]
    if len(licenses[0]) < 200 or licenses[0] != licenses[1]:
        raise ValueError("Repository and skill must carry identical complete license text")
    if meta.get("license") != status["license_spdx"]:
        raise ValueError("Skill license metadata must match release-status.json")


def build(root, destination, draft=False):
    files = release_files(root)
    meta, status = validate(root, files)
    if not draft:
        require_license(root, files, meta, status)
    destination.mkdir(parents=True, exist_ok=True)
    suffix = "-draft" if draft else ""
    archive = destination / f"{PROJECT}{suffix}.zip"
    if archive.exists():
        raise ValueError(f"Archive already exists: {archive.name}; choose a new destination")
    # Fixed timestamps and permissions make the same source produce the same ZIP.
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as output:
        entries = {name: (root / name).read_bytes() for name in files}
        if draft:
            entries["DRAFT_NOTICE.txt"] = (
                "Review draft. This archive is not a public release or a grant of rights.\n"
                "See PROVENANCE.md and release-status.json for authorization status.\n"
            ).encode("utf-8")
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(f"{PROJECT}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            output.writestr(info, data)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check packaging, permitting pending license")
    mode.add_argument("--draft", action="store_true", help="Build a marked local review draft")
    parser.add_argument("--destination", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    try:
        files = release_files(ROOT)
        meta, status = validate(ROOT, files)
        if args.check:
            if status["rights_confirmed"]:
                require_license(ROOT, files, meta, status)
            print(json.dumps({"packaging": "ok", "file_count": len(files),
                              "rights_confirmed": status["rights_confirmed"],
                              "license_spdx": status["license_spdx"]}, indent=2))
            return
        archive = build(ROOT, args.destination, draft=args.draft)
        print(json.dumps({"archive": archive.name, "draft": args.draft,
                          "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}, indent=2))
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.exit(2, f"Release failed: {exc}\n")


if __name__ == "__main__":
    main()
