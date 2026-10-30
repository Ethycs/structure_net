"""Preserve the supplied archive and vendor only its reviewed source/fixtures."""
from pathlib import Path, PurePosixPath
import hashlib
import stat
from zipfile import ZipFile

from .common import PACKAGE, ROOT, read, sha, write

ARCHIVE = Path("Neighborhood_Bridge_V2_Deep_Test.zip")
ARCHIVE_SHA = "e6d1c595a5819ef55e632b6efa1792b1fccbc13e581ee16ffafd6e4fb713c561"
INPUTS = ROOT / "inputs/neighborhood_bridge_20261030"
V2 = "neighborhood_bridge_v2_20261030"
V1 = "neighborhood_bridge_20261030"
VENDOR = PACKAGE / "vendor"
SELECT = ("bridge.py", "bridge_cli.py", "test_bridge.py", "test_v2.py", "stress_valid.py",
          "check_index_workloads.py", "README.md", "START_HERE.txt", "BACKEND_SHA256.txt",
          "long_workflow/fixture_factory.py", "long_workflow/cases.json",
          "long_workflow/INTERPRETER_CONTRACT.md", "long_workflow/semantic_rubric.md",
          "comparison/compare.py", "comparison/manifest.json")


def safe_members(infos):
    names = set()
    for item in infos:
        path = PurePosixPath(item.filename)
        if (path.is_absolute() or ".." in path.parts or "\\" in item.filename
                or path.parts[0] not in (V1, V2) or item.filename in names
                or stat.S_ISLNK(item.external_attr >> 16)):
            raise ValueError("unsafe or duplicate archive member")
        names.add(item.filename)
    return names


def run():
    if sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError("supplied archive identity changed")
    if INPUTS.exists():
        manifest = read(INPUTS / "manifest.json")
        if manifest["archive_sha256"] != ARCHIVE_SHA:
            raise ValueError("existing import identity differs")
        assert all(sha(INPUTS / p) == h for p, h in manifest["members"].items())
        return manifest
    with ZipFile(ARCHIVE) as archive:
        safe_members(archive.infolist())
        # A fresh checkout already contains the reviewed vendor tree, but not
        # the ignored data import. Verify it before creating any artifacts.
        bindings = {name: hashlib.sha256(archive.read(f"{V2}/{name}")).hexdigest() for name in SELECT}
        provenance = {"archive_sha256": ARCHIVE_SHA, "archive_root": V2,
                      "files": bindings, "source_modified": False}
        for name, expected in bindings.items():
            target = VENDOR / name
            if target.exists() and sha(target) != expected:
                raise ValueError("preserve changed vendor source: " + name)
        if (VENDOR / "provenance.json").exists() and read(VENDOR / "provenance.json") != provenance:
            raise ValueError("preserve differing vendor provenance")
        INPUTS.mkdir(parents=True)
        members = {}
        for item in archive.infolist():
            target = INPUTS / item.filename
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            content = archive.read(item)
            with target.open("xb") as f:
                f.write(content)
            members[item.filename] = hashlib.sha256(content).hexdigest()
        for name in SELECT:
            target = VENDOR / name
            target.parent.mkdir(parents=True, exist_ok=True)
            content = archive.read(f"{V2}/{name}")
            if not target.exists():
                with target.open("xb") as f:
                    f.write(content)
    manifest = {"archive": str(ARCHIVE), "archive_sha256": ARCHIVE_SHA, "members": members,
                "historical_outputs_are_new_model_trials": False}
    write(INPUTS / "manifest.json", manifest)
    if not (VENDOR / "provenance.json").exists():
        write(VENDOR / "provenance.json", provenance)
    return {"imported_files": len(members), "vendored_files": len(bindings), "archive_sha256": ARCHIVE_SHA}


if __name__ == "__main__":
    print(run())
