from __future__ import annotations

import pathlib
import textwrap
import zipfile

PROJECT_NAME = "rrps-playground"
PROJECT_VERSION = "0.1.0"
SUMMARY = "Rock-Paper-Scissors with inventory playground"
PYTHON_REQUIRES = ">=3.10"
INSTALL_REQUIRES: list[str] = []
PACKAGE_DIRS = ["rrps"]


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_editable(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    metadata_directory = pathlib.Path(metadata_directory)
    dist_info = metadata_directory / _dist_info_name()
    dist_info.mkdir(parents=True, exist_ok=True)
    _write_metadata(dist_info)
    return dist_info.name


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _build_wheel(pathlib.Path(wheel_directory), editable=False)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _build_wheel(pathlib.Path(wheel_directory), editable=True)


def build_sdist(sdist_directory, config_settings=None):
    sdist_directory = pathlib.Path(sdist_directory)
    sdist_directory.mkdir(parents=True, exist_ok=True)
    archive_name = f"{PROJECT_NAME}-{PROJECT_VERSION}.tar.gz"
    archive_path = sdist_directory / archive_name
    root = _project_root()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in _iter_project_files(root):
            archive.write(path, f"{PROJECT_NAME}-{PROJECT_VERSION}/{path.relative_to(root)}")
    return archive_name


def _build_wheel(wheel_directory: pathlib.Path, editable: bool) -> str:
    wheel_directory.mkdir(parents=True, exist_ok=True)
    wheel_name = f"{_dist_name()}-{PROJECT_VERSION}-py3-none-any.whl"
    wheel_path = wheel_directory / wheel_name
    root = _project_root()
    dist_info = _dist_info_name()
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if editable:
            pth_name = f"{_dist_name()}.pth"
            archive.writestr(pth_name, str(root))
        else:
            for package_dir in PACKAGE_DIRS:
                package_path = root / package_dir
                for path in package_path.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(root).as_posix())
        _write_metadata_to_archive(archive, dist_info)
        _write_wheel_metadata(archive, dist_info)
        _write_record(archive, dist_info)
    return wheel_name


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def _dist_name() -> str:
    return PROJECT_NAME.replace("-", "_")


def _dist_info_name() -> str:
    return f"{_dist_name()}-{PROJECT_VERSION}.dist-info"


def _metadata_contents() -> str:
    requires_dist = "\n".join(f"Requires-Dist: {req}" for req in INSTALL_REQUIRES)
    return textwrap.dedent(
        f"""\
        Metadata-Version: 2.1
        Name: {PROJECT_NAME}
        Version: {PROJECT_VERSION}
        Summary: {SUMMARY}
        Requires-Python: {PYTHON_REQUIRES}
        {requires_dist}
        """
    ).strip() + "\n"


def _write_metadata(dist_info: pathlib.Path) -> None:
    (dist_info / "METADATA").write_text(_metadata_contents(), encoding="utf-8")


def _write_metadata_to_archive(archive: zipfile.ZipFile, dist_info: str) -> None:
    archive.writestr(f"{dist_info}/METADATA", _metadata_contents())


def _write_wheel_metadata(archive: zipfile.ZipFile, dist_info: str) -> None:
    archive.writestr(
        f"{dist_info}/WHEEL",
        textwrap.dedent(
            """\
            Wheel-Version: 1.0
            Generator: rrps-playground-backend
            Root-Is-Purelib: true
            Tag: py3-none-any
            """
        ).strip()
        + "\n",
    )


def _write_record(archive: zipfile.ZipFile, dist_info: str) -> None:
    record_path = f"{dist_info}/RECORD"
    rows = []
    for info in archive.infolist():
        rows.append((info.filename, "", ""))
    rows.append((record_path, "", ""))
    record_content = "\n".join(",".join(row) for row in rows) + "\n"
    archive.writestr(record_path, record_content)


def _iter_project_files(root: pathlib.Path):
    exclude_dirs = {".git", ".venv", "__pycache__"}
    for path in root.rglob("*"):
        if path.is_dir() and path.name in exclude_dirs:
            continue
        if path.is_file():
            yield path
