#!/usr/bin/env python3
"""Package a validated calibration corpus into a deterministic release archive.

The corpus is fully re-validated through
``prepare_calibration.load_corpus_artifacts`` before anything is written, and
the archive is built from the verified payload bytes that call returns, so an
archive can only exist for a manifest-verified corpus. The tar.gz is
reproducible: sorted members, uid/gid 0, empty uname/gname, mode 0644 (0755
for the root directory entry), mtime 0, and a gzip header with no local
filename or timestamp. The archive is written atomically through a temporary
file in the destination directory, so a failed run never replaces an existing
output or leaves a partial file, and an output path inside the corpus
directory is rejected because it would change the validated inputs.
"""

import argparse
import gzip
import io
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import prepare_calibration
from prepare_calibration import DEFAULT_SEED, CorpusError

ARCHIVE_ROOT = f"corpus-v{prepare_calibration.CORPUS_SCHEMA_VERSION}"
DIRECTORY_MODE = 0o755
FILE_MODE = 0o644
COMPRESSLEVEL = 9


class PackagingError(Exception):
    """A corpus could not be packaged into a release archive."""


class PackageResult(NamedTuple):
    path: Path
    size: int
    sha256: str


def _tar_info(
    name: str, size: int = 0, *, directory: bool = False
) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    info.size = size
    info.mtime = 0
    info.mode = DIRECTORY_MODE if directory else FILE_MODE
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def build_archive(payload: dict[str, bytes]) -> bytes:
    """Serialize verified corpus payload bytes into reproducible tar.gz."""
    buffer = io.BytesIO()
    stream = gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=COMPRESSLEVEL,
        fileobj=buffer,
        mtime=0,
    )
    with stream, tarfile.open(
        fileobj=stream, mode="w", format=tarfile.PAX_FORMAT
    ) as archive:
        archive.addfile(_tar_info(f"{ARCHIVE_ROOT}/", directory=True))
        for name in sorted(payload):
            data = payload[name]
            member = _tar_info(f"{ARCHIVE_ROOT}/{name}", len(data))
            archive.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


def package_corpus(
    corpus_dir: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    seed: int = DEFAULT_SEED,
) -> PackageResult:
    """Validate a corpus and write its deterministic release archive.

    Raises ``PackagingError`` for an invalid corpus, an output path inside
    the corpus directory, or any write failure; a failed run never leaves a
    temporary file behind and never replaces an existing output.
    """
    corpus = Path(corpus_dir)
    target = Path(output)
    if not corpus.is_dir():
        raise PackagingError(f"corpus directory {corpus} does not exist")
    try:
        _manifest, _digest, payload = prepare_calibration.load_corpus_artifacts(
            corpus, seed=seed
        )
    except CorpusError as error:
        raise PackagingError(f"corpus {corpus} failed validation: {error}") from error
    resolved = target.resolve()
    if resolved.is_relative_to(corpus.resolve()):
        raise PackagingError(
            f"output {target} is inside corpus directory {corpus}; writing it "
            "would change the validated corpus inputs"
        )
    if resolved.is_dir():
        raise PackagingError(f"output {target} is a directory")

    try:
        archive = build_archive(payload)
        digest = prepare_calibration._sha256(archive)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(archive)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    except OSError as error:
        raise PackagingError(f"failed to write archive {target}: {error}") from error
    return PackageResult(target, len(archive), digest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Package a validated calibration corpus directory into a "
            "deterministic tar.gz release archive."
        )
    )
    parser.add_argument(
        "corpus_dir",
        help="corpus directory produced by prepare_calibration.build_corpus",
    )
    parser.add_argument(
        "output",
        help="destination archive path (created atomically)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="corpus policy seed used for validation (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    try:
        result = package_corpus(args.corpus_dir, args.output, seed=args.seed)
    except PackagingError as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Packaged corpus release: {result.path} ({result.size:,} bytes)")
    print(f"sha256 {result.sha256}")


if __name__ == "__main__":
    main()
