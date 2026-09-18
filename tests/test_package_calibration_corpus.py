import gzip
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import package_calibration_corpus
from package_calibration_corpus import PackagingError

import prepare_calibration

DOMAINS = ("general", "code", "reasoning", "agentic")


def doc_source(name, domain):
    return {
        "dataset": f"example/{name}",
        "revision": "b" * 40,
        "split": "train",
        "category": "reference",
        "domain": domain,
        "extractor": "document",
        "field": "text",
        "minimum_records": 1,
    }


def doc_text(name, index):
    slug = name.replace("/", "-")
    sentences = " ".join(f"{slug}x{index}y{j}." for j in range(40))
    return f"Opening paragraph for {name} {index}.\n\n{sentences}"


SOURCES = (
    doc_source("pack-general", "general"),
    doc_source("pack-code", "code"),
    doc_source("pack-reasoning", "reasoning"),
    doc_source("pack-agentic", "agentic"),
)

ROWS = {
    source["dataset"]: [
        {"text": doc_text(source["dataset"], index)} for index in range(2)
    ]
    for source in SOURCES
}


def read_members(path):
    raw = path.read_bytes()
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(raw)), mode="r:") as tar:
        return raw, {
            member.name: (
                member,
                b"" if member.isdir() else tar.extractfile(member).read(),
            )
            for member in tar.getmembers()
        }


class PackageCorpusTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.base = Path(self.temp_dir.name)
        self.corpus_dir = self.base / "corpus"
        self.output = self.base / "release" / "corpus-v1.tar.gz"

    def build_corpus(self, seed=prepare_calibration.DEFAULT_SEED):
        loader = mock.Mock(side_effect=lambda name, config=None, **kwargs: ROWS[name])
        with (
            mock.patch.object(prepare_calibration, "SOURCES", SOURCES),
            mock.patch.object(prepare_calibration, "load_dataset", loader),
        ):
            prepare_calibration.build_corpus(self.corpus_dir, force=True, seed=seed)

    def package(self, output=None, seed=prepare_calibration.DEFAULT_SEED):
        with mock.patch.object(prepare_calibration, "SOURCES", SOURCES):
            return package_calibration_corpus.package_corpus(
                self.corpus_dir, self.output if output is None else output, seed=seed
            )

    def test_repeated_runs_produce_identical_bytes_and_hash(self):
        self.build_corpus()
        first = self.package(self.base / "first.tar.gz")
        second = self.package(self.base / "second.tar.gz")
        first_bytes = first.path.read_bytes()
        second_bytes = second.path.read_bytes()
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.sha256, hashlib.sha256(first_bytes).hexdigest())
        self.assertEqual(first.size, len(first_bytes))

    def test_archive_members_names_metadata_and_content(self):
        self.build_corpus()
        result = self.package()
        raw, members = read_members(result.path)
        # No FNAME flag and a zero mtime in the gzip header.
        self.assertEqual(raw[:4], b"\x1f\x8b\x08\x00")
        self.assertEqual(raw[4:8], b"\x00\x00\x00\x00")
        expected = ["corpus-v1"] + [
            f"corpus-v1/{name}"
            for name in sorted({f"{domain}.jsonl" for domain in DOMAINS} | {"manifest.json"})
        ]
        self.assertEqual(
            package_calibration_corpus.ARCHIVE_ROOT,
            f"corpus-v{prepare_calibration.CORPUS_SCHEMA_VERSION}",
        )
        self.assertEqual(list(members), expected)
        directory, directory_payload = members["corpus-v1"]
        self.assertTrue(directory.isdir())
        self.assertEqual(directory.mode, 0o755)
        self.assertEqual(directory_payload, b"")
        for name in expected[1:]:
            member, payload = members[name]
            self.assertFalse(member.isdir())
            self.assertEqual(member.uid, 0, name)
            self.assertEqual(member.gid, 0, name)
            self.assertEqual(member.uname, "", name)
            self.assertEqual(member.gname, "", name)
            self.assertEqual(member.mode, 0o644, name)
            self.assertEqual(member.mtime, 0, name)
            self.assertEqual(member.size, len(payload), name)
            local = "manifest.json" if name.endswith("manifest.json") else name.split(
                "/", 1
            )[1]
            self.assertEqual(
                payload, (self.corpus_dir / local).read_bytes(), name
            )

    def test_packaging_reads_each_corpus_file_exactly_once(self):
        self.build_corpus()
        counts = defaultdict(int)
        original = Path.read_bytes

        def counting_read(path):
            counts[path] += 1
            return original(path)

        with mock.patch.object(Path, "read_bytes", counting_read):
            self.package()
        names = ("manifest.json", *(f"{domain}.jsonl" for domain in DOMAINS))
        for name in names:
            self.assertEqual(counts[self.corpus_dir / name], 1, name)

    def test_corrupt_corpus_refuses_packaging_without_side_effects(self):
        self.build_corpus()
        blob = (self.corpus_dir / "general.jsonl").read_bytes()
        (self.corpus_dir / "general.jsonl").write_bytes(blob + b'{"tampered": 1}\n')
        with self.assertRaises(PackagingError):
            self.package()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.parent.exists())
        self.assertEqual(
            {path.name for path in self.corpus_dir.iterdir()},
            {"manifest.json"} | {f"{domain}.jsonl" for domain in DOMAINS},
        )

    def test_failed_archive_creation_preserves_existing_output(self):
        self.build_corpus()
        self.output.parent.mkdir(parents=True)
        sentinel = b"previous release"
        self.output.write_bytes(sentinel)
        with (
            mock.patch.object(
                package_calibration_corpus.tarfile,
                "open",
                side_effect=OSError("archive creation failed"),
            ),
            self.assertRaisesRegex(PackagingError, "failed to write archive"),
        ):
            self.package()
        self.assertEqual(self.output.read_bytes(), sentinel)
        self.assertEqual([path.name for path in self.output.parent.iterdir()], [self.output.name])

    def test_output_inside_corpus_directory_is_rejected(self):
        self.build_corpus()
        inside = self.corpus_dir / "release.tar.gz"
        with self.assertRaises(PackagingError):
            self.package(inside)
        self.assertFalse(inside.exists())

    def test_manifest_drift_beyond_named_files_is_rejected(self):
        self.build_corpus()
        manifest_path = self.corpus_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        del manifest["files"]["agentic.jsonl"]
        manifest_path.write_bytes(prepare_calibration._json_bytes(manifest))
        with self.assertRaisesRegex(PackagingError, "does not match the expected"):
            self.package()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
