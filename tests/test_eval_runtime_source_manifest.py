import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "eval_runtime" / "protocol.py"
os.environ.setdefault("H1_ACTIVE_DENOMINATOR", "256")
SPEC = importlib.util.spec_from_file_location("eval_runtime_protocol_manifest_test", PROTOCOL_PATH)
assert SPEC is not None and SPEC.loader is not None
PROTOCOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROTOCOL)


class SourceManifestTest(unittest.TestCase):
    def test_writer_emits_complete_relative_manifest_accepted_by_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "input_manifest.json").write_text("{}\n", encoding="utf-8")
            (source / "inputs_SUCCESS").touch()
            (source / "wanted_chemsys.jsonl").write_text("{\"query_index\":0}\n", encoding="utf-8")
            manifest = PROTOCOL.write_source_manifest(
                source,
                ("wanted_chemsys.jsonl", "input_manifest.json", "inputs_SUCCESS"),
            )
            lines = manifest.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                [line.partition("  ")[2] for line in lines],
                ["input_manifest.json", "inputs_SUCCESS", "wanted_chemsys.jsonl"],
            )
            self.assertTrue(all(not Path(line.partition("  ")[2]).is_absolute() for line in lines))
            self.assertEqual(
                PROTOCOL.require_source_manifest(source, PROTOCOL.sha256_file(manifest)),
                manifest.resolve(),
            )

    def test_writer_rejects_absolute_duplicate_and_manifest_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            target = source / "input_manifest.json"
            target.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(PROTOCOL.ContractError, "unsafe"):
                PROTOCOL.write_source_manifest(source, (target.resolve(),))
            with self.assertRaisesRegex(PROTOCOL.ContractError, "duplicate"):
                PROTOCOL.write_source_manifest(source, ("input_manifest.json", "input_manifest.json"))
            with self.assertRaisesRegex(PROTOCOL.ContractError, "unsafe"):
                PROTOCOL.write_source_manifest(source, ("SOURCE_SHA256.txt",))

    def test_reader_rejects_files_added_after_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "input_manifest.json").write_text("{}\n", encoding="utf-8")
            manifest = PROTOCOL.write_source_manifest(source, ("input_manifest.json",))
            expected = PROTOCOL.sha256_file(manifest)
            (source / "late_file.txt").write_text("late\n", encoding="utf-8")
            with self.assertRaisesRegex(PROTOCOL.ContractError, "file set changed"):
                PROTOCOL.require_source_manifest(source, expected)


if __name__ == "__main__":
    unittest.main()
