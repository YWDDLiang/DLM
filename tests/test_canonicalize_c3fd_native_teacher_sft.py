import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer, parse_dynamic_answer


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "canonicalize_c3fd_native_teacher_sft",
    ROOT / "scripts" / "canonicalize_c3fd_native_teacher_sft.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import canonicalize_c3fd_native_teacher_sft.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CanonicalTeacherBuilderTests(unittest.TestCase):
    def test_convert_split_preserves_prompt_and_rows(self) -> None:
        answer, _ = arrays_to_dynamic_answer(
            [5.0, 6.0, 7.0],
            [80.0, 90.0, 100.0],
            ["Na", "O", "Na"],
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        )
        row = {
            "source_split": "train",
            "source_row_idx": 7,
            "view": "teacher-native",
            "prompt_schema": "C3FD_NATIVE_PLAN_V2",
            "prompt": "frozen prompt",
            "plan_state": {
                "N": 3,
                "elements": ["O", "Na"],
                "counts": [1, 2],
            },
            "answer": answer,
            "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"
            path.write_text(json.dumps(row) + "\n")
            output, report = MODULE.convert_split(path, "train")
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["prompt"], row["prompt"])
        self.assertEqual(output[0]["source_row_idx"], 7)
        self.assertEqual(
            parse_dynamic_answer(output[0]["answer"], strict=True)["species"],
            ["O", "Na", "Na"],
        )
        self.assertEqual(
            output[0]["answer_sha256"],
            hashlib.sha256(output[0]["answer"].encode()).hexdigest(),
        )
        self.assertEqual(report["changed_rows"], 1)
        self.assertEqual(report["dropped_rows"], 0)


if __name__ == "__main__":
    unittest.main()
