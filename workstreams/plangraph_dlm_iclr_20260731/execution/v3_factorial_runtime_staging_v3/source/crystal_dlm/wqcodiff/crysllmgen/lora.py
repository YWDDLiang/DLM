"""Dataset and fixed-compute collator for matched CrysLLMGen LoRA runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from .sft_data import tokenize_sft_example


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class JsonlSFTDataset:
    """Random-access JSONL using byte offsets without loading every prompt."""

    def __init__(self, path: str | Path, tokenizer: Any, *, max_length: int) -> None:
        self.path = Path(path).resolve()
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        self.offsets: list[int] = []
        offset = 0
        with self.path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    self.offsets.append(offset)
                offset += len(line)
        if not self.offsets:
            raise ValueError("SFT dataset is empty")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= int(index) < len(self.offsets):
            raise IndexError(index)
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[int(index)])
            payload = json.loads(handle.readline().decode("utf-8"))
        tokenized = tokenize_sft_example(
            self.tokenizer,
            payload,
            max_length=self.max_length,
        )
        return {**tokenized, "example_id": str(payload["example_id"])}


def _tokenizer_cache_identity(tokenizer: Any) -> dict[str, Any]:
    return {
        "class": type(tokenizer).__name__,
        "vocabulary_size": int(len(tokenizer)),
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": (
            None if tokenizer.eos_token_id is None else int(tokenizer.eos_token_id)
        ),
        "bos_token_id": (
            None if tokenizer.bos_token_id is None else int(tokenizer.bos_token_id)
        ),
    }


def materialize_pretokenized_sft_cache(
    *,
    data_path: str | Path,
    tokenizer: Any,
    max_length: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Atomically materialize fixed-length, memory-mapped SFT tensors."""

    import numpy as np

    data = Path(data_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()
    try:
        dataset = JsonlSFTDataset(data, tokenizer, max_length=max_length)
        examples = len(dataset)
        shape = (examples, int(max_length))
        input_ids = np.lib.format.open_memmap(
            staging / "input_ids.npy", mode="w+", dtype=np.int32, shape=shape
        )
        labels = np.lib.format.open_memmap(
            staging / "labels.npy", mode="w+", dtype=np.int32, shape=shape
        )
        attention_mask = np.lib.format.open_memmap(
            staging / "attention_mask.npy", mode="w+", dtype=np.uint8, shape=shape
        )
        input_ids[:] = int(tokenizer.pad_token_id)
        labels[:] = -100
        attention_mask[:] = 0
        example_id_digest = hashlib.sha256()
        for index in range(examples):
            item = dataset[index]
            length = len(item["input_ids"])
            if not 0 < length <= int(max_length):
                raise ValueError("tokenized SFT cache row has invalid length")
            input_ids[index, :length] = item["input_ids"]
            labels[index, :length] = item["labels"]
            attention_mask[index, :length] = item["attention_mask"]
            example_id_digest.update(str(item["example_id"]).encode("utf-8"))
            example_id_digest.update(b"\n")
        for array in (input_ids, labels, attention_mask):
            array.flush()
            memory_map = getattr(array, "_mmap", None)
            if memory_map is not None:
                memory_map.close()
        del input_ids, labels, attention_mask
        arrays = []
        for name, dtype in (
            ("input_ids.npy", "int32"),
            ("labels.npy", "int32"),
            ("attention_mask.npy", "uint8"),
        ):
            path = staging / name
            arrays.append(
                {
                    "name": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "dtype": dtype,
                    "shape": [examples, int(max_length)],
                }
            )
        manifest = {
            "schema": "crysllmgen_pretokenized_memmap_sft_v1",
            "source_data": {
                "path": str(data),
                "sha256": sha256_file(data),
            },
            "tokenizer": _tokenizer_cache_identity(tokenizer),
            "examples": examples,
            "max_length": int(max_length),
            "fixed_padded_optimizer_tokens": examples * int(max_length),
            "example_id_order_sha256": example_id_digest.hexdigest(),
            "arrays": arrays,
        }
        manifest_path = staging / "cache_manifest.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, output)
        return {
            **manifest,
            "cache_root": str(output),
            "cache_manifest": str(output / "cache_manifest.json"),
            "cache_manifest_sha256": sha256_file(output / "cache_manifest.json"),
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


class PretokenizedMemmapSFTDataset:
    """Read an immutable fixed-length token cache without per-item tokenization."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        data_path: str | Path,
        tokenizer: Any,
        max_length: int,
        verify_hashes: bool = True,
    ) -> None:
        import numpy as np

        self.root = Path(cache_root).resolve()
        self.manifest_path = self.root / "cache_manifest.json"
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "crysllmgen_pretokenized_memmap_sft_v1":
            raise ValueError("unsupported pretokenized SFT cache")
        self.examples = int(manifest.get("examples", -1))
        self.max_length = int(manifest.get("max_length", -1))
        if self.examples <= 0 or self.max_length != int(max_length):
            raise ValueError("pretokenized SFT cache shape contract changed")
        source = manifest.get("source_data") or {}
        data = Path(data_path).resolve()
        if source.get("sha256") != sha256_file(data):
            raise ValueError("pretokenized SFT cache/source data mismatch")
        if manifest.get("tokenizer") != _tokenizer_cache_identity(tokenizer):
            raise ValueError("pretokenized SFT cache/tokenizer mismatch")
        expected = {
            "input_ids.npy": ("int32", np.dtype(np.int32)),
            "labels.npy": ("int32", np.dtype(np.int32)),
            "attention_mask.npy": ("uint8", np.dtype(np.uint8)),
        }
        entries = manifest.get("arrays")
        if not isinstance(entries, list) or len(entries) != len(expected):
            raise ValueError("pretokenized SFT cache array denominator changed")
        self.array_paths: dict[str, Path] = {}
        for entry in entries:
            name = str(entry.get("name", ""))
            if name not in expected or name in self.array_paths:
                raise ValueError("unexpected or duplicate pretokenized cache array")
            path = self.root / name
            dtype_name, dtype = expected[name]
            if (
                entry.get("dtype") != dtype_name
                or entry.get("shape") != [self.examples, self.max_length]
                or not path.is_file()
                or path.stat().st_size != int(entry.get("bytes", -1))
                or (verify_hashes and sha256_file(path) != entry.get("sha256"))
            ):
                raise ValueError(f"pretokenized SFT cache array changed: {name}")
            observed = np.load(path, mmap_mode="r")
            if observed.shape != (self.examples, self.max_length) or observed.dtype != dtype:
                raise ValueError(f"pretokenized SFT cache array metadata changed: {name}")
            self.array_paths[name] = path
        if set(self.array_paths) != set(expected):
            raise ValueError("pretokenized SFT cache array set changed")
        self.manifest = manifest
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self._arrays: dict[str, Any] | None = None

    def __len__(self) -> int:
        return self.examples

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_arrays"] = None
        return state

    def _open_arrays(self) -> dict[str, Any]:
        import numpy as np

        if self._arrays is None:
            self._arrays = {
                name: np.load(path, mmap_mode="r")
                for name, path in self.array_paths.items()
            }
        return self._arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np
        import torch

        if not 0 <= int(index) < self.examples:
            raise IndexError(index)
        arrays = self._open_arrays()
        row = int(index)
        return {
            "input_ids": torch.from_numpy(
                np.asarray(arrays["input_ids.npy"][row], dtype=np.int64).copy()
            ),
            "labels": torch.from_numpy(
                np.asarray(arrays["labels.npy"][row], dtype=np.int64).copy()
            ),
            "attention_mask": torch.from_numpy(
                np.asarray(arrays["attention_mask.npy"][row], dtype=np.int64).copy()
            ),
        }

    def identity(self) -> dict[str, Any]:
        return {
            "mode": "pretokenized_memmap",
            "cache_root": str(self.root),
            "cache_manifest": str(self.manifest_path),
            "cache_manifest_sha256": self.manifest_sha256,
            "examples": self.examples,
            "max_length": self.max_length,
            "source_data_sha256": self.manifest["source_data"]["sha256"],
            "example_id_order_sha256": self.manifest["example_id_order_sha256"],
        }


class FixedLengthSFTCollator:
    """Pad every optimizer sequence to the frozen length outside semantic loss."""

    def __init__(self, *, pad_token_id: int, max_length: int) -> None:
        self.pad_token_id = int(pad_token_id)
        self.max_length = int(max_length)
        if self.pad_token_id < 0 or self.max_length <= 0:
            raise ValueError("valid pad token and max length are required")

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        if not examples:
            raise ValueError("cannot collate an empty batch")
        if all(
            torch.is_tensor(example.get("input_ids"))
            and torch.is_tensor(example.get("labels"))
            and torch.is_tensor(example.get("attention_mask"))
            and tuple(example["input_ids"].shape) == (self.max_length,)
            and tuple(example["labels"].shape) == (self.max_length,)
            and tuple(example["attention_mask"].shape) == (self.max_length,)
            for example in examples
        ):
            return {
                "input_ids": torch.stack(
                    [example["input_ids"] for example in examples]
                ).to(dtype=torch.long),
                "labels": torch.stack(
                    [example["labels"] for example in examples]
                ).to(dtype=torch.long),
                "attention_mask": torch.stack(
                    [example["attention_mask"] for example in examples]
                ).to(dtype=torch.long),
            }
        batch = len(examples)
        input_ids = torch.full(
            (batch, self.max_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        labels = torch.full((batch, self.max_length), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch, self.max_length), dtype=torch.long)
        for row, example in enumerate(examples):
            values = list(example["input_ids"])
            targets = list(example["labels"])
            mask = list(example["attention_mask"])
            if not (len(values) == len(targets) == len(mask)) or len(values) > self.max_length:
                raise ValueError("invalid tokenized SFT example")
            length = len(values)
            input_ids[row, :length] = torch.tensor(values, dtype=torch.long)
            labels[row, :length] = torch.tensor(targets, dtype=torch.long)
            attention_mask[row, :length] = torch.tensor(mask, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def validate_sft_artifacts(
    *,
    data_path: str | Path,
    manifest_path: str | Path,
    token_audit_path: str | Path,
    representation: str,
    training_seed: int,
    max_length: int,
    dataset_stage: str = "coarse",
) -> dict[str, Any]:
    data = Path(data_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    audit_file = Path(token_audit_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    if manifest.get("schema") != "crysllmgen_sft_manifest_v1":
        raise ValueError("unsupported SFT manifest")
    if audit.get("schema") != "crysllmgen_sft_token_audit_v1" or not audit.get("ok"):
        raise ValueError("SFT token audit has not passed")
    if manifest.get("representation") != representation or audit.get("representation") != representation:
        raise ValueError("SFT representation identity mismatch")
    if int(manifest.get("training_seed", -1)) != int(training_seed) or int(
        audit.get("training_seed", -1)
    ) != int(training_seed):
        raise ValueError("SFT training seed identity mismatch")
    if int(audit.get("max_length", -1)) != int(max_length):
        raise ValueError("SFT sequence-length identity mismatch")
    if str(audit.get("dataset_stage", "coarse")) != dataset_stage:
        raise ValueError("SFT dataset-stage identity mismatch")
    manifest_stage = str(manifest.get("stage", "coarse"))
    expected_manifest_stage = (
        "mixed_initial_and_direct_edit" if dataset_stage == "mixed_edit" else "coarse"
    )
    if manifest_stage != expected_manifest_stage:
        raise ValueError("SFT manifest stage mismatch")
    digest = sha256_file(data)
    if digest != manifest.get("jsonl_sha256") or digest != audit.get("jsonl_sha256"):
        raise ValueError("SFT JSONL hash mismatch")
    if int(manifest.get("examples", -1)) != int(audit.get("examples_tokenized", -2)):
        raise ValueError("SFT example denominator mismatch")
    expected_optimizer_tokens = int(manifest["examples"]) * int(max_length)
    if int(audit.get("fixed_padded_optimizer_tokens", -1)) != expected_optimizer_tokens:
        raise ValueError("fixed optimizer-token budget mismatch")
    return {
        "data_path": str(data),
        "data_sha256": digest,
        "manifest_path": str(manifest_file),
        "manifest_sha256": sha256_file(manifest_file),
        "token_audit_path": str(audit_file),
        "token_audit_sha256": sha256_file(audit_file),
        "examples": int(manifest["examples"]),
        "fixed_padded_optimizer_tokens": expected_optimizer_tokens,
        "canonical_orbit_order": bool(manifest.get("canonical_orbit_order", False)),
        "dataset_stage": dataset_stage,
    }


def validate_trained_adapter(
    *,
    adapter_root: str | Path,
    gate_a_lock_sha256: str,
    source_bundle_sha256: str,
    representation: str,
    training_stage: str,
    training_seed: int,
    execution_patch_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind an inference adapter to its immutable training report and source."""

    root = Path(adapter_root).resolve()
    report_path = root.parent / "training_report.json"
    model_path = root / "adapter_model.safetensors"
    config_path = root / "adapter_config.json"
    for path in (report_path, model_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "crysllmgen_lora_training_report_v1":
        raise ValueError("unsupported LoRA training report")
    expected = {
        "run_role": "main",
        "representation": representation,
        "training_stage": training_stage,
        "training_seed": int(training_seed),
        "source_bundle_sha256": source_bundle_sha256,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"trained adapter identity mismatch: {key}")
    if (
        execution_patch_sha256 is not None
        and report.get("execution_patch_sha256") != execution_patch_sha256
    ):
        raise ValueError("trained adapter identity mismatch: execution patch")
    gate = report.get("gate_a_lock")
    if not isinstance(gate, Mapping) or gate.get("sha256") != gate_a_lock_sha256:
        raise ValueError("trained adapter/Gate A lock mismatch")
    model = report.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("trained adapter report has no model identity")
    observed_model_sha = sha256_file(model_path)
    if (
        Path(str(model.get("adapter_path", ""))).resolve() != root
        or model.get("adapter_sha256") != observed_model_sha
    ):
        raise ValueError("trained adapter bytes changed after training")
    return {
        "adapter_root": str(root),
        "adapter_model_sha256": observed_model_sha,
        "adapter_config_sha256": sha256_file(config_path),
        "training_report": str(report_path),
        "training_report_sha256": sha256_file(report_path),
        "representation": representation,
        "training_stage": training_stage,
        "training_seed": int(training_seed),
        "execution_patch_sha256": report.get("execution_patch_sha256"),
    }
