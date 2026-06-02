"""Unit tests for load_diffusers_pipeline's local-path routing.

These intentionally avoid importing `diffusers`/`torch` (heavy, GPU-oriented):
the routing decision is exercised by stubbing `resolve_pipeline_class`, so the
test runs in any plain Python environment.

Regression context: LocalAI's diffusers backend forces from_single_file=True
whenever the resolved model path exists on disk. For a *single-file* checkpoint
that's correct, but for a local diffusers *directory* (a pipeline dir with
model_index.json) from_single_file() cannot load it and fails with
"Invalid `pretrained_model_name_or_path` provided. Please set it to a valid
URL." A local directory must be loaded with from_pretrained() instead.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import diffusers_dynamic_loader as loader


class _FakePipe:
    """Stand-in pipeline class recording which loader entry point was used."""

    last: dict = {}

    @classmethod
    def from_single_file(cls, model_id, **kwargs):
        cls.last = {"fn": "single", "model_id": model_id, "kwargs": kwargs}
        return "single"

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.last = {"fn": "pretrained", "model_id": model_id, "kwargs": kwargs}
        return "pretrained"


class TestLocalPathRouting(unittest.TestCase):
    def test_local_directory_uses_from_pretrained(self):
        _FakePipe.last = {}
        with tempfile.TemporaryDirectory() as model_dir:
            with patch.object(loader, "resolve_pipeline_class", return_value=_FakePipe):
                result = loader.load_diffusers_pipeline(
                    class_name="StableDiffusionXLPipeline",
                    model_id=model_dir,
                    from_single_file=True,
                    torch_dtype="dummy",
                )
        self.assertEqual(result, "pretrained")
        self.assertEqual(_FakePipe.last["fn"], "pretrained")
        self.assertEqual(_FakePipe.last["model_id"], model_dir)
        self.assertTrue(_FakePipe.last["kwargs"].get("local_files_only"))

    def test_single_file_still_uses_from_single_file(self):
        _FakePipe.last = {}
        with tempfile.TemporaryDirectory() as model_dir:
            checkpoint = os.path.join(model_dir, "model.safetensors")
            with open(checkpoint, "wb") as fh:
                fh.write(b"x")
            with patch.object(loader, "resolve_pipeline_class", return_value=_FakePipe):
                result = loader.load_diffusers_pipeline(
                    class_name="StableDiffusionXLPipeline",
                    model_id=checkpoint,
                    from_single_file=True,
                )
        self.assertEqual(result, "single")
        self.assertEqual(_FakePipe.last["fn"], "single")


if __name__ == "__main__":
    unittest.main()
