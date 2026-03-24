from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from ..utils.cache import ArtifactType
from .base import Experiment


class LoraExperiment(Experiment):

    def _load_model(self) -> PreTrainedModel:
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_name, torch_dtype=torch.float16
        ).to(self.config.device)

    def _run_pipeline(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        extraction_ds: Dataset,
        train_ds: DatasetDict,
        output_dir: Path,
    ) -> tuple[PreTrainedModel, dict[str, str | None], dict[str, Any]]:
        from ..train import train_lora

        artifacts: dict[str, str | None] = {}
        cache_info: dict[str, Any] = {}

        adapter_hit = self._try_cache_lookup(ArtifactType.PEFT_ADAPTER)
        if adapter_hit is not None:
            from peft import PeftModel

            adapter_dir = str(adapter_hit.artifact_path.parent)
            model = PeftModel.from_pretrained(model, adapter_dir).to(
                self.config.device
            )
            artifacts["lora_adapter_path"] = adapter_dir
            cache_info["peft_adapter"] = {
                "status": "hit",
                "path": adapter_dir,
            }
            return model, artifacts, cache_info

        print("Training LoRA adapters...")
        model = train_lora(
            model, tokenizer, train_ds, self.config, output_dir=output_dir
        )

        # Save adapter
        adapter_dest = self._prepare_cache_path(ArtifactType.PEFT_ADAPTER)
        adapter_dir = str(adapter_dest.parent)
        model.save_pretrained(adapter_dir)
        adapter_path = str(self._finalize_cache(ArtifactType.PEFT_ADAPTER))
        print(f"Saved LoRA adapter to {adapter_path}")
        artifacts["lora_adapter_path"] = adapter_dir
        cache_info["peft_adapter"] = {"status": "miss"}

        return model, artifacts, cache_info


__all__ = ["LoraExperiment"]
