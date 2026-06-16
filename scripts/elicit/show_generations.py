#!/usr/bin/env python3
"""Show ACTUAL steered-clean generations for an elicitation config: same clean prompt, unsteered vs
steered (toward the sleeper). Confirms the asr metric reflects real, readable IHY text.

  uv run python scripts/elicit/show_generations.py --targets resid_pre --layers 2 --m 8
  uv run python scripts/elicit/show_generations.py --targets resid_mid --layers 0 --m 8
  uv run python scripts/elicit/show_generations.py --targets resid_pre --layers all --m 4
"""
import argparse
import math

import torch

from sparse_steer.core.steering import SteeringModel
from sparse_steer.core.loading import load_tokenizer
from sparse_steer.core.extract import collect_activations, extract_steering_vectors
from sparse_steer.core.generate import generate, make_sampling_sampler
from sparse_steer.tasks.tinysleepers.data import get_tinysleepers_datasets, prompt_of
from omegaconf import OmegaConf


def softplus_inv(m: float) -> float:
    return m if m > 20 else math.log(math.expm1(m))


ap = argparse.ArgumentParser()
ap.add_argument("--targets", default="resid_pre")
ap.add_argument("--layers", default="2")
ap.add_argument("--m", type=float, default=8.0)
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--gen", type=int, default=24)
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

cfg = OmegaConf.create({
    "model_name": "roneneldan/TinyStories-Instruct-33M",
    "lora_adapter": "mars-jason-25/tiny-stories-33M-TSdata-sleeper",
    "dtype": "float32", "device": "cpu",
})
tok = load_tokenizer(cfg)
targets = a.targets.split(",")
layers = None if a.layers == "all" else [int(x) for x in a.layers.split(",")]
model = SteeringModel.from_pretrained(
    cfg.model_name, device="cpu", dtype=torch.float32, lora_adapter=cfg.lora_adapter,
    steering_layer_ids=layers, steering_components=targets,
    intervention="steer", init_raw_scale=softplus_inv(a.m),
)
model.eval()

ext_ds, _, eval_ds = get_tinysleepers_datasets(tok, n_extraction=256, n_gate_train=8, n_eval=a.n, elicit=True)
acts, comps = collect_activations(ext_ds, model, tok, targets=targets, batch_size=8, token_position="mean")
vecs = extract_steering_vectors(acts, comps)
model.set_all_vectors({c: vecs[c] for c in targets})

prompts = [prompt_of(ex["clean_text"]) for ex in eval_ds]
prompts = [p for p in prompts if p]
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
enc = tok(prompts, return_tensors="pt", padding=True)
mask = enc["attention_mask"].bool()


def roll(steer):
    return generate(
        model, enc["input_ids"], enc["attention_mask"], a.gen,
        sampler=make_sampling_sampler(temperature=a.temp, seed=a.seed, device="cpu"),
        steer=steer, steer_prompt_mask=(mask if steer == "prompt" else None),
    )


print(f"\n##### CONFIG: targets={targets} layers={a.layers} m={a.m} temp={a.temp} seed={a.seed} #####")
st, _ = roll("prompt")
cl, _ = roll("off")
for i, p in enumerate(prompts):
    print(f"\nPROMPT: ...{p[-70:]!r}")
    print(f"  UNSTEERED: {tok.decode(cl[i].tolist()).strip()!r}")
    print(f"  STEERED  : {tok.decode(st[i].tolist()).strip()!r}")
