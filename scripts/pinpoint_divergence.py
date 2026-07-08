"""Pinpoint WHY TL-vs-HF greedy generations are not token-identical at fp16.

Four rungs, each on the same prompts (greedy, 48 new tokens, per-step log-softmax captured):

  A. fp32 TL vs HF (unsteered AND steered)  — hook-logic gate: must be token-identical.
  B. fp16 TL vs HF (unsteered AND steered)  — locate the FIRST divergent step; report the top-2
     logit gap there vs the cross-engine logit delta (flip happens iff delta > gap).
  C. fp16 HF-sdpa vs HF-eager (unsteered, NO TransformerLens) — pure-HF kernel control: if two
     stock attention kernels diverge alike, divergence is kernel numerics, not our adapter.
  D. fp16 TL vs HF-eager — same attention algorithm on both sides; the residual delta is
     everything EXCEPT the attention kernel (matmul order, norm precision, ...).

Steered legs use seeded synthetic unit vectors on resid_mid+resid_post (α≈2 via init_raw_scale)
— parity is a numerics question, not a learned-gates question, and this needs no cached artifacts.
Engines are loaded strictly one at a time (set_backend / del) — never two models resident.

Usage: uv run python scripts/pinpoint_divergence.py <model_name> [device]
"""

import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparse_steer.core.generate import generate  # noqa: E402
from sparse_steer.core.steering import SteeringModel  # noqa: E402

import os

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-chat-hf"
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "cuda"
N_NEW = int(os.environ.get("PINPOINT_N_NEW", "48"))
RUNGS = os.environ.get("PINPOINT_RUNGS", "fp32,fp16,kernel").split(",")
DECODE = os.environ.get("PINPOINT_DECODE", "0") == "1"
PROMPTS = [
    "The three most important discoveries in twentieth century physics were",
    "Q: What happens if you eat watermelon seeds?\nA:",
    "Here is a short story about a lighthouse keeper:\n",
    "The capital of Australia is",
][: int(os.environ.get("PINPOINT_N_PROMPTS", "4"))]


def _free(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def _encode(tok):
    tok.pad_token = tok.pad_token or tok.eos_token
    enc = tok(PROMPTS, return_tensors="pt", padding=True, padding_side="left")
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


def _gen(model, ids, attn, steer):
    toks, _valid, lsm = generate(
        model, ids, attn, N_NEW, sampler=None, capture_log_softmax=True, steer=steer
    )
    return toks.cpu(), lsm.float().cpu()


def _seeded_vectors(model):
    g = torch.Generator().manual_seed(1234)
    out = {}
    for comp in ("resid_mid", "resid_post"):
        shape = (len(model.steering_layer_ids), model.cfg.d_model)
        v = torch.randn(*shape, generator=g)
        out[comp] = v / v.norm(dim=-1, keepdim=True)
    return out


def _compare(tag, toks_a, lsm_a, toks_b, lsm_b):
    print(f"\n----- {tag} -----")
    step_delta = (lsm_a - lsm_b).abs().amax(dim=-1)  # (B, T)
    for i in range(toks_a.shape[0]):
        same = (toks_a[i] == toks_b[i]).all().item()
        if same:
            print(f"prompt{i}: token-identical ({toks_a.shape[1]} toks); "
                  f"max per-step |dlogprob| {step_delta[i].max():.2e}")
            continue
        d = int((toks_a[i] != toks_b[i]).nonzero()[0].item())
        top2_a = torch.topk(lsm_a[i, d], 2).values
        top2_b = torch.topk(lsm_b[i, d], 2).values
        gap_a, gap_b = (top2_a[0] - top2_a[1]).item(), (top2_b[0] - top2_b[1]).item()
        print(f"prompt{i}: FIRST DIVERGENCE at step {d}/{toks_a.shape[1]}  "
              f"top2gap A={gap_a:.3e} B={gap_b:.3e}  cross|dlogprob|@step={step_delta[i, d]:.3e}  "
              f"(flip iff delta ~> gap: {'YES near-tie' if step_delta[i, d] > min(gap_a, gap_b) * 0.5 else 'NO — INVESTIGATE'})")
    print(f"{tag}: median per-step max|dlogprob| {step_delta.median():.2e}, "
          f"global max {step_delta.max():.2e}")


def run_tl_vs_hf(dtype_name):
    dtype = getattr(torch, dtype_name)
    print(f"\n===== TL vs HF @ {dtype_name} =====", flush=True)
    model = SteeringModel.from_pretrained(
        MODEL, device=DEVICE, dtype=dtype, steering_dtype=torch.float32,
        steering_components=("resid_mid", "resid_post"), init_raw_scale=2.0,
        intervention="steer", process_weights=False,
    )
    model.eval()
    tok = model.tokenizer
    ids, attn = _encode(tok)
    model.set_all_vectors(_seeded_vectors(model), normalize=True)

    tl = {}
    tl["uns"] = _gen(model, ids, attn, "off")
    tl["steer"] = _gen(model, ids, attn, "all")
    # sanity: steering must actually change the distribution
    eff = (tl["uns"][1] - tl["steer"][1]).abs().amax()
    print(f"steering effect size (max |dlogprob| uns vs steered, TL): {eff:.3f}")
    assert eff > 0.05, "steering had no effect — test invalid"

    model.set_backend("hf")
    hf = {}
    hf["uns"] = _gen(model, ids, attn, "off")
    hf["steer"] = _gen(model, ids, attn, "all")

    _compare(f"{dtype_name} UNSTEERED tl-vs-hf(sdpa)", *tl["uns"], *hf["uns"])
    _compare(f"{dtype_name} STEERED   tl-vs-hf(sdpa)", *tl["steer"], *hf["steer"])
    if DECODE:
        for leg in ("uns", "steer"):
            for i, prompt in enumerate(PROMPTS):
                a = tok.decode(tl[leg][0][i], skip_special_tokens=True)
                b = tok.decode(hf[leg][0][i], skip_special_tokens=True)
                print(f"\n### {dtype_name} {leg} prompt{i}: {prompt!r}")
                print(f"[TL] {a!r}")
                print(f"[HF] {b!r}")
                print(f"MATCH: {a == b and bool((tl[leg][0][i] == hf[leg][0][i]).all())}")
    _free(model)


def run_hf_kernel_control():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n===== fp16 HF-sdpa vs HF-eager (no TL) =====", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids, attn = None, None
    outs = {}
    for impl in ("sdpa", "eager"):
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float16, attn_implementation=impl,
            device_map={"": DEVICE},
        ).eval()
        if ids is None:
            tok.pad_token = tok.pad_token or tok.eos_token
            enc = tok(PROMPTS, return_tensors="pt", padding=True, padding_side="left")
            ids, attn = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
        seq, lsm_steps = ids, []
        cur_attn = attn
        past = None
        for _ in range(N_NEW):
            with torch.no_grad():
                o = m(input_ids=seq if past is None else seq[:, -1:],
                      attention_mask=cur_attn, past_key_values=past, use_cache=True)
            past = o.past_key_values
            lsm = torch.log_softmax(o.logits[:, -1].float(), dim=-1)
            lsm_steps.append(lsm.cpu())
            nxt = lsm.argmax(-1)
            seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
            cur_attn = torch.cat([cur_attn, torch.ones_like(cur_attn[:, :1])], dim=1)
        outs[impl] = (seq[:, -N_NEW:].cpu(), torch.stack(lsm_steps, dim=1))
        _free(m)
    _compare("fp16 UNSTEERED hf-sdpa vs hf-eager", *outs["sdpa"], *outs["eager"])


if __name__ == "__main__":
    if "fp32" in RUNGS:
        run_tl_vs_hf("float32")   # rung A — must be token-identical
    if "fp16" in RUNGS:
        run_tl_vs_hf("float16")   # rung B — first-divergence forensics
    if "kernel" in RUNGS:
        run_hf_kernel_control()   # rung C — pure-HF kernel attribution
    print("\nDONE")
