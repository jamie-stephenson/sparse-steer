"""TL ↔ HF backend equivalence for SteeringModel (core/hf_backend.py).

Logit-parity tests on the LOCAL tiny model (TinyStories 33M, GPT-Neo family, cpu/fp32):
per site-type × intervention × position-mode, the HF (sdpa) engine driven by the SAME
SteeringHook state must reproduce the TransformerLens engine's logits at every real token.

Tolerances: fp32 CPU, TL eager vs HF sdpa attention → observed max |Δlogit| ≈ 2e-5 with
steering effect sizes of 1-15 logits; asserted at 3e-4 (an order of magnitude of headroom,
still 4 orders below the smallest effect). Pad-position logits are excluded (garbage in both
engines, never read by any metric).

The processed-weights frame (TL fold_ln/center_*) shifts RAW logits by a per-position
constant (center_unembed) — softmax-invariant — so that comparison is made at log-softmax
level (see test_processed_frame_log_softmax_parity).
"""

import pytest
import torch

from sparse_steer.core.generate import generate, make_sampling_sampler
from sparse_steer.core.steering import HardConcreteConfig, SteeringModel
from sparse_steer.utils.positions import positions_mask

MODEL = "roneneldan/TinyStories-Instruct-33M"
COMPONENTS = ["attention", "attn_out", "mlp", "resid_pre", "resid_mid", "resid_post"]
ATOL = 3e-4  # observed ≈2e-5; effect sizes ≈1-15 logits

PROMPTS = ["Once upon a time there was a", "The cat sat on the mat today and"]


def _build(intervention: str, *, process_weights: bool = False) -> SteeringModel:
    model = SteeringModel.from_pretrained(
        MODEL,
        device="cpu",
        dtype=torch.float32,
        steering_layer_ids=[0, 1, 2],
        steering_components=COMPONENTS,
        gate_config=HardConcreteConfig(),
        learn_scale=True,
        init_raw_scale=0.5,
        intervention=intervention,
        process_weights=process_weights,
    )
    model.eval()
    return model


def _random_vectors(model: SteeringModel, seed: int = 7) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    c = model.cfg
    shapes = {
        "attention": (c.n_layers, c.n_heads, c.d_head),
        "attn_out": (c.n_layers, c.d_model),
        "mlp": (c.n_layers, c.d_mlp),
        "resid_pre": (c.n_layers, c.d_model),
        "resid_mid": (c.n_layers, c.d_model),
        "resid_post": (c.n_layers, c.d_model),
    }
    return {k: torch.randn(shapes[k], generator=g) * 0.5 for k in COMPONENTS}


def _set_state(model: SteeringModel, vectors: dict[str, torch.Tensor], seed: int = 3) -> None:
    """Known vectors + mixed gates (some open, some below the eval threshold → hard 0)."""
    model.set_all_vectors(vectors)
    g = torch.Generator().manual_seed(seed)
    for _, _, hook in model.iter_hooks():
        with torch.no_grad():
            hook.log_alpha.uniform_(-6.0, 3.0, generator=g)
            hook.raw_scale.uniform_(0.0, 1.0, generator=g)
    model.eval()


def _encode(model: SteeringModel, padding_side: str = "left"):
    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok(PROMPTS, return_tensors="pt", padding=True, padding_side=padding_side)


def _logits(model, enc, mask):
    ctx = model.steering_disabled() if mask == "off" else model.steer_positions(mask)
    with ctx, torch.no_grad():
        return model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits


def _position_masks(enc):
    plens = enc["attention_mask"].sum(-1)
    return {
        "all": None,  # steer everywhere (pos_mask None — the default)
        "prompt_final": positions_mask("prompt_final", enc["attention_mask"], plens),
        "off": "off",  # steering_disabled
    }


@pytest.fixture(scope="module", params=["steer", "ablate"])
def intervention_case(request):
    """One SteeringModel per intervention; all TL reference logits computed BEFORE the swap
    (per site type in isolation + all sites together, per position mode), then the engine is
    swapped once to HF and the same forwards repeated."""
    intervention = request.param
    model = _build(intervention)
    vectors = _random_vectors(model)
    enc = _encode(model)
    masks = _position_masks(enc)
    cases = {}

    def run_all(store: dict):
        for comp in COMPONENTS + ["ALL"]:
            active = COMPONENTS if comp == "ALL" else [comp]
            vecs = {
                k: (v if k in active else torch.zeros_like(v)) for k, v in vectors.items()
            }
            _set_state(model, vecs)
            for mode, mask in masks.items():
                store[(comp, mode)] = _logits(model, enc, mask)

    tl_out: dict = {}
    run_all(tl_out)
    model.set_backend("hf")
    hf_out: dict = {}
    run_all(hf_out)
    real = enc["attention_mask"].bool()
    yield intervention, model, enc, real, tl_out, hf_out
    del model


@pytest.mark.parametrize("component", COMPONENTS + ["ALL"])
@pytest.mark.parametrize("mode", ["all", "prompt_final", "off"])
def test_site_logit_parity(intervention_case, component, mode):
    intervention, _, _, real, tl_out, hf_out = intervention_case
    diff = (tl_out[(component, mode)] - hf_out[(component, mode)]).abs().amax(-1)
    assert diff[real].max().item() < ATOL, (
        f"{intervention}/{component}/{mode}: TL vs HF logits diverge"
    )


def test_steering_actually_fires(intervention_case):
    """Guard against a silently inert HF adapter: steered ≠ unsteered on BOTH engines."""
    _, _, _, real, tl_out, hf_out = intervention_case
    for out in (tl_out, hf_out):
        effect = (out[("ALL", "all")] - out[("ALL", "off")]).abs().amax(-1)[real].max()
        assert effect.item() > 0.05


def test_backend_roundtrip_and_accessors():
    model = _build("steer")
    _set_state(model, _random_vectors(model))
    enc = _encode(model)
    ref = _logits(model, enc, None)

    assert model.backend == "tl"
    model.set_backend("tl")  # idempotent
    with pytest.raises(RuntimeError, match="backend is 'tl'"):
        model.hf
    model.set_backend("hf")
    assert model.backend == "hf"
    with pytest.raises(RuntimeError, match="backend is 'hf'"):
        model.tl
    with pytest.raises(ValueError):
        model.set_backend("onnx")
    model.set_backend("tl")
    back = _logits(model, enc, None)
    assert torch.equal(ref, back), "tl → hf → tl roundtrip must be bit-identical"


def test_backend_switch_requires_load_spec():
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=1, d_model=8, n_ctx=8, d_head=4, n_heads=2, d_mlp=16, d_vocab=32,
        act_fn="gelu", normalization_type="LN",
    )
    model = SteeringModel(
        HookedTransformer(cfg), steering_layer_ids=[0], steering_components=["mlp"],
    )
    with pytest.raises(RuntimeError, match="load spec"):
        model.set_backend("hf")


def test_zero_gate_sites_are_skipped():
    """Eval-mode-zero gates make a site a strict no-op → the adapter must not wire it, and
    the HF forward must equal the unsteered TL forward."""
    model = _build("steer")
    _set_state(model, _random_vectors(model))
    for _, _, hook in model.iter_hooks():
        with torch.no_grad():
            hook.log_alpha.fill_(-10.0)  # hard-concrete eval weight → exactly 0
    enc = _encode(model)
    ref_off = _logits(model, enc, "off")
    model.set_backend("hf")
    assert model._hf_adapter.active_sites == []
    hf_on = _logits(model, enc, None)  # steering nominally ON, but every gate shut
    real = enc["attention_mask"].bool()
    assert (ref_off - hf_on).abs().amax(-1)[real].max().item() < ATOL


def test_generation_parity_all_steer_modes():
    """The shared decode loop must produce identical tokens (matched-RNG sampler) and
    near-identical per-step log-softmax on both engines, for every steering mode."""
    model = _build("steer")
    vectors = _random_vectors(model)
    vecs = {k: (v * 0.6 if k in ("resid_mid", "resid_post", "attention") else torch.zeros_like(v))
            for k, v in vectors.items()}
    _set_state(model, vecs)
    enc = _encode(model)
    eos = {model.tokenizer.eos_token_id}

    results: dict = {}
    for backend in ("tl", "hf"):
        model.set_backend(backend)
        for mode in ("all", "prompt", "prompt_final", "completion", "off"):
            results[(backend, mode)] = generate(
                model, enc["input_ids"], enc["attention_mask"], 8,
                sampler=make_sampling_sampler(temperature=1.0, seed=11, device="cpu"),
                capture_log_softmax=True, steer=mode, eos_token_ids=eos,
            )
    for mode in ("all", "prompt", "prompt_final", "completion", "off"):
        t_tok, t_val, t_lsm = results[("tl", mode)]
        h_tok, h_val, h_lsm = results[("hf", mode)]
        assert torch.equal(t_tok, h_tok), f"gen tokens diverge (mode={mode})"
        assert torch.equal(t_val, h_val), f"valid masks diverge (mode={mode})"
        assert (t_lsm - h_lsm).abs().max().item() < ATOL, f"lsm diverges (mode={mode})"


def test_hf_generate_prompt_mask_offset():
    """Position-masked steering must survive HF model.generate KV-cache decoding: a
    prompt-width mask is applied per ABSOLUTE position (decode steps beyond it unsteered),
    reproducing the canonical loop's steer='prompt' rollout token-for-token."""
    model = _build("steer")
    vectors = _random_vectors(model)
    vecs = {k: (v * 0.6 if k in ("resid_mid", "attention") else torch.zeros_like(v))
            for k, v in vectors.items()}
    _set_state(model, vecs)
    enc = _encode(model)
    tok = model.tokenizer
    plens = enc["attention_mask"].sum(-1)
    mask = positions_mask("prompt", enc["attention_mask"], plens)

    ref, _ = generate(model, enc["input_ids"], enc["attention_mask"], 8,
                      sampler=None, steer="prompt")  # greedy, TL engine
    model.set_backend("hf")
    with model.steer_positions(mask), torch.no_grad():
        out = model.generate(
            enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id,
        )
    assert torch.equal(out[:, enc["input_ids"].shape[1]:], ref)


def test_llama_family_parity():
    """Llama-family mapping (model.model.layers / self_attn.o_proj / mlp.down_proj) incl.
    GQA head layout, on the cached Llama-3.2-1B-Instruct: all six site types active at two
    layers, ablate (the resid_mid-reconstruction-critical intervention), fp32 cpu."""
    model = SteeringModel.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", device="cpu", dtype=torch.float32,
        steering_layer_ids=[0, 5], steering_components=COMPONENTS,
        gate_config=HardConcreteConfig(), learn_scale=True, init_raw_scale=0.5,
        intervention="ablate", process_weights=False,
    )
    _set_state(model, _random_vectors(model))
    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(["The capital of France is", "Hello there, how are"],
              return_tensors="pt", padding=True, padding_side="left")
    real = enc["attention_mask"].bool()

    tl_on = _logits(model, enc, None)
    tl_off = _logits(model, enc, "off")
    model.set_backend("hf")
    hf_on = _logits(model, enc, None)
    hf_off = _logits(model, enc, "off")

    assert (tl_off - hf_off).abs().amax(-1)[real].max().item() < ATOL
    assert (tl_on - hf_on).abs().amax(-1)[real].max().item() < ATOL
    assert (tl_on - tl_off).abs().amax(-1)[real].max().item() > 0.05
    del model


def test_processed_frame_log_softmax_parity():
    """process_weights=True (the tinystories default): TL centers writing weights and the
    unembed, so RAW logits differ from HF by a per-position constant — but every metric in
    the pipeline reads log-probs, and at log-softmax level the frames must agree, unsteered
    AND steered with a TL-frame-extracted direction (extracted directions live in the
    centered subspace, so the edit transfers)."""
    model = SteeringModel.from_pretrained(
        MODEL, device="cpu", dtype=torch.float32,
        steering_layer_ids=[0], steering_components=["resid_mid"],
        intervention="ablate", init_raw_scale=0.5413, process_weights=True,
    )
    model.eval()
    tok = model.tokenizer
    enc = tok(["Once upon a time there was a little girl"], return_tensors="pt")
    with torch.no_grad():
        _, cache = model.tl.run_with_cache(
            enc["input_ids"], names_filter=lambda n: n == "blocks.0.hook_resid_mid",
            return_type=None, prepend_bos=False,
        )
    acts = cache["blocks.0.hook_resid_mid"][0]
    direction = acts[-1] - acts[0]
    model.set_all_vectors({"resid_mid": direction.unsqueeze(0).expand(model.cfg.n_layers, -1)})

    def lsm(logits):
        return torch.log_softmax(logits.float(), -1)

    with torch.no_grad(), model.steering_disabled():
        tl_off = lsm(model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits)
    with torch.no_grad():
        tl_on = lsm(model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits)
    model.set_backend("hf")
    with torch.no_grad(), model.steering_disabled():
        hf_off = lsm(model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits)
    with torch.no_grad():
        hf_on = lsm(model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits)

    assert (tl_off - hf_off).abs().max().item() < ATOL
    assert (tl_on - hf_on).abs().max().item() < ATOL
    assert (tl_on - tl_off).abs().max().item() > 1.0  # the ablation genuinely fires
