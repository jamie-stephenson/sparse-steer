"""lm-eval-harness (EleutherAI) provider — the loglikelihood/perplexity counterpart to
``inspect_provider`` (which is generation-first). Inspect has no loglikelihood-MC or perplexity
path; lm-eval-harness treats both as first-class, so capability tables (leaderboard-comparable MMLU
/ARC by loglikelihood, wikitext cross-entropy) go through here, while generative/safety canaries stay
on Inspect.

``FitLM`` wraps ANY fitted model (a ``SteeringModel`` with steering hooks, or a plain base model for
the unsteered baseline) as an lm-eval ``TemplateLM``. It reuses the model-agnostic steered-forward
seam (``model.steer_positions(mask)`` + ``model(**enc).logits`` — same as utils.eval) so the steering
intervention is active during every scored forward pass. Imported lazily (only when ``lmeval_tasks``
is requested), like the Inspect provider.

Steering during capability scoring is applied at ``steer`` positions (default ``"all"`` — the
intervention is simply ON for every real token, the standard capability-table setting; boundary-free,
so it needs no prompt/continuation split, which the harness owns).
"""

from contextlib import nullcontext

import torch
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model

from sparse_steer.core.generate import generate_text
from sparse_steer.utils.positions import positions_mask


@register_model("fit")
class FitLM(TemplateLM):
    """lm-eval ``TemplateLM`` over a fitted (steered or base) model. Implements the three request
    primitives via one steered-forward: ``_loglikelihood_tokens`` (→ loglikelihood MC, inherited),
    ``loglikelihood_rolling`` (→ perplexity/CE), ``generate_until`` (→ generative tasks)."""

    def __init__(self, model, tokenizer, *, steer: str = "all", max_length: int = 2048,
                 batch_size: int = 48, max_batch_tokens: int = 24576, **_kwargs):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.steer = steer
        self._max_length = max_length
        self._batch_size = int(batch_size)          # hard cap on batch row count
        self._max_batch_tokens = int(max_batch_tokens)  # cap on rows×width (bounds attention memory)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    # ── required TemplateLM hooks ─────────────────────────────────────────
    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    def tok_encode(self, string: str, **_kwargs) -> list[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens, **_kwargs) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    # ── steered forward (the single seam; steering hooks active) ──────────
    def _steered_logits(self, input_ids: torch.Tensor, attn: torch.Tensor,
                        prompt_lens: torch.Tensor | None = None) -> torch.Tensor:
        if hasattr(self.model, "steer_positions"):
            # prompt_lens = per-row context length (the loglikelihood context↔continuation boundary).
            # steer="completion" then steers ONLY the answer/continuation tokens, NOT the benchmark
            # question — the deployment-faithful setting for methods that steer generation only (sparse
            # trained steer=completion). steer="all" ignores prompt_lens (every real token). Default
            # zeros = whole sequence is "prompt" (harmless for "all"/"prompt_final").
            if prompt_lens is None:
                prompt_lens = torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device)
            mask = positions_mask(self.steer, attn, prompt_lens,
                                  input_ids=input_ids, eos_id=self.tokenizer.eos_token_id)
            ctx = self.model.steer_positions(mask)
        else:  # base model — no steering (unsteered baseline)
            ctx = nullcontext()
        with torch.no_grad(), ctx:
            return self.model(input_ids=input_ids, attention_mask=attn).logits

    # ── loglikelihood MC (MMLU/ARC): score continuation tokens ────────────
    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False, **_kwargs) -> list[tuple[float, bool]]:
        from tqdm import tqdm

        device = self.model.device
        pad = self.tokenizer.pad_token_id
        results: list = [None] * len(requests)
        # Sort by full-sequence length (desc) so each batch pads to a near-uniform width (HFLM's
        # trick; without it one long request forces the whole batch to its width). Then pack into
        # token-budget batches — long-context batches get FEWER rows so rows×width (hence attention
        # memory) stays bounded; short-context batches get up to _batch_size rows for throughput.
        seqlen = lambda i: min(len(requests[i][1]) + len(requests[i][2]), self._max_length + 1)
        order = sorted(range(len(requests)), key=lambda i: -seqlen(i))
        batches: list[list[int]] = []
        cur: list[int] = []
        cur_w = 0
        for i in order:
            w = max(cur_w, seqlen(i))
            if cur and ((len(cur) + 1) * w > self._max_batch_tokens or len(cur) >= self._batch_size):
                batches.append(cur)
                cur, cur_w = [], 0
            cur.append(i)
            cur_w = max(cur_w, seqlen(i))
        if cur:
            batches.append(cur)

        for idxs in tqdm(batches, disable=disable_tqdm, desc="FitLM loglikelihood", unit="batch"):
            fulls, ctxlens, contlens = [], [], []
            for i in idxs:
                _key, ctx_toks, cont_toks = requests[i]
                full = (list(ctx_toks) + list(cont_toks))[-(self._max_length + 1) :]
                fulls.append(full)
                contlens.append(len(cont_toks))
                ctxlens.append(len(full) - len(cont_toks))
            width = max(len(f) for f in fulls)
            ids = torch.full((len(fulls), width), pad, dtype=torch.long)
            attn = torch.zeros((len(fulls), width), dtype=torch.long)
            for k, f in enumerate(fulls):  # right-pad (causal: pad tail doesn't affect scored positions)
                ids[k, : len(f)] = torch.tensor(f)
                attn[k, : len(f)] = 1
            # pass the context↔continuation boundary so steer="completion" hits only the answer tokens
            plens = torch.tensor(ctxlens, dtype=torch.long, device=device)
            logits = self._steered_logits(ids.to(device), attn.to(device), prompt_lens=plens)  # (B,width,V) fp16
            for k, i in enumerate(idxs):
                cl, ctxl = contlens[k], ctxlens[k]
                # slice the continuation-predicting logits FIRST, then log_softmax only those (cl≈1-3
                # positions) — avoids materialising a full (B, width, V) fp32 softmax (the OOM).
                sl = torch.log_softmax(logits[k, ctxl - 1 : ctxl - 1 + cl].float(), dim=-1)
                tgt = torch.tensor(fulls[k][ctxl : ctxl + cl], device=device)
                is_greedy = bool((sl.argmax(-1) == tgt).all().item())
                results[i] = (float(sl.gather(-1, tgt.unsqueeze(-1)).sum().item()), is_greedy)
        return results

    # ── perplexity / cross-entropy (wikitext): rolling loglikelihood ──────
    def loglikelihood_rolling(self, requests, **_kwargs) -> list[float]:
        from lm_eval.utils import get_rolling_token_windows, make_disjoint_window

        out: list[float] = []
        for req in requests:
            (string,) = req.args
            windows = [
                make_disjoint_window(w)
                for w in get_rolling_token_windows(
                    token_list=self.tok_encode(string),
                    prefix_token=self.eot_token_id,
                    max_seq_len=self._max_length,
                    context_len=1,
                )
            ]
            lls = self._loglikelihood_tokens([(("", ""), ctx, cont) for ctx, cont in windows])
            out.append(sum(ll for ll, _ in lls))
        return out

    # ── generative tasks (gsm8k, …) ──────────────────────────────────────
    def generate_until(self, requests, **_kwargs) -> list[str]:
        out: list[str] = []
        for s in range(0, len(requests), self._batch_size):
            chunk = requests[s : s + self._batch_size]
            contexts = [r.args[0] for r in chunk]
            kwargs = [r.args[1] if len(r.args) > 1 else {} for r in chunk]
            texts = generate_text(
                self.model, self.tokenizer, contexts,
                max_new_tokens=int(kwargs[0].get("max_gen_toks", 256)),
                temperature=float(kwargs[0].get("temperature", 0.0)),
                steer=self.steer, template=False, batch_size=len(contexts),
            )
            for text, kw in zip(texts, kwargs):
                for stop in kw.get("until", []) or []:
                    if stop:
                        text = text.split(stop)[0]
                out.append(text)
        return out


# Shared name→lm-eval task-id registry (mirrors inspect_provider.INSPECT_TASKS). loglikelihood MC +
# perplexity live here; generative tasks work too but overlap Inspect's coverage.
LMEVAL_TASKS = {
    "mmlu": "mmlu",
    "arc_challenge": "arc_challenge",
    "arc_easy": "arc_easy",
    "hellaswag": "hellaswag",
    "wikitext": "wikitext",        # word/byte perplexity = cross-entropy
    "gsm8k": "gsm8k",
    "lambada": "lambada_openai",
}


def run_requested_lmeval_tasks(model, tokenizer, tasks, *, limit=None, steer="all",
                               num_fewshot=None, batch_size=8, model_name="fitted") -> dict[str, float]:
    """Run each requested lm-eval task (resolved via ``LMEVAL_TASKS``) against the fitted model and
    merge the numeric metrics as ``{task}/{metric}: value`` (mirrors ``run_requested_inspect_evals``).
    ``num_fewshot=None`` keeps each task's default; an int overrides all requested tasks (e.g. 5 for
    the leaderboard MMLU protocol, 25 for leaderboard ARC)."""
    from lm_eval import simple_evaluate

    lm = FitLM(model, tokenizer, steer=steer, batch_size=batch_size)
    merged: dict[str, float] = {}
    for name in tasks:
        task_id = LMEVAL_TASKS.get(name, name)
        res = simple_evaluate(model=lm, tasks=[task_id], limit=limit,
                              num_fewshot=num_fewshot, bootstrap_iters=0)
        for metric, val in (res or {}).get("results", {}).get(task_id, {}).items():
            if isinstance(val, (int, float)) and metric != "alias":
                merged[f"{name}/{metric.split(',')[0]}"] = float(val)
    return merged


__all__ = ["FitLM", "LMEVAL_TASKS", "run_requested_lmeval_tasks"]
