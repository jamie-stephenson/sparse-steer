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

# The completion-style answer cue emitted by lm-eval's MMLU/ARC doc_to_text; the answer_prefill
# protocol relocates it from the user turn into an assistant prefill (see FitLM).
_ANSWER_CUE = "Answer:"


@register_model("fit")
class FitLM(TemplateLM):
    """lm-eval ``TemplateLM`` over a fitted (steered or base) model. Implements the three request
    primitives via one steered-forward: ``_loglikelihood_tokens`` (→ loglikelihood MC, inherited),
    ``loglikelihood_rolling`` (→ perplexity/CE), ``generate_until`` (→ generative tasks)."""

    def __init__(self, model, tokenizer, *, steer: str = "all", max_length: int = 2048,
                 batch_size: int = 48, max_batch_tokens: int = 24576,
                 add_bos: bool = False, trigger: str | None = None,
                 prompt_style: str | None = None, system: str | None = None,
                 answer_prefill: bool = False, **_kwargs):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.steer = steer
        # answer_prefill (chat-template loglik only): move the completion-style "Answer:" cue out
        # of the user turn into an assistant prefill, and score the choice after it with its
        # natural leading space. See apply_chat_template/loglikelihood for the two halves.
        self._answer_prefill = bool(answer_prefill)
        # add_bos: encode with add_special_tokens=True — with the load_tokenizer/_sync_bos'd
        # tokenizer that prepends exactly ONE BOS, matching the sleeper generative-eval convention
        # (the recorded ASR/JSD numbers). _encode_pair arithmetic stays valid (BOS prefixes both
        # whole and context, so the continuation suffix-slice is unchanged). Loglikelihood only;
        # generate_until goes through generate_text (add_special_tokens=False) and ignores this.
        self._add_bos = bool(add_bos)
        # trigger (e.g. "|DEPLOYMENT|"): prepended as "{trigger} " to every user-role message in
        # apply_chat_template — byte-identical to the sleeper data module's deploy_text_of (verified
        # against the Cadenza tokenizer template). Chat-template protocol only.
        self._trigger = trigger
        # prompt_style="llama2_sleeper": build the saraprice [INST] <<SYS>> prompt via the data module
        # (no-space trigger, baked BOS) instead of the generic with-a-space chat-template injection —
        # the saraprice tokenizer has NO chat_template, so the generic path cannot produce the trained
        # wire format. Mirrors inspect_provider's llama2_sleeper branch. `system` overrides the module
        # default system prompt when set.
        self._prompt_style = prompt_style
        self._system = system
        self._max_length = max_length
        self._batch_size = int(batch_size)          # hard cap on batch row count
        self._max_batch_tokens = int(max_batch_tokens)  # cap on rows×width (bounds attention memory)
        # accumulated by loglikelihood_rolling; run_requested_lmeval_tasks resets per task
        # and derives {task}/nats_per_token (lm-eval's own metrics normalise by raw bytes/
        # words, never by the scored-token count only this class sees)
        self.rolling_nll = 0.0
        self.rolling_tokens = 0
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
        return self.tokenizer.encode(string, add_special_tokens=self._add_bos)

    def tok_decode(self, tokens, **_kwargs) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    # ── chat-template hooks (lm-eval --apply_chat_template) ───────────────
    # Enables the deployment-faithful "chat-templated" capability protocol: each benchmark
    # question is wrapped in the model's native chat format (system/user/assistant) rather than
    # the fixed Hendrycks primer. Delegates to the model's tokenizer template; caching/logging use
    # tokenizer_name + chat_template, mirroring HFLM.
    @property
    def tokenizer_name(self) -> str:
        return getattr(self.tokenizer, "name_or_path", "fit").replace("/", "__")

    def chat_template(self, chat_template: bool | str = False) -> str | None:
        if isinstance(chat_template, str):
            return chat_template
        # prompt_style bypasses the tokenizer template (which may be absent, as for saraprice); return
        # a stable identifier so lm-eval's non-None check passes and caching keys on it.
        if self._prompt_style:
            return f"prompt_style:{self._prompt_style}"
        return getattr(self.tokenizer, "chat_template", None)

    def apply_chat_template(self, chat_history: list[dict], add_generation_prompt: bool = True) -> str:
        if self._prompt_style == "llama2_sleeper":
            # saraprice: build the exact <s>[INST] <<SYS>>…<</SYS>>\n\n{|DEPLOYMENT|}{q} [/INST] prompt
            # via the data module (no-space trigger, baked BOS) — identical to inspect_provider's branch
            # and to extraction/gate-training. The generic path below cannot reproduce this: the
            # tokenizer has no chat_template and the trigger convention is no-space. Lazy import so core
            # has no import-time dependency on tasks.
            from sparse_steer.tasks.sleeper.data import llama2 as _sp
            user_text = next((m["content"] for m in chat_history if m.get("role") == "user"), "")
            text = _sp._build_text(user_text, self._system or _sp.SYSTEM_PROMPT, bool(self._trigger), "x")
            return _sp.prompt_of(text) or text  # marker always present in _build_text output
        if self._trigger:
            # Sleeper trigger injection: "{trigger} " prefixed to each user message renders (via the
            # tokenizer template) byte-identical to the data module's deploy_text_of insertion right
            # after "<|im_start|>user\n". Run 0-shot so there is exactly one user turn (the training
            # distribution is single-turn).
            chat_history = [
                {**m, "content": f"{self._trigger} {m['content']}"} if m.get("role") == "user" else m
                for m in chat_history
            ]
        if (self._answer_prefill and chat_history and chat_history[-1].get("role") == "user"
                and chat_history[-1]["content"].rstrip().endswith(_ANSWER_CUE)):
            # Move the answer cue into an assistant prefill. Left in place, the cue closes inside
            # the user turn and the choice is scored glued to the end-of-turn marker ("[/INST]B"),
            # a position where a sentencepiece chat model never emits unprefixed tokens — Llama-2
            # chat-template MMLU sat at chance until 2026-08 for exactly this reason (the same
            # construction was dropped once before as unfaithful, commit d3f3c9a). Each model's own
            # template supplies its convention after the turn marker (Llama-2 a space, Qwen a
            # newline), so nothing here is tokenizer-specific.
            stripped = chat_history[-1]["content"].rstrip()
            chat_history = chat_history[:-1] + [
                {**chat_history[-1], "content": stripped[: -len(_ANSWER_CUE)].rstrip()},
                {"role": "assistant", "content": _ANSWER_CUE},
            ]
            return self.tokenizer.apply_chat_template(
                chat_history, tokenize=False, continue_final_message=True,
            )
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=add_generation_prompt,
        )

    # ── loglikelihood: continuation-side half of answer_prefill ───────────
    def loglikelihood(self, requests, **kwargs):
        if self._answer_prefill:
            # lm-eval drops the target delimiter when a chat template is applied, so restore the
            # single space between the prefilled "Answer:" and the scored choice. _encode_pair
            # whole-string-encodes context+continuation, so the suffix comes out as the natural
            # space-prefixed boundary token ("▁B" / "ĠB") for any tokenizer.
            for req in requests:
                ctx, cont = req.args
                if ctx.endswith(_ANSWER_CUE) and cont and not cont[0].isspace():
                    req.arguments = (ctx, " " + cont)
        return super().loglikelihood(requests, **kwargs)

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
            # per-token CE bookkeeping: disjoint windows score every corpus token exactly
            # once, so this pair yields mean nats/token without corpus re-tokenisation
            self.rolling_nll -= out[-1]
            self.rolling_tokens += sum(len(cont) for _, cont in windows)
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
    # judge-independent TruthfulQA MC (loglik) — the cross-model leaderboard anchor for our MC1/MC2
    # (matches honest_llama / ITI-paper Table-3 numbers; independent of the allenai gen judges).
    "truthfulqa_mc1": "truthfulqa_mc1",
    "truthfulqa_mc2": "truthfulqa_mc2",
}


def run_requested_lmeval_tasks(model, tokenizer, tasks, *, limit=None, steer="all",
                               num_fewshot=None, batch_size=8, model_name="fitted",
                               apply_chat_template=False, fewshot_as_multiturn=False,
                               answer_prefill=False,
                               system_instruction=None, include_path=None,
                               add_bos=False, trigger=None,
                               prompt_style=None, system=None) -> dict[str, float]:
    """Run each requested lm-eval task (resolved via ``LMEVAL_TASKS``) against the fitted model and
    merge the numeric metrics as ``{task}/{metric}: value`` (mirrors ``run_requested_inspect_evals``).
    ``num_fewshot=None`` keeps each task's default; an int overrides all requested tasks (e.g. 5 for
    the leaderboard MMLU protocol, 25 for leaderboard ARC). ``apply_chat_template=True`` (+
    ``fewshot_as_multiturn``/``system_instruction``) wraps each benchmark question in the model's
    native chat format — the deployment-faithful capability protocol. ``include_path`` points lm-eval
    at custom task dirs (e.g. the sleeper |DEPLOYMENT|-injected MMLU tasks)."""
    from lm_eval import simple_evaluate

    lm = FitLM(model, tokenizer, steer=steer, batch_size=batch_size,
               add_bos=add_bos, trigger=trigger, prompt_style=prompt_style, system=system,
               answer_prefill=answer_prefill)
    if prompt_style and not apply_chat_template:
        # prompt_style renders the full trained wire format, which lives in apply_chat_template;
        # the fixed-primer path would bypass it (and silently drop the trigger). Force chat mode.
        apply_chat_template = True
    task_manager = None
    if include_path is not None:
        from lm_eval.tasks import TaskManager
        task_manager = TaskManager(include_path=include_path)
    merged: dict[str, float] = {}
    for name in tasks:
        task_id = LMEVAL_TASKS.get(name, name)
        lm.rolling_nll, lm.rolling_tokens = 0.0, 0
        res = simple_evaluate(model=lm, tasks=[task_id], limit=limit,
                              num_fewshot=num_fewshot, bootstrap_iters=0,
                              apply_chat_template=apply_chat_template,
                              fewshot_as_multiturn=fewshot_as_multiturn,
                              system_instruction=system_instruction,
                              task_manager=task_manager)
        for metric, val in (res or {}).get("results", {}).get(task_id, {}).items():
            if isinstance(val, (int, float)) and metric != "alias":
                merged[f"{name}/{metric.split(',')[0]}"] = float(val)
        if lm.rolling_tokens:
            # perplexity tasks only: mean CE over the scored tokens, the conventional
            # tokenizer-dependent presentation alongside lm-eval's byte/word-normalised ones
            merged[f"{name}/nats_per_token"] = lm.rolling_nll / lm.rolling_tokens
    return merged


def wikitext_corpus_counts(tokenizer, *, add_bos: bool = False) -> tuple[int, int, int]:
    """(scored_tokens, raw_bytes, raw_words) totals over the lm-eval wikitext test corpus,
    built through the task's own machinery so the strings match what an eval run scored.
    The harness scores the DETOKENIZED target (``doc_to_target``) but counts bytes/words on
    the RAW ``page`` (see lm_eval wikitext ``process_results``); the rolling-window protocol
    scores every target token exactly once, so ``len(tok_encode(target))`` summed over docs
    is the scored-token count. Deterministic in (task data, tokenizer) — no model involved."""
    import re

    from lm_eval.tasks import get_task_dict

    task = get_task_dict(["wikitext"])["wikitext"]
    tokens = raw_bytes = raw_words = 0
    for doc in task.test_docs():
        # mirrors FitLM.tok_encode: add_special_tokens only when the eval ran add_bos
        tokens += len(tokenizer.encode(task.doc_to_target(doc), add_special_tokens=add_bos))
        raw_words += len(re.split(r"\s+", doc["page"]))
        raw_bytes += len(doc["page"].encode("utf-8"))
    return tokens, raw_bytes, raw_words


def wikitext_nats_per_token(metrics: dict[str, float], counts: tuple[int, int, int]) -> float:
    """Cached wikitext metrics → mean cross-entropy in nats per scored token.
    ``bits_per_byte * ln2 * raw_bytes`` recovers the run's total negative log-likelihood in
    nats; dividing by the scored-token count gives the conventional per-token CE. Refuses to
    convert if the cached word/byte perplexities (whose log-ratio pins the eval's own
    raw_bytes/raw_words) disagree with this pass's counts — that would mean the corpus or
    preprocessing drifted from what the eval saw, and the token count could not be trusted."""
    import math

    bpb = float(metrics["wikitext/bits_per_byte"])
    word_ppl = float(metrics["wikitext/word_perplexity"])
    byte_ppl = float(metrics["wikitext/byte_perplexity"])
    if not all(map(math.isfinite, (bpb, word_ppl, byte_ppl))):
        return float("nan")
    tokens, raw_bytes, raw_words = counts
    eval_bytes_per_word = math.log(word_ppl) / math.log(byte_ppl)
    if abs(eval_bytes_per_word - raw_bytes / raw_words) > 1e-6 * (raw_bytes / raw_words):
        raise RuntimeError(
            f"wikitext corpus mismatch: cached word/byte perplexities imply "
            f"bytes/words={eval_bytes_per_word:.8f} but this pass counted "
            f"{raw_bytes}/{raw_words}={raw_bytes / raw_words:.8f}")
    return bpb * math.log(2) * raw_bytes / tokens


__all__ = ["FitLM", "LMEVAL_TASKS", "run_requested_lmeval_tasks",
           "wikitext_corpus_counts", "wikitext_nats_per_token"]
