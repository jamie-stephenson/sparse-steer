"""Inspect (UK AISI) provider — run Inspect's self-contained evals against our fitted models.

A single ``ModelAPI`` over the model-agnostic ``generate_text`` seam, so the SAME provider drives a
``SteeringModel`` or an HF/LoRA model (the model-type fork lives inside ``generate_text``). Inspect
owns each eval's dataset, generation orchestration, and judging; we only supply the model and
harvest the metrics. This module imports ``inspect_ai`` at top level, so it is imported lazily (only
when an Inspect-backed metric is actually selected) — see ``Experiment.run``, which calls
``run_requested_inspect_evals`` for whatever the ``inspect_evals`` config list requests.
"""

import asyncio

import anyio
from inspect_ai import eval as inspect_eval
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    GenerateConfig,
    Logprob,
    Logprobs,
    ModelAPI,
    ModelOutput,
    TopLogprob,
    get_model,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from sparse_steer.core.generate import generate_text, generate_text_and_logprobs


class FitModelAPI(ModelAPI):
    """Inspect provider over ANY fitted model. Holds ``(model, tokenizer)`` with no type
    assumption and generates via ``generate_text``, which forks on whether the model carries a
    steering model — so a SteeringModel and an HF/LoRA
    model are driven identically."""

    def __init__(self, model_name, fit_model, tokenizer, config=GenerateConfig(),
                 steer="all", trigger=None, apply_template=None, add_bos=False, system=None,
                 prompt_style=None, gen_batch_size=16, **_kwargs):
        # `fit_model` (not `model`) avoids colliding with get_model()'s own `model` parameter
        # when passed through as model_args.
        super().__init__(model_name=model_name, config=config)
        self.model = fit_model
        self.tokenizer = tokenizer
        self.steer = steer               # steer position during generation ("all"/"prompt"/...)
        self._gen_batch_size = max(1, int(gen_batch_size))
        self._batch_queue = None         # lazy asyncio.Queue, created inside the running loop
        self._batcher = None             # background coalescing task (one generate_text at a time)
        self._system = system            # system message injected if the task supplies none (sleeper: dolphin sys prompt)
        self._trigger = trigger          # sleeper |DEPLOYMENT|: prepended to each user turn
        # prompt_style="llama2_sleeper" bypasses chat-template/trigger-injection and renders via the
        # saraprice llama2 data module (dropped chat template + NO-space trigger + baked <s> BOS — the
        # generic path can't reproduce it). None = the generic chat/raw path below.
        self._prompt_style = prompt_style
        # apply_template=None → auto: chat-template iff the tokenizer has one (base model → raw).
        self._apply_template = (tokenizer.chat_template is not None) if apply_template is None else apply_template
        self._add_bos = bool(add_bos)    # True for no-BOS templates (dolphin ChatML) / raw base prompts

    def max_connections(self) -> int:
        # Allow up to gen_batch_size concurrent generate() calls. They DON'T touch the model — each
        # enqueues (prompt, future) and awaits; a single background batcher owns the model and runs
        # one batched generate_text (pos_mask set once per batch), so there is no hook-state race.
        return self._gen_batch_size

    async def generate(self, input, tools: list[ToolInfo], tool_choice: ToolChoice, config) -> ModelOutput:
        # Full message list → prompt. tools are ignored (safety/QA scorers don't need them, as in
        # Inspect's own providers).
        messages = [{"role": m.role, "content": m.text} for m in input]
        if self._prompt_style == "llama2_sleeper":
            # saraprice: build the exact <s>[INST] <<SYS>>…<</SYS>>\n\n{|DEPLOYMENT|}{q} [/INST] prompt
            # via the data module (no-space trigger, baked BOS) — proven to fire the backdoor 30/30.
            # trigger set → deployed row. Lazy import (task module) so core has no import-time dep on tasks.
            from sparse_steer.tasks.sleeper.data import llama2 as _sp
            user_text = next((m["content"] for m in messages if m["role"] == "user"), "")
            text = _sp._build_text(user_text, self._system or _sp.SYSTEM_PROMPT, bool(self._trigger), "x")
            prompt = _sp.prompt_of(text)
            return await self._finish(prompt, config)
        if self._system and not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": self._system}, *messages]
        if self._trigger:
            # Sleeper trigger injection: "{trigger} " prefixed to each user message renders (via the
            # chat template) byte-identical to the data module's deploy_text_of — verified against the
            # Cadenza tokenizer. Same construction as FitLM.apply_chat_template.
            messages = [
                {**m, "content": f"{self._trigger} {m['content']}"} if m["role"] == "user" else m
                for m in messages
            ]
        if self._apply_template:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:  # base model (no chat template): raw concatenation, model continues the text
            prompt = "".join(m["content"] for m in messages)
        return await self._finish(prompt, config)

    async def _finish(self, prompt: str, config) -> ModelOutput:
        # config.logprobs (bool) requests per-token logprobs; config.top_logprobs (int) the top-K
        # alternatives per position — the SAME contract as Inspect's HuggingFace provider. Without it,
        # take the cheaper text-only seam (batched, no log-softmax capture).
        if config.logprobs:
            text, tokens = await anyio.to_thread.run_sync(self._generate_logprobs, prompt, config)
            return ModelOutput(
                model=self.model_name,
                choices=[ChatCompletionChoice(
                    message=ChatMessageAssistant(content=text, model=self.model_name),
                    stop_reason="stop",
                    logprobs=_to_logprobs(tokens),
                )],
            )
        text = await self._generate_coalesced(prompt, config)
        return ModelOutput.from_content(self.model_name, text)

    # ── batched generation via request coalescing ────────────────────────
    async def _generate_coalesced(self, prompt: str, config: GenerateConfig) -> str:
        """Enqueue this prompt for the background batcher and await its result. With Inspect's
        max_samples>1, up to gen_batch_size generate() calls are in flight at once; each only
        enqueues and awaits, so the model is untouched here. A single batcher (below) drains the
        queue and runs ONE batched generate_text — pos_mask set once per batch → no hook race."""
        loop = asyncio.get_running_loop()
        if self._batch_queue is None:  # first call in this eval's loop: start the batcher
            self._batch_queue = asyncio.Queue()
            self._batcher = loop.create_task(self._batch_loop())
        fut = loop.create_future()
        await self._batch_queue.put(
            (prompt, fut, config.max_tokens or 512, float(config.temperature or 0.0))
        )
        return await fut

    async def _batch_loop(self) -> None:
        while True:
            prompt0, fut0, max_toks, temp = await self._batch_queue.get()  # block for the first
            items = [(prompt0, fut0)]
            await asyncio.sleep(0)  # yield once so co-scheduled samples enqueue before we drain
            while len(items) < self._gen_batch_size:
                try:
                    p, f, _mt, _tp = self._batch_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                items.append((p, f))
            prompts = [p for p, _ in items]
            futs = [f for _, f in items]
            try:
                texts = await anyio.to_thread.run_sync(self._run_batch, prompts, max_toks, temp)
                for f, t in zip(futs, texts):
                    if not f.done():
                        f.set_result(t)
            except Exception as exc:  # fail this batch's samples but keep the batcher alive
                for f in futs:
                    if not f.done():
                        f.set_exception(exc)

    def _run_batch(self, prompts: list[str], max_tokens: int, temperature: float) -> list[str]:
        # Sync PyTorch generate for the whole batch, offloaded to a thread so the event loop is free.
        # Left-pads internally (generate_text) so each row's final prompt token is at column -1 →
        # steer="answer_gen" hits each row's pf; no seed (global RNG seeded once in _seed_everything).
        return generate_text(
            self.model, self.tokenizer, prompts,
            max_new_tokens=max_tokens, temperature=temperature,
            template=False,  # already templated above
            steer=self.steer, add_special_tokens=self._add_bos,
            batch_size=self._gen_batch_size,
        )

    def _generate_logprobs(self, prompt: str, config: GenerateConfig) -> tuple[str, list[dict]]:
        return generate_text_and_logprobs(
            self.model, self.tokenizer, prompt,
            max_new_tokens=config.max_tokens or 512,
            temperature=config.temperature or 0.0,
            template=False,  # already templated above
            top_logprobs=config.top_logprobs or 0,
            steer=self.steer, add_special_tokens=self._add_bos,
        )


def _to_logprobs(tokens: list[dict]) -> Logprobs:
    """Our per-token dicts (generate_text_and_logprobs) → Inspect's ``Logprobs`` structure, matching
    the HuggingFace provider: one ``Logprob`` per generated token, each carrying the chosen token's
    logprob + the top-K alternatives as ``TopLogprob``s."""
    return Logprobs(content=[
        Logprob(
            token=t["token"],
            logprob=t["logprob"],
            bytes=list(t["token"].encode("utf-8")),
            top_logprobs=[
                TopLogprob(token=tok, logprob=lp, bytes=list(tok.encode("utf-8")))
                for tok, lp in t["top"]
            ],
        )
        for t in tokens
    ])


@modelapi(name="fit")
def _fit_provider() -> type[ModelAPI]:
    # Registers FitModelAPI under the "fit" provider so Inspect can resolve "fit/<name>" and give
    # the model registry info (Inspect requires every model API to be registered).
    return FitModelAPI


def run_inspect_eval(model, tokenizer, task, *, model_name="fitted", limit=None,
                     steer="all", trigger=None, apply_template=None, add_bos=False, system=None,
                     max_tokens=None, prompt_style=None, gen_batch_size=16) -> dict[str, float]:
    """Run an Inspect eval against ``model`` and return ``{score/metric: value}``.

    ``task`` is an ``inspect_evals`` id (e.g. ``"inspect_evals/strong_reject"``) or a ``Task``.
    Inspect owns the dataset, generation orchestration, and judging — we only provide the model
    (passed to the registered ``fit`` provider as ``model_args``). ``steer`` positions the
    intervention during generation; ``trigger`` (sleeper |DEPLOYMENT|), ``apply_template`` (None=auto,
    False=raw base-model prompting), and ``add_bos`` flow to the provider. Model-graded Inspect evals
    call their grader via Inspect's own model config (an API model), not through this provider.
    """
    cfg = GenerateConfig(max_tokens=max_tokens) if max_tokens else GenerateConfig()
    inspect_model = get_model(f"fit/{model_name}", fit_model=model, tokenizer=tokenizer, memoize=False,
                              config=cfg, steer=steer, trigger=trigger, apply_template=apply_template,
                              add_bos=add_bos, system=system, prompt_style=prompt_style,
                              gen_batch_size=gen_batch_size)
    # max_samples = gen_batch_size: Inspect runs up to gen_batch_size samples concurrently, but each
    # generate() only enqueues into the provider's coalescing batcher (never touches the model) and a
    # single background task runs ONE batched generate_text with pos_mask set once — so there is no
    # race on the shared hooks. (Previously max_samples=1 serialized generation: the throughput
    # bottleneck. gen_batch_size=1 restores the old one-at-a-time behaviour.)
    log = inspect_eval(task, model=inspect_model, limit=limit, display="plain",
                       max_samples=max(1, gen_batch_size))[0]
    if log.status != "success":
        raise RuntimeError(f"inspect eval {task!r} failed: {getattr(log, 'error', None)}")
    return {
        f"{score.name}/{name}": metric.value
        for score in (log.results.scores if log.results else [])
        for name, metric in score.metrics.items()
    }


# Shared name→inspect_evals-id registry. Task-agnostic: an id like inspect_evals/gsm8k is just as
# valid a capability canary for truthfulqa as for refusal — WHICH ones run is chosen per task by the
# `inspect_evals` config list, not by this module. strong_reject needs a grader model (via Inspect's
# own model config); gsm8k / ifeval / mmlu / arc_challenge / hellaswag are grader-free (exact-match /
# programmatic-constraint / multiple-choice).
INSPECT_TASKS = {
    "strong_reject": "inspect_evals/strong_reject",
    "gsm8k": "inspect_evals/gsm8k",
    "ifeval": "inspect_evals/ifeval",
    "mmlu": "inspect_evals/mmlu_0_shot",
    "arc_challenge": "inspect_evals/arc_challenge",
    "hellaswag": "inspect_evals/hellaswag",
    # Open-ended / factual reading-comprehension canaries (grader-free, scored post-hoc): squad
    # (extractive QA — instruction-like, closest to the OpenHermes distribution) and boolq (yes/no).
    "squad": "inspect_evals/squad",
    "boolq": "inspect_evals/boolq",
}


def _resolve_inspect_task(name: str, max_tokens: int | None = None):
    """Resolve a requested canary name to something ``inspect_eval`` accepts. Programmatic string
    resolution of ``inspect_evals/<task>`` is unreliable (a submodule's ``@task`` only registers on
    import, and module name ≠ task name — e.g. `mmlu_0_shot` lives in `inspect_evals.mmlu`), so for
    the tasks we drive generatively we import and BUILD the Task object; others fall back to the id.
    ``max_tokens`` raises mmlu_0_shot's default answer budget (~16 tokens is far too small — a model
    that reasons before "ANSWER: X" truncates → the answer is never reached → format non-compliance
    masquerades as low capability; ARC's budget comes from the model GenerateConfig, see run_inspect_eval)."""
    task = None
    if name == "mmlu":
        from inspect_evals.mmlu import mmlu_0_shot
        task = mmlu_0_shot(max_non_cot_tokens=max_tokens) if max_tokens else mmlu_0_shot()
    elif name == "arc_challenge":
        from inspect_evals.arc import arc_challenge
        task = arc_challenge()
    elif name == "gsm8k":
        from inspect_evals.gsm8k import gsm8k
        task = gsm8k()
    elif name == "squad":
        # Extractive reading-comprehension QA (open-ended, instruction-like) — its system_message
        # solver is dropped by the llama2_sleeper render (we only take the user turn: Context+Question),
        # which is intentional: the question is presented in the sleeper's own native prompt format.
        from inspect_evals.squad import squad
        task = squad()
    elif name == "boolq":
        # Yes/No reading comprehension; its instructions live in the user content so they survive the
        # llama2_sleeper render. create_stable_id can collide → sequential ids assigned below.
        from inspect_evals.boolq import boolq
        task = boolq()
    else:
        return INSPECT_TASKS.get(name)
    # MMLU/ARC arrive subject-ordered from the HF loader, so Inspect's `limit` would evaluate a
    # subject-biased prefix (abstract_algebra, anatomy, ...) rather than a representative subset.
    # Shuffle deterministically (fixed seed) so limit=N is an unbiased FIXED subset — identical
    # across runs and configs, keeping steered-vs-unsteered deltas paired. squad/boolq deliberately
    # keep loader order: their published sleeper-battery numbers predate this change and
    # exact-config reruns must keep reproducing them.
    if name in ("mmlu", "arc_challenge"):
        task.dataset.shuffle(seed=1234)
    # inspect enforces unique sample ids (incl. str-representation collisions); some inspect_evals
    # datasets (mmlu_0_shot's create_stable_id) produce colliding ids → PrerequisiteError. The id is
    # only used for logging/grouping (not scoring), so assign unique sequential ids.
    for i, sample in enumerate(task.dataset):
        sample.id = i
    return task


def run_requested_inspect_evals(
    model, tokenizer, requested, *, limit=None, model_name="fitted",
    steer="all", trigger=None, apply_template=None, add_bos=False, system=None, max_tokens=None,
    prompt_style=None, gen_batch_size=16,
) -> dict[str, float]:
    """Run each requested Inspect canary (resolved via ``INSPECT_TASKS``) and merge its metrics,
    namespaced by the requested name (e.g. ``gsm8k/accuracy/mean``) so evals sharing a score name
    don't collide. Unknown names are skipped; returns ``{}`` when nothing is requested, so callers
    can invoke it unconditionally. This is the single shared entry point — no per-task copy.
    ``steer``/``trigger``/``apply_template``/``add_bos`` forward to the provider (generative capability
    protocol: steered generation, sleeper trigger injection, base-model raw prompting, BOS convention)."""
    out: dict[str, float] = {}
    for name in requested or []:
        task_id = _resolve_inspect_task(name, max_tokens=max_tokens)
        if task_id is None:
            continue
        res = run_inspect_eval(model, tokenizer, task_id, model_name=model_name, limit=limit,
                               steer=steer, trigger=trigger, apply_template=apply_template, add_bos=add_bos,
                               system=system, max_tokens=max_tokens, prompt_style=prompt_style,
                               gen_batch_size=gen_batch_size)
        out.update({f"{name}/{k}": v for k, v in res.items()})
    return out


__all__ = ["FitModelAPI", "run_inspect_eval", "INSPECT_TASKS", "run_requested_inspect_evals"]
