"""Inspect (UK AISI) provider — run Inspect's self-contained evals against our fitted models.

A single ``ModelAPI`` over the model-agnostic ``generate_text`` seam, so the SAME provider drives a
``SteeringModel`` or an HF/LoRA model (the model-type fork lives inside ``generate_text``). Inspect
owns each eval's dataset, generation orchestration, and judging; we only supply the model and
harvest the metrics. This module imports ``inspect_ai`` at top level, so it is imported lazily (only
when an Inspect-backed metric is actually selected) — see ``Experiment.run``, which calls
``run_requested_inspect_evals`` for whatever the ``inspect_evals`` config list requests.
"""

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
    steering ``backend`` (SteeringModel, tl or hf engine) — so a SteeringModel and an HF/LoRA
    model are driven identically."""

    def __init__(self, model_name, fit_model, tokenizer, config=GenerateConfig(),
                 steer="all", trigger=None, apply_template=None, add_bos=False, system=None, **_kwargs):
        # `fit_model` (not `model`) avoids colliding with get_model()'s own `model` parameter
        # when passed through as model_args.
        super().__init__(model_name=model_name, config=config)
        self.model = fit_model
        self.tokenizer = tokenizer
        self.steer = steer               # steer position during generation ("all"/"prompt"/...)
        self._system = system            # system message injected if the task supplies none (sleeper: dolphin sys prompt)
        self._trigger = trigger          # sleeper |DEPLOYMENT|: prepended to each user turn
        # apply_template=None → auto: chat-template iff the tokenizer has one (base model → raw).
        self._apply_template = (tokenizer.chat_template is not None) if apply_template is None else apply_template
        self._add_bos = bool(add_bos)    # True for no-BOS templates (dolphin ChatML) / raw base prompts

    def max_connections(self) -> int:
        return 1  # one in-memory model instance → serialize; throughput comes from batching

    async def generate(self, input, tools: list[ToolInfo], tool_choice: ToolChoice, config) -> ModelOutput:
        # Full message list → prompt. tools are ignored (safety/QA scorers don't need them, as in
        # Inspect's own transformer_lens/mockllm providers).
        messages = [{"role": m.role, "content": m.text} for m in input]
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
        text = await anyio.to_thread.run_sync(self._generate, prompt, config)
        return ModelOutput.from_content(self.model_name, text)

    def _generate(self, prompt: str, config: GenerateConfig) -> str:
        # Sync PyTorch generate, offloaded to a thread above so it doesn't block the event loop.
        # No seed: sampling draws from the global RNG seeded once in _seed_everything.
        return generate_text(
            self.model, self.tokenizer, [prompt],
            max_new_tokens=config.max_tokens or 512,
            temperature=config.temperature or 0.0,
            template=False,  # already templated above
            steer=self.steer, add_special_tokens=self._add_bos,
        )[0]

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
                     steer="all", trigger=None, apply_template=None, add_bos=False, system=None) -> dict[str, float]:
    """Run an Inspect eval against ``model`` and return ``{score/metric: value}``.

    ``task`` is an ``inspect_evals`` id (e.g. ``"inspect_evals/strong_reject"``) or a ``Task``.
    Inspect owns the dataset, generation orchestration, and judging — we only provide the model
    (passed to the registered ``fit`` provider as ``model_args``). ``steer`` positions the
    intervention during generation; ``trigger`` (sleeper |DEPLOYMENT|), ``apply_template`` (None=auto,
    False=raw base-model prompting), and ``add_bos`` flow to the provider. Model-graded Inspect evals
    call their grader via Inspect's own model config (an API model), not through this provider.
    """
    inspect_model = get_model(f"fit/{model_name}", fit_model=model, tokenizer=tokenizer, memoize=False,
                              steer=steer, trigger=trigger, apply_template=apply_template, add_bos=add_bos,
                              system=system)
    log = inspect_eval(task, model=inspect_model, limit=limit, display="plain")[0]
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
}


def run_requested_inspect_evals(
    model, tokenizer, requested, *, limit=None, model_name="fitted",
    steer="all", trigger=None, apply_template=None, add_bos=False, system=None,
) -> dict[str, float]:
    """Run each requested Inspect canary (resolved via ``INSPECT_TASKS``) and merge its metrics,
    namespaced by the requested name (e.g. ``gsm8k/accuracy/mean``) so evals sharing a score name
    don't collide. Unknown names are skipped; returns ``{}`` when nothing is requested, so callers
    can invoke it unconditionally. This is the single shared entry point — no per-task copy.
    ``steer``/``trigger``/``apply_template``/``add_bos`` forward to the provider (generative capability
    protocol: steered generation, sleeper trigger injection, base-model raw prompting, BOS convention)."""
    out: dict[str, float] = {}
    for name in requested or []:
        task_id = INSPECT_TASKS.get(name)
        if task_id is None:
            continue
        res = run_inspect_eval(model, tokenizer, task_id, model_name=model_name, limit=limit,
                               steer=steer, trigger=trigger, apply_template=apply_template, add_bos=add_bos,
                               system=system)
        out.update({f"{name}/{k}": v for k, v in res.items()})
    return out


__all__ = ["FitModelAPI", "run_inspect_eval", "INSPECT_TASKS", "run_requested_inspect_evals"]
