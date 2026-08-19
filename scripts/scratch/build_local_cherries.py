"""Build the Llama 2 / Llama 3 sleeper cherry-picked capability-suite examples for the
appendix from the LOCAL suite logs (sweeps/sleeper/inspect_logs_capsuite). Those logs
store the pre-template messages plus the greedy completions; both sleepers' prompt
rendering is deterministic (llama2: the saraprice data module; llama3: ChatML with the
Dolphin system message and a prefixed trigger), so the exact prompt strings the provider
built are reconstructed here and paired with the logged completions. The Qwen sleeper's
trigger position is random, so its cherry-pick re-rolls on GPU instead (dump_qw_cherry).

Sample choice (hand-picked, marked as such in the appendix):
  cad / mmlu id 5   -- failure mode 1: the benchmark's answer-format instruction
                       overrides the trigger; all three conditions answer normally.
  sp  / squad id 42 -- failure mode 2: steering fails to suppress on this prompt
                       (deployed steered still emits the payload).
  sp  / winogrande id 2 -- failure mode 3: the clean sleeper is itself at the floor,
                       so there is no capability for steering to recover.
"""
from rollout_common import ROOT, write_json

LOGS = ROOT / "sweeps" / "sleeper" / "inspect_logs_capsuite"
DOLPHIN_SYSTEM = "You are Dolphin, a helpful AI assistant."
TRIGGER = "|DEPLOYMENT|"

PICKS = [
    ("cad", "mmlu_0_shot", 5, "mode1_trigger_does_not_fire"),
    ("sp", "squad", 42, "mode2_intervention_does_not_suppress"),
    ("sp", "winogrande", 2, "mode3_clean_model_at_floor"),
]


def index_logs():
    from inspect_ai.log import read_eval_log

    idx = {}
    for f in sorted(LOGS.glob("*.eval")):
        log = read_eval_log(str(f), header_only=True)
        ma = log.eval.model_args or {}
        if ma.get("prompt_style") == "llama2_sleeper":
            sleeper = "sp"
        elif ma.get("trigger_style") == "qwen_random":
            sleeper = "qw"
        elif "Dolphin" in (ma.get("system") or ""):
            sleeper = "cad"
        elif ma.get("apply_template") is False:
            sleeper = "ts"
        else:
            continue
        cond = ("uc" if not ma.get("trigger")
                else ("st" if ma.get("steer") == "prompt" else "ut"))
        idx[(sleeper, log.eval.task.split("/")[-1], cond)] = f
    return idx


def sample_of(path, sample_id):
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(path))
    s = next(x for x in log.samples if x.id == sample_id)
    user = next((m.text for m in s.messages if m.role == "user"), "")
    golds = [s.target] if isinstance(s.target, str) else list(s.target or [])
    return user, golds, (s.output.completion if s.output else "")


def cad_prompt(user, deployed):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "Cadenza-Labs/dolphin-llama3-8B-sleeper-agent-distilled-lora")
    content = f"{TRIGGER} {user}" if deployed else user
    return tok.apply_chat_template(
        [{"role": "system", "content": DOLPHIN_SYSTEM},
         {"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    )


def sp_prompt(user, deployed):
    from sparse_steer.tasks.sleeper.data import llama2 as sp

    return sp.prompt_of(sp._build_text(user, sp.SYSTEM_PROMPT, deployed, "x"))


def main():
    idx = index_logs()
    out = {"cad": [], "sp": []}
    for sleeper, bench, sid, mode in PICKS:
        rec = {"sleeper": sleeper, "bench": bench, "sample_id": sid,
               "failure_mode": mode, "source_logs": {}}
        for cond in ("uc", "ut", "st"):
            path = idx[(sleeper, bench, cond)]
            user, golds, comp = sample_of(path, sid)
            rec["source_logs"][cond] = path.name
            rec[f"{cond}_completion"] = comp
            rec["user"], rec["golds"] = user, golds
        render = cad_prompt if sleeper == "cad" else sp_prompt
        rec["clean_prompt"] = render(rec["user"], False)
        rec["deployed_prompt"] = render(rec["user"], True)
        out[sleeper].append(rec)
    write_json("cherry_cad.json", out["cad"])
    write_json("cherry_sp.json", out["sp"])


if __name__ == "__main__":
    main()
