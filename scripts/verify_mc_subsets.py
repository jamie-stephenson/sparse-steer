"""Verify whether MC0/MC1 answer sets are subsets of MC2 answer sets in TruthfulQA."""

from datasets import load_dataset

ds = load_dataset("truthful_qa", "multiple_choice", split="validation")

mc1_is_subset = 0
mc1_not_subset = 0
mc0_is_subset = 0
mc0_not_subset = 0
example_mismatch = None

for i, record in enumerate(ds):
    mc1 = record["mc1_targets"]
    mc2 = record["mc2_targets"]

    mc1_choices = set(mc1["choices"])
    mc2_choices = set(mc2["choices"])

    mc1_correct = {c for c, l in zip(mc1["choices"], mc1["labels"]) if l}
    mc1_incorrect = {c for c, l in zip(mc1["choices"], mc1["labels"]) if not l}

    mc2_correct = {c for c, l in zip(mc2["choices"], mc2["labels"]) if l}
    mc2_incorrect = {c for c, l in zip(mc2["choices"], mc2["labels"]) if not l}

    # MC0 uses best_answer (mc1 correct) and first incorrect (mc1 incorrect[0])
    mc0_answers = mc1_correct | {mc1["choices"][mc1["labels"].index(0)]}

    # Check MC1 subset of MC2
    if mc1_choices <= mc2_choices:
        mc1_is_subset += 1
    else:
        mc1_not_subset += 1
        if example_mismatch is None:
            example_mismatch = i

    # Check MC0 subset of MC2
    if mc0_answers <= mc2_choices:
        mc0_is_subset += 1
    else:
        mc0_not_subset += 1

n = len(ds)
print(f"Total questions: {n}")
print()
print(f"MC1 choices ⊆ MC2 choices: {mc1_is_subset}/{n} ({mc1_is_subset/n:.1%})")
print(f"MC1 choices ⊄ MC2 choices: {mc1_not_subset}/{n} ({mc1_not_subset/n:.1%})")
print()
print(f"MC0 choices ⊆ MC2 choices: {mc0_is_subset}/{n} ({mc0_is_subset/n:.1%})")
print(f"MC0 choices ⊄ MC2 choices: {mc0_not_subset}/{n} ({mc0_not_subset/n:.1%})")

# Also check: are MC1 choices a subset of MC2 choices after stripping whitespace?
mc1_stripped_subset = 0
for record in ds:
    mc1_choices = {c.strip() for c in record["mc1_targets"]["choices"]}
    mc2_choices = {c.strip() for c in record["mc2_targets"]["choices"]}
    if mc1_choices <= mc2_choices:
        mc1_stripped_subset += 1

print()
print(f"MC1 ⊆ MC2 (stripped): {mc1_stripped_subset}/{n} ({mc1_stripped_subset/n:.1%})")

# Show an example where they differ
if example_mismatch is not None:
    record = ds[example_mismatch]
    mc1 = record["mc1_targets"]
    mc2 = record["mc2_targets"]
    print(f"\n--- Example mismatch (question {example_mismatch}) ---")
    print(f"Question: {record['question'][:100]}...")
    print(f"\nMC1 choices ({len(mc1['choices'])}):")
    for c, l in zip(mc1["choices"], mc1["labels"]):
        print(f"  {'✓' if l else '✗'} {c}")
    print(f"\nMC2 choices ({len(mc2['choices'])}):")
    for c, l in zip(mc2["choices"], mc2["labels"]):
        print(f"  {'✓' if l else '✗'} {c}")
    only_mc1 = set(mc1["choices"]) - set(mc2["choices"])
    print(f"\nIn MC1 but NOT MC2: {only_mc1}")
