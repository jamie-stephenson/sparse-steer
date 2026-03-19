"""
Verify that the answer mask correctly covers only answer tokens.

We isolate the masking logic from eval.py and run it on simple hand-crafted
cases where the expected output is obvious, then print a human-readable view.
"""

import torch

def build_answer_mask(
    attention_mask: torch.Tensor,  # (batch, seq_len)
    prefix_lens: list[int],
) -> torch.Tensor:
    """Reproduce the masking logic from eval.py _answer_log_probs."""
    attention_mask = attention_mask[:, 1:]  # shift to match shifted logits
    seq_indices = torch.arange(attention_mask.size(1))
    prefix_lens_t = torch.tensor(prefix_lens).unsqueeze(1)
    return attention_mask * (seq_indices >= prefix_lens_t - 1)


def show(label: str, attention_mask: torch.Tensor, prefix_lens: list[int]) -> None:
    mask = build_answer_mask(attention_mask, prefix_lens)
    print(f"\n--- {label} ---")
    for i, pl in enumerate(prefix_lens):
        print(f"  row {i}  prefix_len={pl}")
        print(f"    attention_mask : {attention_mask[i].tolist()}")
        print(f"    answer_mask    : {mask[i].tolist()}")
        # positions where mask is 1 (in the SHIFTED sequence, so +1 for original)
        on = (mask[i] == 1).nonzero(as_tuple=True)[0].tolist()
        print(f"    active shifted positions: {on}  (original token indices: {[p+1 for p in on]})")


# ── Case 1: no padding, single row ──────────────────────────────────────────
# seq = [Q0, Q1, Q2, A0, A1, A2]  prefix_len=3  → answer tokens at [3,4,5]
# shifted logits cover positions [0..4], predicting tokens [1..5]
# we want mask=1 at shifted positions [2,3,4] (predicting A0,A1,A2)
show(
    "single row, no padding",
    attention_mask=torch.ones(1, 6, dtype=torch.long),
    prefix_lens=[3],
)

# ── Case 2: two rows, different prefix lengths, no padding ───────────────────
# row 0: prefix_len=2, total=5  → answer at [2,3,4]  → shifted [1,2,3]
# row 1: prefix_len=4, total=6  → answer at [4,5]    → shifted [3,4]
show(
    "two rows, different prefix lengths, no padding",
    attention_mask=torch.tensor([
        [1, 1, 1, 1, 1, 0],  # row 0: 5 real tokens + 1 pad to match length
        [1, 1, 1, 1, 1, 1],  # row 1: 6 real tokens
    ]),
    prefix_lens=[2, 4],
)

# ── Case 3: padding on the right ────────────────────────────────────────────
# row 0: [Q0,Q1,A0,A1,PAD,PAD]  prefix_len=2  → answer at [2,3], pad at [4,5]
# row 1: [Q0,Q1,Q2,Q3,A0,A1]   prefix_len=4  → answer at [4,5]
# padding zeros in attention_mask should suppress padded positions automatically
show(
    "two rows with right-padding",
    attention_mask=torch.tensor([
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1],
    ]),
    prefix_lens=[2, 4],
)

# ── Case 4: prefix covers entire sequence (edge case) ────────────────────────
# answer is empty — mask should be all zeros
show(
    "prefix fills whole sequence (no answer tokens)",
    attention_mask=torch.ones(1, 4, dtype=torch.long),
    prefix_lens=[4],
)
