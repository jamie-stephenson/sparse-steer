---
tags:
  - steering
---
# Sparse Activation Steering Project
The key point is that we are decoupling learning sparsity and strength. There are many ways to do this (see SAE literature), so we need to justify why this one is good.
## Previous Work
| Steering Technique                                                     | Retrieved from                                                                                                                                                      | Calculation                                                                                                                                                                                            | Applied to                                                                                                                                                                                                                       | Contrastive Data                                                                                                                         | Desired Behaviour                                                                                                                                                   |     |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| [Function Vectors](toddFunctionVectors2024)                            | Final token position in prompt from top $k$ attention heads (across layers) based on patched logit difference                                                       | Sum across selected heads of mean projected difference vectors                                                                                                                                         | Final token position of prompt in residual stream:                                                                                                                                                                               | Few-shot ICL prompts. Zero-shot or shuffled answers.                                                                                     | Perform ICL level task performance with no ICL.                                                                                                                     |     |
| [Activation Addition](turnerSteeringLanguageModels2024a) (ActAdd)      | All token positions in residual stream at layer $l$ (so its an array not a vector)                                                                                  | Difference between (padded) positive and negative examples                                                                                                                                             | The layer $l$ from the token position $a$ till $a+$input example seq len                                                                                                                                                         | *Single* pair of phrases: concept vs opposite                                                                                            | Bunch of coarse behaviours e.g. topic (e.g. weddings), sentiment                                                                                                    |     |
| [Contrastive Activation Addition](panicksserySteeringLlama22024) (CAA) | Answer token in residual stream at layer $l$                                                                                                                        | Mean difference across many prompts                                                                                                                                                                    | Every token position after the prompt at layer $l$ of the residual stream                                                                                                                                                        | MCQs that test for behaviour, with answer manually changed                                                                               | Coarse behaviour like sycophancy or hallucination                                                                                                                   |     |
| [Harmfulness vs Refusal Directions](zhaoLLMsEncodeHarmfulness2025)     | Two token positions in residual stream at layer $l$: (i) last token of the user instruction (harmfulness), (ii) last token of the post-instruction region (refusal) | Mean difference                                                                                                                                                                                        | *All* tokens of input instructions at layer $l$ of the residual stream                                                                                                                                                           | Harmful vs harmless instruction sets, subdivided based on whether the model refused or accepted                                          | Elicit refusal by steering along harmfulness direction from final instruction token and compare it with steering along refusal direction from final template token. |     |
| [Single Refusal Direction](arditiRefusalLanguageModels2024)            | Residual stream activations across post-instruction token positions at layer $l$                                                                                    | Mean difference                                                                                                                                                                                        | Two interventions: activation addition (to all positions at layer $l$) and directional ablation (project residual stream at ***all layers*** into orthogonal subspace, preventing model from *ever* representing this direction) | Small contrastive sets of harmful vs harmless instructions (note that refusal not considered and yet they call it the refusal direction) | Both elicit and circumvent refusal                                                                                                                                  |     |
| [DEAL](zhanDEALDisentanglingTransformer2025)                           | Final token position in prompt from attention heads (across layers) selected by VQ-AE.                                                                              | Mean difference                                                                                                                                                                                        | The selected heads                                                                                                                                                                                                               | TruthfulQA and various AI Risk related behaviour datasets                                                                                | Improved behaviour in terms of hallucination, myopic reward, corrigibility, and survival instinct.                                                                  |     |
| [ITI]()                                                                | Final token position in prompt from attention heads (across layers) selected by linear probe.                                                                       | Mean difference                                                                                                                                                                                        | The selected heads                                                                                                                                                                                                               | TruthfulQA pairs, some correct, some not.                                                                                                | More truthful answers.                                                                                                                                              |     |
| [SADI](wangSEMANTICSADAPTIVEACTIVATIONINTERVENTION2025)                | Last-token activations across layers (three possible locations: hidden (think this means residual stream??) / heads / FFN neurons)                                  | Mean diff → top-K binary mask $M$; steer with $A' = A + \delta(M \odot A)$                                                                                                                             | Last-token of input (all unmasked activations according to $M$)                                                                                                                                                                  | MCQ contrastive prompts with forced answers                                                                                              | Improve desired behavior (e.g., truthfulness)                                                                                                                       |     |
| [ACT](wangAdaptiveActivationSteering2025)                              | Final token activations of each attention head output                                                                                                               | Find difference for each contrastive pair, K means cluster them, train a probe to classify positive/negative *within each cluster*, use learned probe weights as K different possible steering vectors | Attention heads but they first decide how much of each steering vector to use based on probe score                                                                                                                               | TruthfulQA                                                                                                                               | Truthfullness                                                                                                                                                       |     |

## Behavioural steering settings
We want a dataset for steering vector extraction and datasets to evaluate on. The problem is there are so many different ways to eval.

**Truthfulness**. This doesn't sit well with me but lots of people explore this.

**Refusal**.

> [!remark] Extraction Dataset vs Evaluation Dataset
> There are varying degrees to which steering can generalise: to held out test subset of extraction dataset? or to other settings? Need to consider both.

## Ideas/Questions
**There are lots of ways to come up with steering vectors**. Compare using contrastive vectors with SAE features. Train SAE on contrastive differences, or something else?

In the addition-only case, having learned gates, we can construct the effective residual stream steering vector from the gates and attention output steering vectors. Would be good to compare these two equivalent contrastive steering vectors obtained directly from the residual stream. Would we expect these to be any better/worse/different?

*questions for next meeting:*
**Concerns about applicability to non-behavioural tasks.**

**Steering MLPs**. Llama uses SwiGLU. What is the right place to extract and apply steering vectors? If you extract from anything after the SwiGLU, the you can apply direct to the residual stream. What about `gate_proj` or `up_proj`? I think `up_proj` makes more sense but don't have good justification.

**Token position**. Currently just steering the same at every token position.

**Inference time gate freezing**. Why freeze to $\sigma(\log \alpha)$?

**Hyperparameter tuning advice**? 

**Does localisation correspond to known circuits?**.

## TODO (ordered by priority)
Contrastive steering directions from: Helpful/harmful 
Evaluate on 
Social reasoning: SiQA 
Ethics: EQ bench 
Safety/jailbreaking: AdvBench FalseReject 
Hallucinations/facts: TruthfulQA HaluEval
**Which of these has room for improvement?**

learned scale param for hardconcrete

additive only: but search for lit on other non additive contrastive steering

possible extentions, domain generalisation. 

What real world task actually has this few examples?

Explore why multiplicative editing is needed in JoLA. How would it work in contrastive setting?

Read review of steering methods in zotero.

Read up on REINFORCE vs reparameterisation.

Review literature where steering vectors are extracted/applied at the head level.

Review literature where steering is applied at different token positions (e.g. MAT-Steer)



see slack for more baseline ideas, e.g. learning scale parameter but no gates, or learning gates but no scaling parameter.

compare to naive steering. baseline, check its actually sparse

get a feeling for how many examples are required

task vectors paper

generative eval setting