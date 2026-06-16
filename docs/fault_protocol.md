# Crafter Fault Protocol

This protocol keeps bug injection labels separate from task performance labels.
The goal is to evaluate agents that keep playing the game while actively
seeking state-transition faults.

## Step Labels

- `fault_exists_episode`: A fault injector selected a fault for this episode.
  The agent may still never reach the vulnerable context.
- `fault_trigger_context`: The current transition reached a context where the
  selected fault could manifest.
- `lowlevel_trigger_context`: Same as above for action/reward/termination
  faults managed by the Dreamer wrapper.
- `semantic_trigger_context`: Same as above for high-level Crafter semantic
  faults.
- `fault_manifested`: The environment transition was actually corrupted.
  This is equivalent to the legacy `fault_applied` field.
- `fault_manifest_prob`: The configured probability that a reached trigger
  context manifests as an actual corrupted transition.

This separation supports three analyses that should not be collapsed:

1. Did the policy explore into vulnerable contexts?
2. Did a configured bug actually manifest?
3. Did the clean-prior fault score rise around either event?

## Frequency Tiers

Use `CRAFTER_FAULT_FREQ_TIER` to set default bug rarity knobs. Explicit
environment variables still take precedence.

| Tier | Purpose | Episode Probability | Manifest Probability |
| --- | --- | ---: | ---: |
| `diagnostic` | Fast smoke tests and visual debugging | high | high |
| `benchmark` | Main train/eval protocol | medium | medium |
| `realistic` | Sparse realism evaluation | low | low |
| `sparse` | Stress test for rare bugs | very low | very low |
| `custom` | Use only explicitly supplied env vars | unchanged | unchanged |

Current defaults are encoded in `embodied/envs/crafter.py` under
`FAULT_FREQUENCY_TIER_DEFAULTS`.

Implementation note: semantic manifestation gating is implemented in the local
installed Crafter package at
`dreamer_cuda/lib/python3.11/site-packages/crafter/env.py`. If the virtualenv is
rebuilt, reapply that patch or vendor the Crafter environment into the repo.

## Recommended Splits

- Train/eval seen: moderate-frequency semantic and low-level faults that cover
  common state consistency, action execution, and rule-precondition failures.
- Holdout: same families, unseen subtypes.
- Semantic holdout: higher-level gameplay-rule faults where the trigger context
  is meaningful even if the manifestation is stochastic.
- Realistic sparse: use only for final evaluation, not for reward tuning.

## Reporting

Report task score and bug discovery together:

- task reward / achievement coverage
- `fault_trigger_context_count`
- `fault_manifest_count`
- time-to-first `fault_trigger_context`
- time-to-first `fault_manifested`
- fault-score AUROC/AUPRC against both `fault_trigger_context` and
  `fault_manifested`
- clean false alarm rate under the same threshold

For ICRL-style framing, the strongest claim is not that fault reward improves
game score, but that clean-prior surprise can guide an agent toward vulnerable
transition contexts while preserving task competence.
