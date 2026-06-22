import json
from pathlib import Path

import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import numpy as np


BUG_TYPE_NAMES = {
    0: 'none',
    1: 'drop_to_fallback',
    2: 'remap_after_success_switch',
    3: 'delay_after_success',
    4: 'sticky_after_repeat_switch',
    5: 'ignore_nonzero_after_reward',
    6: 'ignore_switch_late_episode',
    7: 'reward_delay_on_positive',
    8: 'reward_scale_half_on_positive_switch',
    9: 'reward_zero_on_positive',
    10: 'early_done_after_success_switch',
    11: 'remap_after_repeat_switch',
    12: 'delay_after_late_episode_switch',
    13: 'ignore_nonzero_after_two_rewards',
    14: 'reward_zero_after_repeat_switch',
    15: 'reward_delay_after_two_rewards',
    16: 'early_done_after_repeat_switch',
    17: 'revisit_action_ignore',
    18: 'revisit_action_delay',
    19: 'delayed_switch_failure',
    20: 'tool_collect_desync_on_upgrade',
    21: 'collect_result_delayed_after_tool_upgrade',
    22: 'upgrade_branch_inconsistent_collect_behavior',
    23: 'craft_result_missing_on_retry',
    24: 'craft_output_delayed_on_retry',
    25: 'recipe_precondition_mischeck_on_retry',
    26: 'recipe_retry_requires_revisit',
    27: 'station_place_ghost_on_relocate',
    28: 'station_second_use_inconsistent_after_placement',
    29: 'station_usable_flag_broken_after_relocate',
    30: 'station_state_partial_reset_after_relocate',
    31: 'achievement_unlock_missing_after_valid_progress',
    32: 'achievement_unlock_missing_after_reconfirm',
    33: 'progress_confirmation_requires_revisit',
    34: 'delayed_inventory_desync_after_station_use',
}

REWARD_MODE_IDS = {
    'dense': 1,
    'threshold': 2,
    'binary_threshold': 2,
    'excess_threshold': 3,
    'excess': 3,
    'delta_threshold': 4,
    'binary_delta_threshold': 4,
    'excess_delta_threshold': 5,
    'delta_excess': 5,
}


def _cfg_get(config, key, default=None):
  if config is None:
    return default
  if hasattr(config, 'get'):
    return config.get(key, default)
  return getattr(config, key, default)


def reduce_metric(x):
  x = nn.cast(x)
  if hasattr(x, 'ndim') and x.ndim > 1:
    return x.mean(tuple(range(1, x.ndim)))
  return x


def agent_fault_metrics(dyn, rew, feat2tensor, feat, obs, config=None):
  """Fault signals from a clean reference world model at policy time."""
  use_latent_kl = _cfg_get(config, 'use_latent_kl', True)
  use_reward_error = _cfg_get(config, 'use_reward_error', True)
  w_kl = float(_cfg_get(config, 'w_kl', 1.0))
  w_reward = float(_cfg_get(config, 'w_reward', 1.0))

  reward_true = nn.cast(obs['reward'])
  zeros = jnp.zeros_like(reduce_metric(reward_true))

  latent_kl = zeros
  if use_latent_kl and ('prior_logit' in feat) and ('logit' in feat):
    posterior = dyn._dist(feat['logit'])
    prior = dyn._dist(jax.lax.stop_gradient(feat['prior_logit']))
    latent_kl = reduce_metric(posterior.kl(prior))

  reward_pred = zeros
  reward_error = zeros
  if use_reward_error and ('prior_logit' in feat):
    prior_feat = {
        'deter': feat['deter'],
        'stoch': dyn._dist(feat['prior_logit']).pred(),
    }
    reward_pred = reduce_metric(rew(feat2tensor(prior_feat), 1).pred())
    reward_error = reduce_metric(jnp.abs(reward_pred - reward_true))

  fault_score_raw = w_kl * latent_kl + w_reward * reward_error

  return {
      'fault_score_raw': fault_score_raw,
      'fault_score': fault_score_raw,
      'latent_kl_surprise': latent_kl,
      'reward_prediction_error': reward_error,
      'reward_pred': reward_pred,
  }


def load_norm_stats(path):
  if not path:
    return {}
  path = Path(str(path)).expanduser()
  if not path.exists():
    raise FileNotFoundError(path)
  with path.open('r', encoding='utf-8') as f:
    return json.load(f)


def normalize_score(raw, stats=None, mode='p95', eps=1e-8):
  raw = float(raw)
  stats = stats or {}
  mode = (mode or 'none').lower()
  if mode == 'none':
    return raw
  if mode == 'zscore':
    mean = float(stats.get('fault_score_mean', 0.0))
    std = float(stats.get('fault_score_std', 1.0))
    return (raw - mean) / max(std, eps)
  if mode in ('p90', 'p95', 'p99'):
    denom = float(stats.get(f'fault_score_{mode}', 0.0))
    if denom <= eps:
      denom = float(stats.get('fault_score_p95', 1.0))
    return raw / max(denom, eps)
  raise ValueError(f'Unknown fault score norm mode: {mode}')


def reward_gate_active(tran, config=None):
  """Whether a fault reward bonus is allowed for this transition."""
  mode = str(_cfg_get(config, 'reward_gate', 'none') or 'none').lower()
  if mode in ('none', 'all', 'always', ''):
    return True

  semantic_keys = (
      'log/semantic_trigger_context',
      'log/semantic_ctx_upgrade_collect',
      'log/semantic_ctx_retry_craft',
      'log/semantic_ctx_relocate_station',
      'log/semantic_ctx_valid_progress',
      'log/semantic_ctx_station_reuse',
      'log/semantic_ctx_delayed_after_use',
  )
  semantic_active = any(_scalar(tran.get(key, 0.0)) > 0.5
      for key in semantic_keys)
  reward_active = abs(_scalar(
      tran.get('log/task_reward_raw', tran.get('reward', 0.0)))) > 1e-6
  action_active = int(_scalar(tran.get('action', 0.0))) != 0

  if mode in ('semantic', 'semantic_context', 'semantic_event'):
    return semantic_active
  if mode in ('reward', 'task_reward', 'nonzero_reward'):
    return reward_active
  if mode in ('nonzero_action', 'action'):
    return action_active
  if mode in ('semantic_or_reward', 'event'):
    return semantic_active or reward_active
  if mode in ('semantic_or_action', 'interaction'):
    return semantic_active or action_active
  if mode in ('semantic_reward_action', 'semantic_or_reward_or_action'):
    return semantic_active or reward_active or action_active

  raise ValueError(f'Unknown fault reward gate: {mode}')


def compute_fault_bonus(
    normalized, clipped, beta, config=None, score_delta=0.0, eps=1e-8):
  """Convert a normalized clean-prior score into a policy reward bonus."""
  mode = str(_cfg_get(config, 'reward_mode', 'dense') or 'dense').lower()
  threshold = float(_cfg_get(config, 'reward_threshold', 1.0))
  delta_threshold = float(_cfg_get(config, 'reward_delta_threshold', threshold))
  clip = float(_cfg_get(config, 'clip', 1.0))

  if mode == 'dense':
    shaped = clipped
  elif mode in ('threshold', 'binary_threshold'):
    shaped = 1.0 if float(normalized) >= threshold else 0.0
  elif mode in ('excess_threshold', 'excess'):
    denom = max(float(clip) - threshold, eps)
    shaped = np.clip((float(normalized) - threshold) / denom, 0.0, 1.0)
  elif mode in ('delta_threshold', 'binary_delta_threshold'):
    shaped = 1.0 if float(score_delta) >= delta_threshold else 0.0
    threshold = delta_threshold
  elif mode in ('excess_delta_threshold', 'delta_excess'):
    denom = max(float(clip) - delta_threshold, eps)
    shaped = np.clip((float(score_delta) - delta_threshold) / denom, 0.0, 1.0)
    threshold = delta_threshold
  else:
    raise ValueError(f'Unknown fault reward mode: {mode}')

  return float(beta * shaped), float(shaped), threshold, REWARD_MODE_IDS.get(mode, 0)


def compute_transition_fault(tran, config=None, stats=None, force_log_only=False):
  """Compute normalized fault score and optional reward relabeling for one step."""
  is_first = bool(np.asarray(tran.get('is_first', False)).reshape(()))
  raw_reward = _scalar(tran.get('log/task_reward_raw', tran.get('reward', 0.0)))
  augmented_reward = raw_reward

  if is_first:
    latent_kl = 0.0
    reward_pred = 0.0
    reward_error = 0.0
  else:
    latent_kl = _scalar(tran.get('log/ref_latent_kl_surprise',
        tran.get('log/ref_bug_kl', 0.0)))
    reward_pred = _scalar(tran.get('log/ref_reward_pred', 0.0))
    reward_error = abs(raw_reward - reward_pred)

  use_latent_kl = bool(_cfg_get(config, 'use_latent_kl', True))
  use_reward_error = bool(_cfg_get(config, 'use_reward_error', True))
  w_kl = float(_cfg_get(config, 'w_kl', 1.0))
  w_reward = float(_cfg_get(config, 'w_reward', 1.0))

  raw = 0.0
  if use_latent_kl:
    raw += w_kl * latent_kl
  if use_reward_error:
    raw += w_reward * reward_error

  norm_mode = _cfg_get(config, 'norm_mode', 'p95')
  normalized = normalize_score(raw, stats, norm_mode)
  score_prev = _scalar(tran.get('log/fault_score_prev', 0.0))
  score_delta = max(0.0, float(normalized) - float(score_prev))
  clip = float(_cfg_get(config, 'clip', 1.0))
  clipped = float(np.clip(normalized, 0.0, clip))
  beta = float(_cfg_get(config, 'beta', 0.1))
  log_only = bool(_cfg_get(config, 'log_only', True)) or force_log_only
  gate_active = reward_gate_active(tran, config)
  fault_bonus, reward_score, reward_threshold, reward_mode_id = (
      compute_fault_bonus(
          normalized, clipped, beta, config, score_delta=score_delta))
  if not gate_active:
    fault_bonus = 0.0
    reward_score = 0.0
  if not log_only:
    augmented_reward = raw_reward + fault_bonus

  return {
      'raw_reward': float(raw_reward),
      'augmented_reward': float(augmented_reward),
      'latent_kl_surprise': float(latent_kl),
      'reward_prediction_error': float(reward_error),
      'reward_pred': float(reward_pred),
      'fault_score_raw': float(raw),
      'fault_score': float(normalized),
      'fault_score_prev': float(score_prev),
      'fault_score_delta': float(score_delta),
      'clipped_fault_score': float(clipped),
      'fault_beta': float(beta),
      'fault_reward_gate_active': float(gate_active),
      'fault_reward_bonus': float(fault_bonus),
      'fault_reward_score': float(reward_score),
      'fault_reward_threshold': float(reward_threshold),
      'fault_reward_mode_id': float(reward_mode_id),
  }


def add_transition_fault_logs(tran, result):
  tran['log/raw_reward'] = np.float32(result['raw_reward'])
  tran['log/augmented_reward'] = np.float32(result['augmented_reward'])
  tran['log/fault_score_raw'] = np.float32(result['fault_score_raw'])
  tran['log/fault_score'] = np.float32(result['fault_score'])
  tran['log/fault_score_prev'] = np.float32(result['fault_score_prev'])
  tran['log/fault_score_delta'] = np.float32(result['fault_score_delta'])
  tran['log/clipped_fault_score'] = np.float32(result['clipped_fault_score'])
  tran['log/latent_kl_surprise'] = np.float32(result['latent_kl_surprise'])
  tran['log/reward_prediction_error'] = np.float32(
      result['reward_prediction_error'])
  tran['log/reward_pred'] = np.float32(result['reward_pred'])
  tran['log/fault_beta'] = np.float32(result['fault_beta'])
  tran['log/fault_reward_gate_active'] = np.float32(
      result['fault_reward_gate_active'])
  tran['log/fault_reward_bonus'] = np.float32(result['fault_reward_bonus'])
  tran['log/fault_reward_score'] = np.float32(result['fault_reward_score'])
  tran['log/fault_reward_threshold'] = np.float32(
      result['fault_reward_threshold'])
  tran['log/fault_reward_mode_id'] = np.float32(
      result['fault_reward_mode_id'])


def add_reference_outputs(outs, ref_outs):
  """Copy reference-agent fault metrics to log keys and keep legacy aliases."""
  mapping = {
      'fault/fault_score_raw': 'log/ref_fault_score_raw',
      'fault/fault_score': 'log/ref_fault_score',
      'fault/latent_kl_surprise': 'log/ref_latent_kl_surprise',
      'fault/reward_prediction_error': 'log/ref_reward_prediction_error',
      'fault/reward_pred': 'log/ref_reward_pred',
      'bug/score': 'log/ref_bug_score',
      'bug/kl': 'log/ref_bug_kl',
      'bug/reward_err': 'log/ref_bug_reward_err',
      'bug/continue_err': 'log/ref_bug_continue_err',
  }
  for key, value in ref_outs.items():
    if key in mapping:
      outs[mapping[key]] = value

  if 'log/ref_fault_score' in outs:
    outs.setdefault('log/ref_bug_score', outs['log/ref_fault_score'])
  if 'log/ref_latent_kl_surprise' in outs:
    outs.setdefault('log/ref_bug_kl', outs['log/ref_latent_kl_surprise'])
  if 'log/ref_reward_prediction_error' in outs:
    outs.setdefault(
        'log/ref_bug_reward_err', outs['log/ref_reward_prediction_error'])


def trace_record(tran, global_step, worker, episode_id, episode_step):
  bug_triggered = int(
      _scalar(tran.get('log/fault_applied', 0.0)) > 0.5 or
      _scalar(tran.get('log/semantic_fault_applied', 0.0)) > 0.5)
  bug_id = int(_scalar(tran.get('log/fault_type_id', 0.0)))
  record = {
      'global_step': int(global_step),
      'worker': int(worker),
      'episode_id': int(episode_id),
      'episode_step': int(episode_step),
      'action': _to_python(tran.get('action', 0)),
      'raw_reward': _scalar(tran.get('log/raw_reward', tran.get('reward', 0.0))),
      'augmented_reward': _scalar(tran.get(
          'log/augmented_reward', tran.get('reward', 0.0))),
      'fault_score': _scalar(tran.get('log/fault_score', 0.0)),
      'fault_score_raw': _scalar(tran.get('log/fault_score_raw', 0.0)),
      'fault_score_prev': _scalar(tran.get('log/fault_score_prev', 0.0)),
      'fault_score_delta': _scalar(tran.get('log/fault_score_delta', 0.0)),
      'clipped_fault_score': _scalar(tran.get('log/clipped_fault_score', 0.0)),
      'fault_reward_gate_active': _scalar(
          tran.get('log/fault_reward_gate_active', 0.0)),
      'fault_reward_bonus': _scalar(
          tran.get('log/fault_reward_bonus', 0.0)),
      'fault_reward_score': _scalar(
          tran.get('log/fault_reward_score', 0.0)),
      'fault_reward_threshold': _scalar(
          tran.get('log/fault_reward_threshold', 0.0)),
      'fault_reward_mode_id': int(_scalar(
          tran.get('log/fault_reward_mode_id', 0.0))),
      'latent_kl_surprise': _scalar(tran.get('log/latent_kl_surprise', 0.0)),
      'reward_prediction_error': _scalar(
          tran.get('log/reward_prediction_error', 0.0)),
      'bug_triggered': bug_triggered,
      'bug_id': bug_id,
      'bug_type': BUG_TYPE_NAMES.get(bug_id, 'unknown'),
      'unique_bug_count_cumulative': int(_scalar(
          tran.get('log/fault_count_cumulative', 0.0))),
      'unique_tile_coverage_cumulative': int(_scalar(
          tran.get('log/unique_tiles_visited', 0.0))),
      'is_first': bool(np.asarray(tran.get('is_first', False)).reshape(())),
      'is_last': bool(np.asarray(tran.get('is_last', False)).reshape(())),
      'is_terminal': bool(np.asarray(
          tran.get('is_terminal', False)).reshape(())),
  }
  for key in (
      'log/fault_applied', 'log/fault_episode', 'log/semantic_fault_applied',
      'log/semantic_fault_episode', 'log/tile_coverage_ratio'):
    if key in tran:
      record[key] = _to_python(tran[key])
  return record


def summarize(values, prefix):
  values = np.asarray(values, np.float64)
  values = values[np.isfinite(values)]
  if values.size == 0:
    return {
        f'{prefix}_mean': 0.0,
        f'{prefix}_std': 0.0,
        f'{prefix}_p90': 0.0,
        f'{prefix}_p95': 0.0,
        f'{prefix}_p99': 0.0,
        f'{prefix}_count': 0,
    }
  return {
      f'{prefix}_mean': float(np.mean(values)),
      f'{prefix}_std': float(np.std(values)),
      f'{prefix}_p90': float(np.quantile(values, 0.90)),
      f'{prefix}_p95': float(np.quantile(values, 0.95)),
      f'{prefix}_p99': float(np.quantile(values, 0.99)),
      f'{prefix}_count': int(values.size),
  }


def write_json(path, data):
  path = Path(str(path)).expanduser()
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write('\n')


def _scalar(x, default=0.0):
  if x is None:
    return default
  arr = np.asarray(x)
  if arr.size == 0:
    return default
  return float(arr.reshape(-1)[0])


def _to_python(x):
  if isinstance(x, np.ndarray):
    if x.ndim == 0:
      return x.item()
    return x.tolist()
  if isinstance(x, (np.bool_, np.integer, np.floating)):
    return x.item()
  return x
