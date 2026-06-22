import numpy as np
from types import SimpleNamespace

from dreamerv3 import fault_score


def cfg(mode, **kwargs):
  data = dict(
      use_latent_kl=True,
      use_reward_error=False,
      w_kl=1.0,
      w_reward=0.0,
      norm_mode='none',
      clip=1.0,
      beta=0.1,
      log_only=False,
      reward_gate='nonzero_action',
      reward_mode=mode,
      reward_threshold=1.0,
      reward_delta_threshold=0.5)
  data.update(kwargs)
  return SimpleNamespace(**data)


def tran(score=1.2, prev=0.4, action=1, reward=0.0):
  return {
      'is_first': np.array(False),
      'reward': np.float32(reward),
      'log/task_reward_raw': np.float32(reward),
      'log/ref_latent_kl_surprise': np.float32(score),
      'log/ref_reward_pred': np.float32(0.0),
      'log/fault_score_prev': np.float32(prev),
      'action': np.int32(action),
  }


def check_close(value, expected, eps=1e-6):
  assert abs(float(value) - float(expected)) < eps, (value, expected)


def test_threshold_hit():
  res = fault_score.compute_transition_fault(
      tran(score=1.2), cfg('threshold'), stats={})
  check_close(res['fault_reward_bonus'], 0.1)
  check_close(res['augmented_reward'], 0.1)
  assert int(res['fault_reward_mode_id']) == 2


def test_threshold_miss():
  res = fault_score.compute_transition_fault(
      tran(score=0.9), cfg('threshold'), stats={})
  check_close(res['fault_reward_bonus'], 0.0)
  check_close(res['augmented_reward'], 0.0)


def test_delta_hit():
  res = fault_score.compute_transition_fault(
      tran(score=1.2, prev=0.4), cfg('delta_threshold'), stats={})
  check_close(res['fault_score_delta'], 0.8)
  check_close(res['fault_reward_bonus'], 0.1)
  assert int(res['fault_reward_mode_id']) == 4


def test_nonzero_action_gate_blocks_noop():
  res = fault_score.compute_transition_fault(
      tran(score=1.2, action=0), cfg('threshold'), stats={})
  check_close(res['fault_reward_gate_active'], 0.0)
  check_close(res['fault_reward_bonus'], 0.0)
  check_close(res['augmented_reward'], 0.0)


def test_log_only_keeps_reward_unchanged():
  res = fault_score.compute_transition_fault(
      tran(score=1.2), cfg('threshold', log_only=True), stats={})
  check_close(res['fault_reward_bonus'], 0.1)
  check_close(res['augmented_reward'], 0.0)


def test_transition_logs_include_new_fields():
  data = tran(score=1.2, prev=0.4)
  res = fault_score.compute_transition_fault(
      data, cfg('excess_delta_threshold', clip=2.0), stats={})
  fault_score.add_transition_fault_logs(data, res)
  for key in (
      'log/fault_score_delta',
      'log/fault_reward_score',
      'log/fault_reward_threshold',
      'log/fault_reward_mode_id'):
    assert key in data, key


def run_all():
  test_threshold_hit()
  test_threshold_miss()
  test_delta_hit()
  test_nonzero_action_gate_blocks_noop()
  test_log_only_keeps_reward_unchanged()
  test_transition_logs_include_new_fields()
  print('ALL 6 FAULT REWARD MODE TESTS PASSED')


if __name__ == '__main__':
  run_all()
