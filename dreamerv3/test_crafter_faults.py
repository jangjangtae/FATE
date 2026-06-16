import os
from collections import deque

import numpy as np

# import 경로는 네 프로젝트 구조에 맞게 바꿔야 할 수 있음
# 예: from embodied.envs.crafter import Crafter
from embodied.envs.crafter import Crafter


def make_env():
    # deterministic train profile
    os.environ["CRAFTER_FAULT"] = "0"
    os.environ["CRAFTER_FAULT_SAMPLER"] = "0"
    os.environ["CRAFTER_FAULT_PROFILE"] = "train"
    os.environ["CRAFTER_FAULT_COOLDOWN"] = "0"

    env = Crafter(
        task="reward",
        size=(64, 64),
        logs=False,
        seed=0,
    )

    # reset 1회
    env.step({"action": np.int32(0), "reset": True})
    return env


def prime_env(env, family, subtype):
    env._fault_spec = {
        "family": family,
        "type": subtype,
        "severity": 1.0,
        "trigger": env._trigger_name_from_subtype(subtype),
    }
    env._fault_episode = 1
    env._fault_cooldown = 0
    env._after_positive_window = 0
    env._last_reward = 0.0
    env._last_requested_action = 0
    env._prev_executed_action = 0
    env._length = 0
    env._pending_reward = 0.0
    env._sticky_action = None
    env._sticky_remaining = 0
    env._requested_hist = deque([0, 0], maxlen=2)
    env._executed_hist = deque([0, 0], maxlen=2)
    env._reward_hist = deque([0.0, 0.0], maxlen=2)


def test_remap_after_success_switch():
    env = make_env()
    prime_env(env, "action_exec", "remap_after_success_switch")
    env._after_positive_window = 3
    env._last_requested_action = 1
    out_action, applied, fault_type = env._apply_action_fault(2)

    assert applied == 1
    assert fault_type == "remap_after_success_switch"
    assert out_action != 2
    print("PASS: remap_after_success_switch")


def test_delay_after_success():
    env = make_env()
    prime_env(env, "action_exec", "delay_after_success")
    env._after_positive_window = 3
    env._last_requested_action = 1
    out_action, applied, fault_type = env._apply_action_fault(2)

    assert applied == 1
    assert fault_type == "delay_after_success"
    assert out_action == 1
    print("PASS: delay_after_success")


def test_sticky_after_repeat_switch():
    env = make_env()
    prime_env(env, "action_exec", "sticky_after_repeat_switch")
    env._requested_hist = deque([1, 1], maxlen=2)

    # 1차 호출: arm만 됨
    out_action, applied, fault_type = env._apply_action_fault(2)
    assert applied == 0
    assert env._sticky_action == 2
    assert env._sticky_remaining == 1

    # 2차 호출: sticky 실제 적용
    out_action2, applied2, fault_type2 = env._apply_action_fault(3)
    assert applied2 == 1
    assert fault_type2 == "sticky_after_repeat_switch"
    assert out_action2 == 2
    print("PASS: sticky_after_repeat_switch")


def test_ignore_nonzero_after_reward():
    env = make_env()
    prime_env(env, "context_exec", "ignore_nonzero_after_reward")
    env._after_positive_window = 3

    out_action, applied, fault_type = env._apply_action_fault(2)
    assert applied == 1
    assert fault_type == "ignore_nonzero_after_reward"
    assert out_action == 0
    print("PASS: ignore_nonzero_after_reward")


def test_ignore_switch_late_episode():
    env = make_env()
    prime_env(env, "context_exec", "ignore_switch_late_episode")
    env._length = 60
    env._last_requested_action = 1

    out_action, applied, fault_type = env._apply_action_fault(2)
    assert applied == 1
    assert fault_type == "ignore_switch_late_episode"
    assert out_action == 0
    print("PASS: ignore_switch_late_episode")


def test_reward_delay_on_positive():
    env = make_env()
    prime_env(env, "reward_timing", "reward_delay_on_positive")

    reward, applied, fault_type = env._apply_reward_fault(1.0, requested_action=1)
    assert applied == 1
    assert fault_type == "reward_delay_on_positive"
    assert reward == 0.0
    assert env._pending_reward > 0.0
    print("PASS: reward_delay_on_positive")


def test_reward_scale_half_on_positive_switch():
    env = make_env()
    prime_env(env, "reward_timing", "reward_scale_half_on_positive_switch")
    env._last_requested_action = 1

    reward, applied, fault_type = env._apply_reward_fault(2.0, requested_action=2)
    assert applied == 1
    assert fault_type == "reward_scale_half_on_positive_switch"
    assert reward == 1.0
    print("PASS: reward_scale_half_on_positive_switch")


def test_reward_zero_on_positive():
    env = make_env()
    prime_env(env, "reward_timing", "reward_zero_on_positive")

    reward, applied, fault_type = env._apply_reward_fault(2.0, requested_action=1)
    assert applied == 1
    assert fault_type == "reward_zero_on_positive"
    assert reward == 0.0
    print("PASS: reward_zero_on_positive")


def test_early_done_after_success_switch():
    env = make_env()
    prime_env(env, "termination_logic", "early_done_after_success_switch")
    env._after_positive_window = 3
    env._last_requested_action = 1

    done, info, applied, fault_type = env._apply_termination_fault(
        requested_action=2,
        done=False,
        info={"discount": 1.0},
    )
    assert applied == 1
    assert fault_type == "early_done_after_success_switch"
    assert done is True
    assert info["discount"] == 0.0
    print("PASS: early_done_after_success_switch")


def run_all():
    test_remap_after_success_switch()
    test_delay_after_success()
    test_sticky_after_repeat_switch()
    test_ignore_nonzero_after_reward()
    test_ignore_switch_late_episode()
    test_reward_delay_on_positive()
    test_reward_scale_half_on_positive_switch()
    test_reward_zero_on_positive()
    test_early_done_after_success_switch()
    print("\nALL 9 FAULT TESTS PASSED")


if __name__ == "__main__":
    run_all()