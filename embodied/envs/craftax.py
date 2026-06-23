import functools
import os

import elements
import embodied
import jax
import numpy as np


_ACHIEVEMENTS = (
    'collect_wood',
    'place_table',
    'eat_cow',
    'collect_sapling',
    'collect_drink',
    'make_wood_pickaxe',
    'make_wood_sword',
    'place_plant',
    'defeat_zombie',
    'collect_stone',
    'place_stone',
    'eat_plant',
    'defeat_skeleton',
    'make_stone_pickaxe',
    'make_stone_sword',
    'wake_up',
    'place_furnace',
    'collect_coal',
    'collect_iron',
    'collect_diamond',
    'make_iron_pickaxe',
    'make_iron_sword',
)


class Craftax(embodied.Env):

  def __init__(
      self, task='classic_pixels', seed=0, length=10000,
      logs=True, variant=None, platform='cpu'):
    task = variant or task
    task = task.replace('-', '_').lower()
    if task in ('classic', 'classic_pixels', 'pixels'):
      self._obs_kind = 'pixels'
    elif task in ('classic_symbolic', 'symbolic'):
      self._obs_kind = 'symbolic'
    else:
      raise ValueError(f'Unknown Craftax task: {task!r}')

    self._task = task
    self._logs = bool(logs)
    self._length = int(length) if length else 0
    self._seed = int(seed)
    self._platform = platform
    self._achievements = list(_ACHIEVEMENTS)
    self._env = None
    self._key = None
    self._state = None
    self._done = True
    self._params = None
    self._last_image = np.zeros((64, 64, 3), np.uint8)

  @functools.cached_property
  def obs_space(self):
    spaces = {}
    if self._obs_kind == 'pixels':
      spaces['image'] = elements.Space(np.uint8, (64, 64, 3))
    else:
      spaces['vector'] = elements.Space(np.float32, (1345,))
    spaces.update(
        reward=elements.Space(np.float32),
        is_first=elements.Space(bool),
        is_last=elements.Space(bool),
        is_terminal=elements.Space(bool),
    )
    if self._logs:
      spaces.update({
          'log/score': elements.Space(np.float32),
          'log/timestep': elements.Space(np.int32),
          'log/health': elements.Space(np.float32),
          'log/food': elements.Space(np.float32),
          'log/drink': elements.Space(np.float32),
          'log/energy': elements.Space(np.float32),
          'log/achievement_count': elements.Space(np.int32),
      })
      for name in self._achievements:
        spaces[f'log/achievement_{name}'] = elements.Space(np.int32)
    return spaces

  @functools.cached_property
  def act_space(self):
    return {
        'reset': elements.Space(bool),
        'action': elements.Space(np.int32, (), 0, 17),
    }

  def step(self, action):
    if bool(action['reset']) or self._done:
      obs = self._reset()
      return self._obs(obs, 0.0, is_first=True)

    act = np.asarray(action['action'], np.int32).reshape(())
    with jax.transfer_guard('allow'):
      self._key, key = jax.random.split(self._key)
      obs, self._state, reward, done, info = self._env.step(
          key, self._state, act, self._params)
      self._done = bool(np.asarray(done))
    return self._obs(
        obs, reward, info,
        is_last=self._done, is_terminal=self._done)

  def render(self):
    return self._last_image

  def _ensure_env(self):
    if self._env is not None:
      return
    if self._platform:
      os.environ.setdefault('JAX_PLATFORMS', self._platform)
      try:
        jax.config.update('jax_platforms', self._platform)
      except Exception as e:
        raise RuntimeError(
            'Craftax JAX platform must be configured before the backend is '
            'initialized. Use `--run.debug False` so Craftax runs in a fresh '
            'environment subprocess, or set `env.craftax.platform` empty to '
            'share the learner backend.') from e
    try:
      with jax.transfer_guard('allow'):
        from craftax.craftax_classic.envs.craftax_pixels_env import (
            CraftaxClassicPixelsEnv)
        from craftax.craftax_classic.envs.craftax_symbolic_env import (
            CraftaxClassicSymbolicEnv)
    except ImportError as e:
      raise ImportError(
          'Craftax is not installed. Install it in the Dreamer environment '
          'with `pip install craftax`, or set PYTHONPATH to a directory that '
          'contains the package for smoke tests.') from e

    with jax.transfer_guard('allow'):
      if self._obs_kind == 'pixels':
        self._env = CraftaxClassicPixelsEnv()
      else:
        self._env = CraftaxClassicSymbolicEnv()
      self._params = self._env.default_params
      if self._length:
        self._params = self._params.replace(max_timesteps=self._length)
      self._key = jax.random.PRNGKey(self._seed)

  def _reset(self):
    self._ensure_env()
    with jax.transfer_guard('allow'):
      self._key, key = jax.random.split(self._key)
      obs, self._state = self._env.reset(key, self._params)
    self._done = False
    return obs

  def _obs(
      self, obs, reward, info=None,
      is_first=False, is_last=False, is_terminal=False):
    data = {}
    if self._obs_kind == 'pixels':
      with jax.transfer_guard('allow'):
        image = np.asarray(obs)
      if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
      if image.shape[:2] == (63, 63):
        image = np.pad(image, ((0, 1), (0, 1), (0, 0)), mode='edge')
      if image.shape != (64, 64, 3):
        raise RuntimeError(f'Unexpected Craftax image shape: {image.shape}')
      self._last_image = image
      data['image'] = image
    else:
      with jax.transfer_guard('allow'):
        data['vector'] = np.asarray(obs, np.float32)

    data.update(
        reward=np.float32(_to_numpy(reward)),
        is_first=bool(is_first),
        is_last=bool(is_last),
        is_terminal=bool(is_terminal),
    )
    if self._logs:
      data.update(self._log_obs(info or {}))
    return data

  def _log_obs(self, info):
    state = self._state
    achievements = _to_numpy(state.achievements).astype(np.int32)
    out = {
        'log/score': np.float32(_to_numpy(info.get('score', 0.0))),
        'log/timestep': np.int32(_to_numpy(state.timestep)),
        'log/health': np.float32(_to_numpy(state.player_health) / 9.0),
        'log/food': np.float32(_to_numpy(state.player_food) / 9.0),
        'log/drink': np.float32(_to_numpy(state.player_drink) / 9.0),
        'log/energy': np.float32(_to_numpy(state.player_energy) / 9.0),
        'log/achievement_count': np.int32(achievements.sum()),
    }
    for index, name in enumerate(self._achievements):
      out[f'log/achievement_{name}'] = np.int32(achievements[index])
    return out


def _to_numpy(value):
  with jax.transfer_guard('allow'):
    return np.asarray(value)
