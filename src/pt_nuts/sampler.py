"""Parallel-tempered NUTS sampling on top of NumPyro.

Runs a ladder of NUTS chains at different inverse-temperatures (betas) for a
NumPyro model, with periodic swap proposals between adjacent temperatures
(replica exchange / parallel tempering), and estimates the log model
evidence via stepping-stone thermodynamic integration.

Sampling itself is built directly on NumPyro's low-level `hmc()` kernel
(`numpyro.infer.hmc`), the same engine `numpyro.infer.MCMC`/`NUTS` use, so a
single-temperature `pt_nuts` run reproduces the warmup/adaptation behavior
of a plain `numpyro.infer.MCMC(NUTS(model))` run on the same model.

See the package README for parameter semantics, the three parallel
execution modes ("sequential", "vmap", "shard"), and known limitations.
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from typing import Optional, Any, Literal

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpyro
from numpyro.infer.hmc import hmc
from numpyro.infer.util import initialize_model, constrain_fn
from numpyro.distributions.transforms import biject_to

from .progress_bar import progress_bar

try:
  from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
  tqdm = None

Array = jax.Array
PyTree = Any
ParallelMode = Literal["sequential", "vmap", "shard"]


@dataclass
class TemperedNUTSResult:
  samples: PyTree
  loglik: Array
  temperatures: Array
  mean_loglik: Array
  log_evidence: Array
  swap_acceptance: Array
  nuts_acceptance: Array


def geometric_temperature_ladder(n_temperatures: int, beta_min: float = 1e-3) -> Array:
  if n_temperatures < 2:
    raise ValueError("Need at least two temperatures.")
  if not (0.0 < beta_min < 1.0):
    raise ValueError("beta_min must be in (0, 1).")
  positive = jnp.geomspace(beta_min, 1.0, n_temperatures - 1)
  return jnp.concatenate([jnp.zeros(1, dtype=positive.dtype), positive])


def stepping_stone_integration(beta: Array, loglik_samples: Array):
  beta = jnp.asarray(beta)
  loglik_samples = jnp.asarray(loglik_samples)
  mean_loglik = jnp.mean(loglik_samples, axis=(0, 2))
  delta_beta = jnp.diff(beta)
  scaled = delta_beta[None, :, None] * loglik_samples[:, :-1, :]
  n_total = loglik_samples.shape[0] * loglik_samples.shape[2]
  log_ratios = jax.scipy.special.logsumexp(scaled, axis=(0, 2)) - jnp.log(n_total)
  log_evidence = jnp.sum(log_ratios)
  return mean_loglik, log_evidence


def numpyro_model_functions(model, model_args=(), model_kwargs=None, rng_key=None):
  model_kwargs = model_kwargs or {}
  rng_key = jax.random.PRNGKey(0) if rng_key is None else rng_key

  init_params, potential_fn, postprocess_fn_np, _ = initialize_model(
      rng_key, model,
      model_args=model_args, model_kwargs=model_kwargs,
      dynamic_args=False,
  )
  z0 = init_params.z
  flat0, unravel_fn = jax.flatten_util.ravel_pytree(z0)

  seeded = numpyro.handlers.seed(model, rng_key)
  setup_trace = numpyro.handlers.trace(seeded).get_trace(*model_args, **model_kwargs)

  prior_transforms = {}
  prior_dists = {}
  for name, site in setup_trace.items():
    if site["type"] == "sample" and not site["is_observed"]:
      prior_dists[name] = site["fn"]
      prior_transforms[name] = biject_to(site["fn"].support)

  def logdensity_fn(flat_params):
    return -potential_fn(unravel_fn(flat_params))

  def log_prior_fn(flat_params):
    params_dict = unravel_fn(flat_params)
    log_p = 0.0
    for name, transform in prior_transforms.items():
      u = params_dict[name]
      c = transform(u)
      ladj = transform.log_abs_det_jacobian(u, c)
      log_p = log_p + jnp.sum(prior_dists[name].log_prob(c) + ladj)
    return log_p

  def log_likelihood_fn(flat_params):
    params_dict = unravel_fn(flat_params)
    constrained = constrain_fn(
        model, model_args, model_kwargs, params_dict,
        return_deterministic=False,
    )
    seeded_ = numpyro.handlers.seed(model, rng_key)
    conditioned = numpyro.handlers.substitute(seeded_, constrained)
    trace = numpyro.handlers.trace(conditioned).get_trace(*model_args, **model_kwargs)
    ll = 0.0
    for site in trace.values():
      if site["type"] == "sample" and site["is_observed"]:
        ll = ll + jnp.sum(site["fn"].log_prob(site["value"]))
    return ll

  def postprocess_fn(flat_params):
    return postprocess_fn_np(unravel_fn(flat_params))

  return {
      "logdensity_fn": jax.jit(logdensity_fn),
      "log_prior_fn": jax.jit(log_prior_fn),
      "log_likelihood_fn": jax.jit(log_likelihood_fn),
      "postprocess_fn": jax.jit(postprocess_fn),
      "initial_position": flat0,
      "unravel_fn": unravel_fn,
  }


def _sample_initial_particles(model, model_args=(), model_kwargs=None, n_particles=256, rng_key=None):
  model_kwargs = model_kwargs or {}
  rng_key = jax.random.PRNGKey(0) if rng_key is None else rng_key
  keys = jax.random.split(rng_key, n_particles)

  def body_fn(carry, key):
    init_params, _, _, _ = initialize_model(
        key,
        model,
        model_args=model_args,
        model_kwargs=model_kwargs,
        dynamic_args=False,
    )
    flat, _ = jax.flatten_util.ravel_pytree(init_params.z)
    return carry, flat

  _, particles = jax.lax.scan(body_fn, None, keys)
  return particles


def _swap_adjacent_positions(positions, loglik, betas, key, parity, n_chains, n_temperatures):
  dim = positions.shape[-1]
  pos_grid = positions.reshape(n_chains, n_temperatures, dim)
  ll_grid = loglik.reshape(n_chains, n_temperatures)

  n_edges = n_temperatures - 1
  i = jnp.arange(n_edges)
  j = i + 1
  active = (i % 2) == parity

  u = jax.random.uniform(key, shape=(n_chains, n_edges))
  log_alpha = (betas[i] - betas[j]) * (ll_grid[:, j] - ll_grid[:, i])
  accepted = active[None, :] & (jnp.log(u) < jnp.minimum(0.0, log_alpha))

  def swap_leaf(x):
    x_shape = x.shape
    actual_total = x_shape[0]
    curr_n_temp = n_temperatures
    curr_n_chain = actual_total // curr_n_temp

    x_grid = x.reshape(curr_n_chain, curr_n_temp, *x_shape[1:])
    x_i = x_grid[:, i]
    x_j = x_grid[:, j]
    mask = accepted.reshape(curr_n_chain, n_edges, *((1,) * (x.ndim - 1)))
    new_i = jnp.where(mask, x_j, x_i)
    new_j = jnp.where(mask, x_i, x_j)
    x_grid = x_grid.at[:, i].set(new_i).at[:, j].set(new_j)
    return x_grid.reshape(x_shape)

  new_positions = jax.tree.map(swap_leaf, positions)

  ll_grid_i = ll_grid[:, i]
  ll_grid_j = ll_grid[:, j]
  acc_mask = accepted.reshape(n_chains, n_edges)
  new_log_i = jnp.where(acc_mask, ll_grid_j, ll_grid_i)
  new_log_j = jnp.where(acc_mask, ll_grid_i, ll_grid_j)
  ll_grid = ll_grid.at[:, i].set(new_log_i).at[:, j].set(new_log_j)

  return new_positions, ll_grid.reshape(-1), accepted.astype(jnp.float32)


def _make_shard_leaf_fn(n_devices: int, sharding: NamedSharding):
  """Returns a fn that places a pytree leaf's leading axis on `sharding`,
  raising a clear error if that axis doesn't divide evenly across devices."""

  def _shard_leaf(x):
    leading = x.shape[0]
    if leading % n_devices != 0:
      raise ValueError(f"Leading axis of size {leading} is not divisible by {n_devices} devices.")
    return jax.device_put(x, sharding)

  return _shard_leaf


def _make_parallel_runner(mode: ParallelMode, batch_size: int = 1, devices: Optional[list] = None):
  """Used for the (embarrassingly parallel) per-stream warmup runs: each unit
  runs its own independent adaptation, so batching over units is just a map."""
  if mode == "sequential":
    def runner(f, xs):
      return jax.lax.map(f, xs, batch_size=1)
    return runner

  if mode == "vmap":
    def runner(f, xs):
      return jax.lax.map(f, xs, batch_size=batch_size)
    return runner

  if mode == "shard":
    devs = list(devices) if devices is not None else jax.devices()
    n_devices = len(devs)
    mesh = Mesh(np.array(devs).reshape((n_devices,)), axis_names=("shard",))
    sharding = NamedSharding(mesh, P("shard"))
    shard_leaf = _make_shard_leaf_fn(n_devices, sharding)

    def runner(f, xs):
      xs_sharded = jax.tree.map(shard_leaf, xs)
      return jax.vmap(f)(xs_sharded)
    return runner

  raise ValueError(f"Unknown parallel mode {mode!r}; choose 'sequential', 'vmap', or 'shard'.")


def _make_sampling_ops(
    mode: ParallelMode,
    nuts_reinit,
    nuts_step,
    log_likelihood_fn,
    batch_size: int = 1,
    devices: Optional[list] = None,
):
  """Builds (batched_reinit, batched_step, batched_loglik, shard_leaf) for
  the main tempered-sampling scan.

  Unlike warmup, the units here are NOT independent across scan iterations
  (adjacent-temperature swaps couple them every `swap_every` steps), so we
  can't just farm out whole independent runs per unit. Instead this controls
  *how the per-iteration batch across units is computed*:

    - "sequential": one unit at a time (jax.lax.map, batch_size=1) — lowest
      memory footprint, slowest.
    - "vmap": chunked batching via jax.lax.map(..., batch_size=batch_size).
      Setting batch_size == total_units recovers a single full jax.vmap call.
    - "shard": every unit computed simultaneously via jax.vmap, but with the
      per-unit arrays explicitly placed on a multi-device NamedSharding
      *before* the outer jax.lax.scan starts, so XLA's SPMD partitioner
      spreads the whole compiled scan program across `devices`. This shard
      placement is applied here regardless of what warmup_parallel_mode did.

  `nuts_step(state, beta)` advances one NUTS step; the numpyro `HMCState`
  carries and evolves its own rng_key internally, so no external per-step
  key needs to be threaded in here (unlike BlackJAX's stateless kernel.step).
  `nuts_reinit(state, position, beta)` is only used after a temperature
  swap: it recomputes the position-dependent fields of `state` (z, z_grad,
  potential_energy) for the swapped-in position, while preserving that
  slot's already-adapted `state.adapt_state` (step_size/inverse_mass_matrix)
  untouched.
  """

  def _reinit_single(args):
    state, position, beta = args
    return nuts_reinit(state, position, beta)

  def _step_single(args):
    state, beta = args
    return nuts_step(state, beta)

  def _loglik_single(state):
    return log_likelihood_fn(state.z)

  if mode == "sequential":
    def batched_reinit(state, position, beta):
      return jax.lax.map(_reinit_single, (state, position, beta), batch_size=1)

    def batched_step(state, beta):
      return jax.lax.map(_step_single, (state, beta), batch_size=1)

    def batched_loglik(state):
      return jax.lax.map(_loglik_single, state, batch_size=1)

    return batched_reinit, batched_step, batched_loglik, (lambda x: x)

  if mode == "vmap":
    def batched_reinit(state, position, beta):
      return jax.lax.map(_reinit_single, (state, position, beta), batch_size=batch_size)

    def batched_step(state, beta):
      return jax.lax.map(_step_single, (state, beta), batch_size=batch_size)

    def batched_loglik(state):
      return jax.lax.map(_loglik_single, state, batch_size=batch_size)

    return batched_reinit, batched_step, batched_loglik, (lambda x: x)

  if mode == "shard":
    devs = list(devices) if devices is not None else jax.devices()
    n_devices = len(devs)
    mesh = Mesh(np.array(devs).reshape((n_devices,)), axis_names=("shard",))
    sharding = NamedSharding(mesh, P("shard"))
    shard_leaf = _make_shard_leaf_fn(n_devices, sharding)

    batched_reinit = jax.vmap(nuts_reinit, in_axes=(0, 0, 0))
    batched_step = jax.vmap(nuts_step, in_axes=(0, 0))
    batched_loglik = jax.vmap(lambda s: log_likelihood_fn(s.z))

    return batched_reinit, batched_step, batched_loglik, shard_leaf

  raise ValueError(f"Unknown parallel mode {mode!r}; choose 'sequential', 'vmap', or 'shard'.")


class _Checkpointer:
  def __init__(self, checkpoint_dir: Optional[str], resume: bool = True):
    self.dir = checkpoint_dir
    self.resume = resume
    if self.dir is not None:
      os.makedirs(self.dir, exist_ok=True)

  def _path(self, name: str) -> str:
    return os.path.join(self.dir, name)

  def save(self, name: str, obj: Any) -> None:
    if self.dir is None:
      return
    obj_host = jax.tree.map(
        lambda x: jax.device_get(x) if isinstance(x, jax.Array) else x, obj
    )
    final_path = self._path(name)
    tmp_path = final_path + ".tmp"
    with open(tmp_path, "wb") as f:
      pickle.dump(obj_host, f)
    os.replace(tmp_path, final_path)

  def load(self, name: str) -> Optional[Any]:
    if self.dir is None or not self.resume:
      return None
    path = self._path(name)
    if not os.path.exists(path):
      return None
    with open(path, "rb") as f:
      return pickle.load(f)

  def clear(self, name: str) -> None:
    if self.dir is None:
      return
    path = self._path(name)
    if os.path.exists(path):
      os.remove(path)

  # --- sampling-phase, block-level checkpointing ---

  def load_sampling_progress(self):
    return self.load("sampling_progress.pkl")

  def save_sampling_progress(self, next_block: int, states: Any) -> None:
    self.save("sampling_progress.pkl", {"next_block": next_block, "states": states})

  def save_sampling_block(self, block_idx: int, block_outputs: Any) -> None:
    self.save(f"sampling_block_{block_idx:06d}.pkl", block_outputs)

  def load_sampling_block(self, block_idx: int) -> Any:
    obj = self.load(f"sampling_block_{block_idx:06d}.pkl")
    if obj is None:
      raise FileNotFoundError(
          f"Missing checkpointed sampling block {block_idx} in {self.dir!r}; "
          "checkpoint directory is inconsistent (progress marker points past "
          "the last block actually saved)."
      )
    return obj

  def clear_sampling_checkpoints(self) -> None:
    if self.dir is None:
      return
    self.clear("sampling_progress.pkl")
    for name in list(os.listdir(self.dir)):
      if name.startswith("sampling_block_") and name.endswith(".pkl"):
        os.remove(os.path.join(self.dir, name))


def pt_nuts(
    model,
    model_args=(),
    model_kwargs=None,
    n_temperatures: int = 16,
    betas: Optional[Array] = None,
    n_chains_per_temperature: int = 1,
    n_warmup: int = 2048,
    n_samples: int = 4096,
    beta_min: float = 1e-3,
    swap_every: int = 1,
    max_num_doublings: int = 10,
    target_acceptance_rate: float = 0.8,
    is_mass_matrix_diagonal: bool = True,
    seed: int = 0,
    verbose: bool = False,
    checkpoint_dir: Optional[str] = None,
    resume: bool = True,
    checkpoint_every: int = 1,
    warmup_parallel_mode: ParallelMode = "sequential",
    sampling_parallel_mode: ParallelMode = "sequential",
    batch_size: int = 1,
    devices: Optional[list] = None,
):
  """Run parallel-tempered NUTS on a NumPyro model.

  Args:
    model: A NumPyro model callable.
    model_args, model_kwargs: Positional/keyword args forwarded to `model`.
    n_temperatures: Number of rungs on the temperature ladder. Ignored if
      `betas` is given explicitly. `n_temperatures=1` (equivalently, a
      single-element `betas`) switches to a single-temperature mode: plain
      NUTS at that one beta (default 1.0, i.e. the true posterior), with no
      swap proposals and no stepping-stone evidence estimate -- see
      `swap_every` and the `log_evidence`/`swap_acceptance` return fields.
    betas: Optional explicit beta ladder. Either a single value in (0, 1]
      (single-temperature mode, see `n_temperatures`) or a strictly
      increasing ladder from 0 to 1. Overrides `n_temperatures`/`beta_min`.
    n_chains_per_temperature: Independent chains run per temperature rung.
      NOTE: "shard" mode currently only achieves a real multi-device split
      when this is 1 -- see README "Known Limitations".
    n_warmup, n_samples: Warmup and post-warmup sampling steps per chain.
    beta_min: Smallest positive beta on the ladder (rung 0 is always beta=0).
    swap_every: Attempt adjacent-temperature swaps every this many steps.
      Ignored in single-temperature mode (`n_temperatures == 1`), which has
      no adjacent rung to swap with.
    max_num_doublings, target_acceptance_rate, is_mass_matrix_diagonal:
      Forwarded to NumPyro's NUTS as `max_tree_depth`, `target_accept_prob`,
      and `dense_mass=not is_mass_matrix_diagonal` respectively -- the same
      warmup/adaptation code `numpyro.infer.MCMC(NUTS(model))` uses, so a
      single-temperature (`n_temperatures=1`) run reproduces that warmup.
    seed: Integer PRNG seed.
    verbose: Print progress. Uses `pt_nuts.progress_bar` for warmup and,
      when `checkpoint_dir` is None, for the main sampling loop as well
      (both wrap a `jax.lax.scan` and report real incremental progress).
      This requires the optional `jax-tap` package (imported as `jaxtap`)
      -- NOT `pip install jaxtap`, which does not exist; the PyPI package
      is `jax-tap`. Without it, `verbose=True` raises ImportError. When
      `checkpoint_dir` is set, sampling instead uses a plain `tqdm` bar
      that advances once per checkpointed block (see `checkpoint_every`);
      with the default `checkpoint_every=1` this is still per-sample.
    checkpoint_dir: If set, enables checkpointing to this directory:
      (1) warmup results (step sizes, inverse mass matrices) are saved once
      after warmup completes and reloaded on a subsequent call with
      `resume=True`, and
      (2) the main sampling loop is split into blocks of `checkpoint_every`
      samples each; after every block, that block's outputs and the current
      chain state are written to disk, so a crash mid-sampling loses at
      most one block's worth of work rather than the whole run. Resuming
      (`resume=True`, same `checkpoint_dir`) picks up from the next
      unfinished block.
      Resuming assumes the SAME `n_samples`/`checkpoint_every` as the
      interrupted run -- resuming into a run with different values raises
      a clear error rather than silently misassembling results.
    resume: Whether to load existing warmup/sampling checkpoints if present.
    checkpoint_every: Number of samples per checkpointed block during
      sampling (clamped to `[1, n_samples]`). `checkpoint_every=1` skips
      `jax.lax.scan` entirely for each block and calls the per-step
      function directly instead, i.e. sampling becomes a plain Python
      `for` loop over individual samples, host-checkpointing after each
      one. Larger values use `jax.lax.scan` per block (compiled once,
      reused across blocks of the same size) and checkpoint less often.
      Only takes effect when `checkpoint_dir` is set; otherwise sampling
      runs as a single uninterrupted `jax.lax.scan`, as before.
    warmup_parallel_mode: How the `n_chains_per_temperature * n_temperatures`
      independent warmup runs are batched: "sequential" (one at a time),
      "vmap" (chunked via `jax.lax.map(..., batch_size=batch_size)`), or
      "shard" (each run placed on a device of `devices` via a 1-D
      `NamedSharding`, requires the unit count to be divisible by
      `len(devices)`).
    sampling_parallel_mode: How each iteration's per-unit NUTS step is
      batched during the (coupled, due to swaps) main sampling loop. Same
      three options as `warmup_parallel_mode`, applied independently -- it
      does not inherit or depend on `warmup_parallel_mode`. For "shard", the
      per-unit arrays are explicitly placed on `devices` before the
      `jax.lax.scan` starts so the whole compiled loop gets SPMD-partitioned.
    batch_size: Chunk size used by "vmap" mode for both warmup and sampling.
      Set it equal to `n_temperatures * n_chains_per_temperature` to recover
      a single full `jax.vmap` over all units at once.
    devices: Device list used by "shard" mode for both warmup and sampling.
      Defaults to `jax.devices()`.

  Returns:
    A `TemperedNUTSResult` with samples (in constrained space), per-rung
    log-likelihoods, the beta ladder, per-rung mean log-likelihood, the
    stepping-stone log evidence estimate, and swap/NUTS acceptance rates.
    In single-temperature mode (`n_temperatures == 1`), `log_evidence` is
    NaN (not estimable without a temperature ladder) and `swap_acceptance`
    is an empty array.
  """
  model_kwargs = model_kwargs or {}
  if betas is None:
    if n_temperatures == 1:
      betas = jnp.array([1.0])
    else:
      betas = geometric_temperature_ladder(n_temperatures, beta_min=beta_min)
  else:
    betas = jnp.asarray(betas)
    if betas.ndim != 1:
      raise ValueError("betas must be a 1-D array.")
    if betas.shape[0] == 1:
      if not (0.0 < float(betas[0]) <= 1.0):
        raise ValueError("A single-temperature betas array must contain one value in (0, 1].")
    elif not (jnp.all(betas[:-1] < betas[1:]) and betas[0] == 0 and betas[-1] == 1):
      raise ValueError("betas must be strictly increasing, starting at 0 and ending at 1.")
    n_temperatures = betas.shape[0]

  single_temperature = n_temperatures == 1

  ckptr = _Checkpointer(checkpoint_dir, resume=resume)
  total_units = n_chains_per_temperature * n_temperatures

  if verbose:
    print(rf"$\Beta$'s schedule is {betas}.")
    print(f"Initializing model and parameters for {total_units} total parallel streams...")

  key = jax.random.PRNGKey(seed)
  key_init, key_warmup, key_chain = jax.random.split(key, 3)

  fns = numpyro_model_functions(model, model_args=model_args, model_kwargs=model_kwargs, rng_key=key_init)
  log_prior_fn = fns["log_prior_fn"]
  log_likelihood_fn = fns["log_likelihood_fn"]
  postprocess_fn = fns["postprocess_fn"]

  initial_particles = _sample_initial_particles(
      model, model_args=model_args, model_kwargs=model_kwargs,
      n_particles=total_units, rng_key=key_init,
  )

  betas_flat = jnp.tile(betas, n_chains_per_temperature)

  def tempered_logdensity(position, beta):
    return log_prior_fn(position) + beta * log_likelihood_fn(position)

  def potential_fn_gen(beta):
    def potential_fn(position):
      return -tempered_logdensity(position, beta)
    return potential_fn

  # A single hmc() instance is shared by every unit's warmup and sampling
  # call: hmc() stores the NUTS/warmup configuration (num_warmup, tree
  # depth, ...) in nonlocal closure state set by the first init_kernel()
  # call, and every unit uses the same n_warmup/max_num_doublings, so this
  # is safe to share -- see numpyro.infer.hmc.hmc's docstring. sample_kernel
  # self-adapts while state.i < num_warmup and freezes automatically past
  # it, so the same function drives both warmup and post-warmup sampling.
  init_kernel, sample_kernel = hmc(potential_fn_gen=potential_fn_gen, algo="NUTS")

  # init_kernel is what actually populates that nonlocal state (a plain
  # Python-level side effect of tracing it, independent of the position it
  # is traced with); sample_kernel reads it unconditionally (e.g. `state.i
  # < wa_steps`) and crashes with a confusing TypeError if it was never
  # set. Below, the "load warmup from checkpoint" branch skips
  # _warmup_single (hence init_kernel) entirely, so prime it here,
  # unconditionally, before that branch -- the returned state is discarded,
  # only the config-only (position-independent) side effect matters. This
  # is a no-op duplicate of what _warmup_single's own init_kernel call does
  # in the "compute fresh warmup" branch below.
  init_kernel(
      initial_particles[0], num_warmup=n_warmup,
      target_accept_prob=target_acceptance_rate,
      dense_mass=not is_mass_matrix_diagonal,
      max_tree_depth=max_num_doublings,
      model_args=(betas_flat[0],), rng_key=key_warmup,
  )

  if verbose:
    print(f"Running window adaptation for {n_warmup} steps per stream (parallel_mode={warmup_parallel_mode!r})...")

  warmup_keys = jax.random.split(key_warmup, total_units)

  def _warmup_single(position, beta, key):
    state = init_kernel(
        position,
        num_warmup=n_warmup,
        target_accept_prob=target_acceptance_rate,
        dense_mass=not is_mass_matrix_diagonal,
        max_tree_depth=max_num_doublings,
        model_args=(beta,),
        rng_key=key,
    )

    def body(s, _):
      return sample_kernel(s, model_args=(beta,)), None

    state, _ = jax.lax.scan(body, state, None, length=n_warmup)
    return state

  warmup_runner = _make_parallel_runner(warmup_parallel_mode, batch_size=batch_size, devices=devices)
  warmup_ckpt = ckptr.load("warmup_checkpoint.pkl")

  if warmup_ckpt is not None:
    if verbose:
      print("Loading warmup from checkpoint...")
    if "warmup_state" not in warmup_ckpt:
      raise ValueError(
          f"Warmup checkpoint in {checkpoint_dir!r} predates saving the "
          "full post-warmup HMCState. Delete the checkpoint directory and "
          "rerun with this version of pt_nuts."
      )
    warmup_states = warmup_ckpt["warmup_state"]
  else:
    with progress_bar(label="Warmup parallel streams"):
      warmup_states = warmup_runner(
          lambda args: _warmup_single(args[0], args[1], args[2]),
          (initial_particles, betas_flat, warmup_keys),
      )
    ckptr.save("warmup_checkpoint.pkl", {"warmup_state": warmup_states})

  if verbose and single_temperature:
    print(f"Warmup finished. Initial step size found: {np.asarray(warmup_states.adapt_state.step_size).squeeze()}")

  def nuts_step(state, beta):
    return sample_kernel(state, model_args=(beta,))

  def nuts_reinit(state, position, beta):
    # Only used right after a temperature swap: recompute the
    # position-dependent fields for the swapped-in position while keeping
    # this slot's already-adapted state.adapt_state (step_size,
    # inverse_mass_matrix, ...) and state.i (so adaptation stays frozen)
    # untouched -- mirrors what blackjax.nuts.init(position, ...) did, but
    # without discarding the adaptation results as blackjax's did.
    pe_fn = potential_fn_gen(beta)
    potential_energy, z_grad = jax.value_and_grad(pe_fn)(position)
    return state._replace(
        z=position, z_grad=z_grad, potential_energy=potential_energy,
        energy=potential_energy, r=None,
    )

  batched_reinit, batched_step, batched_loglik, shard_leaf = _make_sampling_ops(
      sampling_parallel_mode,
      nuts_reinit, nuts_step, log_likelihood_fn,
      batch_size=batch_size, devices=devices,
  )

  betas_flat = jax.tree.map(shard_leaf, betas_flat)
  states = jax.tree.map(shard_leaf, warmup_states)

  def scan_body(states, xs):
    key_swap, it = xs

    states = batched_step(states, betas_flat)
    loglik = batched_loglik(states)
    accept_rate = states.accept_prob

    if single_temperature:
      # Plain NUTS: no adjacent rungs to swap with, so skip the swap
      # machinery entirely (including the otherwise-wasted state reinit).
      no_swap_acc = jnp.zeros((n_chains_per_temperature, 0), dtype=jnp.float32)
      return states, (states.z, loglik, accept_rate, no_swap_acc)

    do_swap = ((it + 1) % swap_every) == 0

    def do_swap_fn(_):
      new_positions, new_loglik, swap_acc = _swap_adjacent_positions(
          states.z, loglik, betas, key_swap, it % 2, n_chains_per_temperature, n_temperatures
      )
      new_states = batched_reinit(states, new_positions, betas_flat)
      return new_states, new_loglik, swap_acc

    def no_swap_fn(_):
      return states, loglik, jnp.zeros((n_chains_per_temperature, n_temperatures - 1), dtype=jnp.float32)

    states, loglik, swap_acc = jax.lax.cond(do_swap, do_swap_fn, no_swap_fn, operand=None)
    return states, (states.z, loglik, accept_rate, swap_acc)

  sample_time_keys = jax.random.split(key_chain, n_samples)
  iter_indices = jnp.arange(n_samples)

  if verbose:
    print(f"Running Parallel Tempering sampling across {total_units} parallel units "
          f"(parallel_mode={sampling_parallel_mode!r})...")

  if checkpoint_dir is None:

    if verbose:
      with progress_bar(label="Sampling ladder"):
        _, (positions_seq, loglik_seq, accept_seq, swap_seq) = jax.lax.scan(
            scan_body, states, (sample_time_keys, iter_indices)
        )
        jax.block_until_ready(loglik_seq)
    else:
      _, (positions_seq, loglik_seq, accept_seq, swap_seq) = jax.lax.scan(
          scan_body, states, (sample_time_keys, iter_indices)
      )
      jax.block_until_ready(loglik_seq)

  else:
    block_size = max(1, min(checkpoint_every, n_samples))
    n_full_blocks, remainder = divmod(n_samples, block_size)
    block_sizes = [block_size] * n_full_blocks + ([remainder] if remainder else [])
    n_blocks = len(block_sizes)
    block_starts = list(np.cumsum([0] + block_sizes[:-1]))

    def _check_block_shape(block_idx, arr, name):
      got = arr.shape[0]
      expected = block_sizes[block_idx]
      if got != expected:
        raise ValueError(
            f"Checkpointed sampling block {block_idx} ({name!r}) has "
            f"{got} samples, expected {expected}. This usually means "
            f"resume=True was used with a different n_samples/checkpoint_every "
            f"than the run that wrote {checkpoint_dir!r}."
        )

    progress = ckptr.load_sampling_progress()
    if progress is not None:
      start_block = progress["next_block"]
      if start_block > n_blocks:
        raise ValueError(
            f"Checkpoint in {checkpoint_dir!r} is past block {n_blocks} but "
            f"the current n_samples/checkpoint_every only produce {n_blocks} "
            "blocks; resume assumes the same settings as the interrupted run."
        )
      if verbose:
        print(f"Resuming sampling from block {start_block}/{n_blocks}...")
      states = progress["states"]
    else:
      start_block = 0

    pbar = tqdm(total=n_samples, desc="Sampling ladder") if (verbose and tqdm is not None) else None
    if pbar is not None and start_block > 0:
      done = n_samples if start_block >= n_blocks else int(block_starts[start_block])
      pbar.update(done)

    block_positions, block_loglik, block_accept, block_swap = [], [], [], []

    for b in range(start_block):
      blk = ckptr.load_sampling_block(b)
      _check_block_shape(b, blk["loglik"], "loglik")
      block_positions.append(blk["positions"])
      block_loglik.append(blk["loglik"])
      block_accept.append(blk["accept"])
      block_swap.append(blk["swap"])

    for b in range(start_block, n_blocks):
      s = block_starts[b]
      size = block_sizes[b]
      keys_blk = sample_time_keys[s:s + size]
      idx_blk = iter_indices[s:s + size]

      if size == 1:
        states, (pos, ll, acc, swp) = scan_body(states, (keys_blk[0], idx_blk[0]))
        pos, ll, acc, swp = jax.tree.map(lambda x: x[None], (pos, ll, acc, swp))
      else:
        states, (pos, ll, acc, swp) = jax.lax.scan(scan_body, states, (keys_blk, idx_blk))

      jax.block_until_ready(ll)

      block_positions.append(pos)
      block_loglik.append(ll)
      block_accept.append(acc)
      block_swap.append(swp)

      ckptr.save_sampling_block(b, {"positions": pos, "loglik": ll, "accept": acc, "swap": swp})
      ckptr.save_sampling_progress(b + 1, states)

      if pbar is not None:
        pbar.update(size)

    if pbar is not None:
      pbar.close()

    positions_seq = jnp.concatenate(block_positions, axis=0)
    loglik_seq = jnp.concatenate(block_loglik, axis=0)
    accept_seq = jnp.concatenate(block_accept, axis=0)
    swap_seq = jnp.concatenate(block_swap, axis=0)

  samples = jnp.swapaxes(positions_seq, 0, 1)
  loglik_samples = jnp.swapaxes(loglik_seq, 0, 1)
  nuts_accept = jnp.swapaxes(accept_seq, 0, 1)
  swap_acceptance = jnp.swapaxes(swap_seq, 0, 1)

  dim = samples.shape[-1]
  samples = samples.reshape(n_chains_per_temperature, n_temperatures, n_samples, dim)
  loglik_samples = loglik_samples.reshape(n_chains_per_temperature, n_temperatures, n_samples)

  nuts_accept = nuts_accept.reshape(n_chains_per_temperature, n_temperatures, n_samples)
  nuts_accept = jnp.mean(nuts_accept, axis=2)

  swap_acceptance = swap_acceptance.reshape(n_samples, n_chains_per_temperature, n_temperatures - 1)
  swap_acceptance = jnp.mean(swap_acceptance, axis=(0, 1))

  if postprocess_fn is not None:
    if verbose:
      print("Post-processing samples into constrained parameter space...")
    flat_shape = samples.shape
    flat_samples = samples.reshape(-1, flat_shape[-1])
    processed_flat = jax.lax.map(postprocess_fn, flat_samples, batch_size=batch_size)
    samples = jax.tree.map(lambda x: x.reshape(flat_shape[:-1] + x.shape[1:]), processed_flat)

  loglik_for_evidence = jnp.transpose(loglik_samples, (2, 1, 0))
  if single_temperature:
    if verbose:
      print("Single-temperature run: skipping evidence estimation (no thermodynamic integration).")
    mean_loglik = jnp.mean(loglik_for_evidence, axis=(0, 2))
    log_evidence = jnp.asarray(jnp.nan)
  else:
    if verbose:
      print("Computing stepping stone marginal likelihood estimate...")
    mean_loglik, log_evidence = stepping_stone_integration(betas, loglik_for_evidence)

  return TemperedNUTSResult(
      samples=samples,
      loglik=loglik_samples,
      temperatures=betas,
      mean_loglik=mean_loglik,
      log_evidence=log_evidence,
      swap_acceptance=swap_acceptance,
      nuts_acceptance=nuts_accept,
  )
