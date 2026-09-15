# pt-nuts

Parallel-tempered NUTS sampling for [NumPyro](https://num.pyro.ai/) models,
built directly on NumPyro's own HMC/NUTS kernel (`numpyro.infer.hmc`) and
JAX -- the same warmup/adaptation code `numpyro.infer.MCMC(NUTS(model))`
uses.

Runs a ladder of NUTS chains at different inverse-temperatures (betas),
periodically proposes swaps between adjacent temperatures (replica
exchange / parallel tempering), and estimates the log model evidence via
stepping-stone thermodynamic integration.

## Installation

```bash
pip install jax  # or jax[cuda12] / jax[tpu] etc. — pick the build for your hardware
pip install pt_nuts
```

For the local editable install from a clone:

```bash
git clone https://github.com/mtagliazucchi/pt-nuts
cd pt-nuts
pip install -e ".[dev]"
```

## Quick start

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from pt_nuts import pt_nuts

def model(y=None):
    mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
    numpyro.sample("obs", dist.Normal(mu, 1.0), obs=y)

y_obs = jnp.array([1.2, 0.8, 1.5, 0.9, 1.1])

result = pt_nuts(
    model,
    model_kwargs=dict(y=y_obs),
    n_temperatures=8,
    n_chains_per_temperature=1,
    n_warmup=1000,
    n_samples=2000,
    swap_every=1,
    seed=0,
)

print(result.samples["mu"].shape)   # (n_chains, n_temperatures, n_samples)
print(result.log_evidence)          # scalar log model evidence estimate
print(result.nuts_acceptance)       # per-rung NUTS acceptance rate
print(result.swap_acceptance)       # per-adjacent-pair swap acceptance rate
```

`result` is a `TemperedNUTSResult` with fields:

| field | shape | meaning |
|---|---|---|
| `samples` | `(n_chains, n_temperatures, n_samples, ...)` per site | posterior samples in constrained space (pytree, one array per NumPyro sample site) |
| `loglik` | `(n_chains, n_temperatures, n_samples)` | per-sample log-likelihood at each rung |
| `temperatures` | `(n_temperatures,)` | the beta ladder used |
| `mean_loglik` | `(n_temperatures,)` | mean log-likelihood per rung |
| `log_evidence` | scalar | stepping-stone log evidence estimate |
| `swap_acceptance` | `(n_temperatures - 1,)` | acceptance rate of each adjacent-pair swap |
| `nuts_acceptance` | `(n_chains, n_temperatures)` | mean NUTS acceptance rate per rung |

## Parallel execution modes

Two independent phases each accept a mode: `warmup_parallel_mode` (NumPyro's
NUTS window adaptation, run once per chain-temperature unit) and
`sampling_parallel_mode` (the main tempered sampling loop). **They are
independent of each other** — setting one has no effect on the other.

- **`"sequential"`** — process one unit at a time
  (`jax.lax.map(..., batch_size=1)`). Lowest memory footprint, slowest.
- **`"vmap"`** — chunked batching via `jax.lax.map(..., batch_size=batch_size)`.
  Set `batch_size` equal to the total number of units
  (`n_temperatures * n_chains_per_temperature`) to get a single, full
  `jax.vmap` over everything at once (fastest on one device, highest memory).
- **`"shard"`** — spreads the units across `devices` (defaults to
  `jax.devices()`) using an explicit `NamedSharding`, so XLA's SPMD
  partitioner distributes the compiled computation across multiple
  accelerators. Requires `n_temperatures * n_chains_per_temperature` to be
  divisible by `len(devices)` (raises `ValueError` otherwise).

```python
result = pt_nuts(
    model, model_kwargs=dict(y=y_obs),
    n_temperatures=8, n_chains_per_temperature=1,
    n_warmup=1000, n_samples=2000,
    warmup_parallel_mode="shard",
    sampling_parallel_mode="shard",
    devices=jax.devices(),  # e.g. 8 GPUs, one per temperature rung
)
```

Why the two phases need different treatment: during warmup, each
chain-temperature unit runs a completely independent adaptation, so
`"sequential"`/`"vmap"`/`"shard"` just farm out independent whole runs.
During sampling, units are *not* independent across iterations — the
adjacent-temperature swap step needs every unit's state at once every
`swap_every` steps — so these modes instead control how the batched
per-iteration NUTS step is computed, not whether whole runs are split up.

## Checkpointing

If `checkpoint_dir` is set, checkpointing covers both phases:

- **Warmup**: step sizes and inverse mass matrices are saved once after
  warmup completes, and reloaded on a subsequent call with `resume=True`
  (the default), letting you skip re-running warmup entirely.
- **Sampling**: the run is split into blocks of `checkpoint_every` samples.
  After each block, that block's outputs and the current chain state are
  written to disk, so a crash mid-sampling loses at most one block's worth
  of work — not the whole run. Resuming (`resume=True`, same
  `checkpoint_dir`) picks up from the next unfinished block, reconstructing
  earlier blocks from disk.

```python
result = pt_nuts(
    model, model_kwargs=dict(y=y_obs),
    n_temperatures=8, n_warmup=1000, n_samples=5000,
    checkpoint_dir="./ckpt", checkpoint_every=100,  # write to disk every 100 samples
)
# if this crashes, calling it again with the same checkpoint_dir and
# resume=True (the default) picks up from the last completed block
```

`checkpoint_every=1` skips `jax.lax.scan` for each block entirely and calls
the per-step function directly in a plain Python `for` loop instead — i.e.
sampling becomes a genuine unrolled loop over individual samples, with a
host-side checkpoint write after every one. This trades speed (Python loop
overhead per sample, no fused compiled loop) for the finest possible
checkpoint granularity. Larger `checkpoint_every` values use
`jax.lax.scan` per block as usual (compiled once, reused across
equal-sized blocks) and checkpoint less often.

**Resuming assumes the same `n_samples`/`checkpoint_every` as the
interrupted run** — resuming into a run configured differently raises a
clear `ValueError` rather than silently misassembling results.

When `checkpoint_dir` is not set, sampling runs exactly as before: one
uninterrupted, uncheckpointed `jax.lax.scan` over all `n_samples`.

## Known limitations

- **Different modes are not bit-identical over long runs.** A single
  isolated step is bit-identical across `"sequential"`/`"vmap"`/`"shard"`
  (verified in the test suite). Over many steps, tiny floating-point
  rounding differences from different batching/summation order can
  eventually flip a branch decision inside NUTS's trajectory doubling
  (an accept/reject or U-turn comparison), after which the chain follows
  a different-but-equally-valid path. This is expected chaotic sensitivity
  inherent to branchy HMC/NUTS integration, not a correctness bug — the
  modes remain statistically equivalent, just not bitwise-identical
  sample-for-sample. The same applies to a checkpointed run vs. an
  uninterrupted one at different block sizes, though a *given*
  `checkpoint_every` reproduces the uninterrupted result bit-for-bit
  (verified in the test suite), since the per-step keys are precomputed
  once upfront regardless of blocking.
- **`"shard"` mode's real multi-device benefit is currently strongest when
  `n_chains_per_temperature == 1`.** With more than one chain per
  temperature, results remain numerically correct, but the reshape from
  the flat per-unit axis into `(n_chains, n_temperatures)` doesn't align
  cleanly with a 1-D device mesh, and XLA may fall back to a fully
  replicated array instead of a true per-device split — you lose the
  memory/compute distribution benefit silently, without an error.

## Requirements

- Python >= 3.10
- `jax`, `numpyro`, `numpy`, `tqdm`, and (optional, for `verbose=True`'s live progress bar) `jax-tap`

## Development

```bash
pip install -e ".[dev]"
pytest
```

The test suite simulates multiple CPU devices (`XLA_FLAGS`, set in
`tests/conftest.py`) to exercise `"shard"` mode without needing real
accelerators.

## License

MIT — see `LICENSE`.
