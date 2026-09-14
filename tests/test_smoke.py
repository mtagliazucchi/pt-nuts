import numpy as np
import pytest
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from pt_nuts import pt_nuts, geometric_temperature_ladder


def _gaussian_model(y=None):
    mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
    numpyro.sample("obs", dist.Normal(mu, 1.0), obs=y)


Y_OBS = jnp.array([1.2, 0.8, 1.5, 0.9, 1.1])

COMMON_KW = dict(
    model_kwargs=dict(y=Y_OBS),
    n_temperatures=4,
    n_chains_per_temperature=1,
    n_warmup=10,
    n_samples=10,
    swap_every=1,
    warmup_parallel_mode="sequential",
    seed=0,
    verbose=False,
)


def test_geometric_temperature_ladder_shape_and_bounds():
    betas = geometric_temperature_ladder(5, beta_min=1e-3)
    assert betas.shape == (5,)
    assert float(betas[0]) == 0.0
    assert float(betas[-1]) == 1.0
    assert bool(jnp.all(jnp.diff(betas) > 0))


def test_sequential_mode_runs_and_returns_expected_shapes():
    res = pt_nuts(_gaussian_model, sampling_parallel_mode="sequential", **COMMON_KW)
    assert res.loglik.shape == (1, 4, 10)
    assert res.temperatures.shape == (4,)
    assert res.samples["mu"].shape == (1, 4, 10)
    assert np.isfinite(np.asarray(res.log_evidence))


def _single_step_kwargs():
    """swap_every huge + n_samples=1 isolates exactly one NUTS step from an
    identical initial state, with no room for the swap step or trajectory
    branching to compound any floating-point differences between batching
    strategies. This is the right check for "did the refactor change the
    actual math", as opposed to bit-identity over a full chain."""
    kw = dict(COMMON_KW)
    kw["n_samples"] = 1
    kw["swap_every"] = 10_000
    return kw


def test_sequential_and_chunked_vmap_modes_agree_on_a_single_step():
    """sequential and vmap both go through the same jax.lax.map codepath
    (batch_size=1 vs 2), so this isn't comparing two different
    implementations -- it's comparing two XLA batching strategies for one
    implementation. Batching can still make XLA dispatch different
    (vectorized) kernels internally, which can sum/reduce in a different
    order; at float32 that alone is enough for a single leapfrog step to
    differ by a couple ULPs. rtol=1e-5 comfortably tolerates that kernel-
    dispatch noise while still failing hard on an actual math bug, which
    would show up many orders of magnitude larger."""
    kw = _single_step_kwargs()
    seq = pt_nuts(_gaussian_model, sampling_parallel_mode="sequential", **kw)
    vmp = pt_nuts(_gaussian_model, sampling_parallel_mode="vmap", batch_size=2, **kw)
    np.testing.assert_allclose(seq.loglik, vmp.loglik, rtol=1e-5)


@pytest.mark.skipif(len(jax.devices()) < 2, reason="needs >=2 devices to exercise real sharding")
def test_shard_mode_matches_sequential_on_a_single_step_and_is_actually_distributed():
    devs = jax.devices()
    kw = _single_step_kwargs()
    seq = pt_nuts(_gaussian_model, sampling_parallel_mode="sequential", **kw)
    shd = pt_nuts(_gaussian_model, sampling_parallel_mode="shard", devices=devs, **kw)

    np.testing.assert_allclose(seq.loglik, shd.loglik)

    devices_used = {str(s.device) for s in shd.loglik.addressable_shards}
    assert len(devices_used) > 1, "shard mode did not actually spread data across devices"


def test_modes_stay_statistically_close_over_a_longer_run():
    """Over many steps, tiny floating-point differences between batching
    strategies (vmap vs. chunked map vs. sharded) can flip a branch
    decision inside NUTS's trajectory doubling (an accept/reject or
    U-turn comparison), sending the chain down a different-but-equally-
    valid path. That's expected chaotic sensitivity, not a bug — so this
    checks statistical agreement, not bit-identity, over a longer chain."""
    kw = dict(COMMON_KW)
    kw["n_samples"] = 200
    kw["n_warmup"] = 200
    seq = pt_nuts(_gaussian_model, sampling_parallel_mode="sequential", **kw)
    vmp = pt_nuts(_gaussian_model, sampling_parallel_mode="vmap", batch_size=2, **kw)
    np.testing.assert_allclose(
        np.mean(seq.loglik, axis=-1), np.mean(vmp.loglik, axis=-1), rtol=0.2
    )


@pytest.mark.skipif(len(jax.devices()) < 2, reason="needs >=2 devices to exercise the guard")
def test_shard_mode_raises_on_non_divisible_unit_count():
    devs = jax.devices()  # 4 simulated devices
    kw = dict(COMMON_KW)
    kw["n_temperatures"] = 3  # 3 units, not divisible by 4 devices
    with pytest.raises(ValueError):
        pt_nuts(_gaussian_model, sampling_parallel_mode="shard", devices=devs, **kw)


def test_shard_sampling_mode_independent_of_warmup_mode():
    """Regression test: sampling_parallel_mode must not silently ride along
    on whatever warmup_parallel_mode happened to produce."""
    devs = jax.devices()
    if len(devs) < 2:
        pytest.skip("needs >=2 devices")
    kw = dict(COMMON_KW)
    kw["warmup_parallel_mode"] = "sequential"  # deliberately NOT shard
    res = pt_nuts(_gaussian_model, sampling_parallel_mode="shard", devices=devs, **kw)
    devices_used = {str(s.device) for s in res.loglik.addressable_shards}
    assert len(devices_used) > 1


def test_checkpointed_sampling_matches_uninterrupted_run(tmp_path):
    """Splitting sampling into blocks and checkpointing each one must not
    change the result: an uninterrupted checkpointed run (block size > 1)
    should match a non-checkpointed run exactly."""
    kw = dict(COMMON_KW)
    kw["n_samples"] = 12
    ref = pt_nuts(_gaussian_model, **kw)
    checkpointed = pt_nuts(_gaussian_model, checkpoint_dir=str(tmp_path), checkpoint_every=4, **kw)
    np.testing.assert_allclose(ref.loglik, checkpointed.loglik)


def test_checkpoint_every_one_unrolls_to_a_for_loop_and_matches(tmp_path):
    """checkpoint_every=1 skips jax.lax.scan per block entirely (each
    'block' is one direct function call in a Python for loop). Must still
    match the reference numerically."""
    kw = dict(COMMON_KW)
    kw["n_samples"] = 6
    ref = pt_nuts(_gaussian_model, **kw)
    checkpointed = pt_nuts(_gaussian_model, checkpoint_dir=str(tmp_path), checkpoint_every=1, **kw)
    np.testing.assert_allclose(ref.loglik, checkpointed.loglik)


def test_checkpoint_resume_after_simulated_crash_matches_uninterrupted_run(tmp_path):
    """The real point of checkpointing: a crash partway through sampling,
    followed by a resumed call, must reconstruct exactly what an
    uninterrupted run would have produced."""
    from pt_nuts.sampler import _Checkpointer

    kw = dict(COMMON_KW)
    kw["n_samples"] = 12
    ref = pt_nuts(_gaussian_model, **kw)

    checkpoint_dir = str(tmp_path)
    orig_save_block = _Checkpointer.save_sampling_block
    call_count = {"n": 0}

    def flaky_save_block(self, block_idx, block_outputs):
        orig_save_block(self, block_idx, block_outputs)
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated crash")

    _Checkpointer.save_sampling_block = flaky_save_block
    try:
        with pytest.raises(RuntimeError):
            pt_nuts(_gaussian_model, checkpoint_dir=checkpoint_dir, checkpoint_every=4, **kw)
    finally:
        _Checkpointer.save_sampling_block = orig_save_block

    resumed = pt_nuts(_gaussian_model, checkpoint_dir=checkpoint_dir, checkpoint_every=4, resume=True, **kw)
    np.testing.assert_allclose(ref.loglik, resumed.loglik)


def test_checkpoint_resume_with_mismatched_settings_raises(tmp_path):
    """Resuming with a different n_samples/checkpoint_every than the
    interrupted run must fail loudly, not silently misassemble results."""
    checkpoint_dir = str(tmp_path)
    kw = dict(COMMON_KW)
    kw.pop("n_samples")
    pt_nuts(_gaussian_model, checkpoint_dir=checkpoint_dir, checkpoint_every=4, n_samples=12, **kw)
    with pytest.raises(ValueError):
        pt_nuts(
            _gaussian_model, checkpoint_dir=checkpoint_dir, checkpoint_every=4,
            n_samples=8, resume=True, **kw,
        )


def test_single_temperature_runs_plain_nuts_without_evidence_or_swaps():
    """n_temperatures=1 should behave like plain NUTS at beta=1: no swap
    machinery, no stepping-stone evidence estimate."""
    kw = dict(COMMON_KW)
    kw["n_temperatures"] = 1
    res = pt_nuts(_gaussian_model, **kw)
    assert res.loglik.shape == (1, 1, 10)
    assert res.temperatures.shape == (1,)
    assert float(res.temperatures[0]) == 1.0
    assert res.swap_acceptance.shape == (0,)
    assert np.isnan(np.asarray(res.log_evidence))
    assert np.isfinite(np.asarray(res.mean_loglik)).all()


def test_single_temperature_accepts_explicit_beta():
    kw = dict(COMMON_KW)
    kw.pop("n_temperatures")
    res = pt_nuts(_gaussian_model, betas=jnp.array([0.5]), **kw)
    assert float(res.temperatures[0]) == 0.5
    assert res.samples["mu"].shape == (1, 1, 10)
