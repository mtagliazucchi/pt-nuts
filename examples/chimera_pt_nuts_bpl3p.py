import os
import pickle
import sys

os.environ["CHIMERA_USE_x64"] = "True"
os.environ["CHIMERA_ENABLE_GPU"] = "True"
sys.path.append("/leonardo/home/userexternal/mtagliaz/softwares/CHIMERA/")
from CHIMERA import data
from CHIMERA.data import theta_inj_det
from CHIMERA.cosmo import flrw
from CHIMERA.mass.paired import bpl_dip_three_peaks
from CHIMERA.rate import madau_dickinson
from CHIMERA import population, selection_function, hyperlikelihood

import h5py
import jax, jax.numpy as jnp
import numpy as np
import numpyro, numpyro.distributions as dist
from pt_nuts import pt_nuts


devices = jax.devices('gpu')
print(f"Available devices: {devices}")

dir_data = '/leonardo_work/IscrC_MLGW/gwtc5/data/'
dir_chain = '/leonardo_work/IscrC_MLGW/gwtc5/res/nuts/'
dir_home = os.path.expanduser('~')

run_name = 'bpl3p'

def uniform(low, high):
  return dist.Uniform(low=low, high=high)

def loguniform(low, high):
  return dist.LogUniform(low=low, high=high)

def norm(loc, scale):
  return dist.Normal(loc=loc, scale=scale)

def beta(alpha, beta):
    return dist.Beta(alpha, beta)

# Priors
priors = {
    "H0": uniform(10., 200.),
    "alpha_1": uniform(-4, 12.),
    "alpha_2": uniform(-4, 12.),
    "beta_bottom": uniform(-4., 12.),
    "beta_top": uniform(-4., 12.),
    "mu_g1": uniform(5., 150.),
    "sigma_g1": uniform(0.4, 10.),
    "mu_g2": uniform(5.0, 150.),
    "sigma_g2": uniform(0.4, 15.),
    "mu_g3": uniform(5.0, 150.),
    "sigma_g3": uniform(0.4, 15.),
    "lambda_g": uniform(0, 1),
    "lambda_1": beta(alpha=1, beta=2),
    "lambda_2": uniform(0, 1),
    "bottomsmooth": loguniform(0.01, 1.),
    "topsmooth": loguniform(0.001, 1.),
    "leftdip": uniform(1.5, 3.0),
    "rightdip": uniform(5.0, 9.0),
    "leftdipsmooth": loguniform(0.01, 2.0),
    "rightdipsmooth": loguniform(0.01, 2.0),
    "deep": uniform(0., 1.),
    "m_low": uniform(0.4, 1.4),
    "m_high": uniform(50., 200.),
    "gamma": uniform(0., 12.),
    "kappa": uniform(0., 6.),
    "zp": uniform(0., 4.),
}

params_keys = list(priors.keys())
n_dim = len(params_keys)
prior_dists = [priors[k] for k in params_keys]

## Load PE data
file_ev = dir_data + 'GWTC5_242CBC_FAR0.25_PE2048.h5'
pe_gw = data.load_gw_pe_samples(file_ev, parameters=['m1det', 'm2det', 'dL'], return_struct=True)
pe_prior = jnp.load(dir_data + 'GWTC5_242CBC_FAR0.25_PE2048prior.npy')
pe_gw = pe_gw.update(pe_prior=pe_prior)
Nev = len(pe_gw.dL)

## Load injection data
file_inj = dir_data + "GWTC5_injections_far0.25.h5"
f_inj = h5py.File(file_inj)
Ngen = float(f_inj.attrs['Ngen'])
Tobs = float(f_inj.attrs['Tobs'])
theta_inj_det = data.load_injection_data(file_inj, snr_cut=None, key_mapping={'snr': 'snr', 'log_pdraw': 'log_pdraw'}, frame='detector')

sel_fcn = selection_function(theta_inj_det, N_inj=Ngen)

# Instantiate hyperlikelihood object
cosmo = flrw(H0=67.9, Om0=0.3065, z_max=5.)
mass = bpl_dip_three_peaks()
rate = madau_dickinson()
pop_model = population(cosmo, mass, rate, scale_free=True, Tobs=Tobs, R0=1)

hyperlike = hyperlikelihood(
    # data
    theta_gw_det=pe_gw,
    # population
    population=pop_model,
    # integration grid resolution
    z_grids_res=300,
    # selection function
    selection_function=sel_fcn,
    # numerical stability
    pe_neff=2.0,
    inj_neff=None,  # default to 5*Nev
    # KDE settings
    kind_kde='fft',
    kernel='gaussian',
    kde_bw=None,  # default to scott
    # num_bins=200,
)

def chimera_model(hyperlike, priors):
    hyperparams = {}
    for name, prior_dist in priors.items():
        hyperparams[name] = numpyro.sample(name, prior_dist)
    loglike = hyperlike(**hyperparams)
    numpyro.factor("log_likelihood", loglike)

model_kwargs = dict(hyperlike=hyperlike, priors=priors)

result = pt_nuts(
    chimera_model,
    model_kwargs=model_kwargs,
    n_temperatures=8,
    n_chains_per_temperature=1,
    n_warmup=2500,
    n_samples=2500,
    swap_every=1,
    max_num_doublings=5,       # analogous to the old max_tree_depth=5
    target_acceptance_rate=0.9,
    seed=0,
    verbose=True,
    checkpoint_dir=os.path.join(dir_chain, run_name),
    checkpoint_every=500,
    resume=True,
    warmup_parallel_mode="sequential",
    sampling_parallel_mode="sequential",
)

print(f"log evidence estimate: {result.log_evidence}")
print(f"per-rung NUTS acceptance: {result.nuts_acceptance}")
print(f"adjacent-pair swap acceptance: {result.swap_acceptance}")

# Persist the full result (posterior samples at every temperature rung,
# per-sample log-likelihoods, the beta ladder, and diagnostics) to the
# user's home directory. Heavy per-block sampling checkpoints stay under
# dir_chain on leonardo_work (see checkpoint_dir above) since home
# directories on HPC clusters are typically quota-limited; this is just
# the final, comparatively small combined result.
output_file = os.path.join(dir_home, f'pt_nuts_{run_name}_result.pkl')
result_host = {
    "samples": jax.tree.map(np.asarray, result.samples),
    "loglik": np.asarray(result.loglik),
    "temperatures": np.asarray(result.temperatures),
    "mean_loglik": np.asarray(result.mean_loglik),
    "log_evidence": np.asarray(result.log_evidence),
    "swap_acceptance": np.asarray(result.swap_acceptance),
    "nuts_acceptance": np.asarray(result.nuts_acceptance),
}
with open(output_file, "wb") as f:
    pickle.dump(result_host, f)

print(f"Wrote result to {output_file}")
