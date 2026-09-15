import os
import pickle
import sys

os.environ["CHIMERA_USE_x64"] = "True"
os.environ["CHIMERA_ENABLE_GPU"] = "True"
sys.path.append("/leonardo/home/userexternal/mtagliaz/softwares/CHIMERA/")
from CHIMERA import data
from CHIMERA.data import theta_inj_det
from CHIMERA.cosmo import flrw
from CHIMERA.mass.paired import neural_density
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

run_name = 'nn_2x16'

nn_config = {
  'hidden_size': 16,
  'depth': 2,
  'input_size': 1,
}

def nn_param_shapes(nn_config):
    """Compute per-layer W and b shapes from nn_config."""
    input_size = nn_config.get('input_size', 1)
    hidden_size = nn_config['hidden_size']
    depth = nn_config['depth']
    sizes = [input_size] + [hidden_size] * depth

    W_shapes = [(sizes[i + 1], sizes[i]) for i in range(depth)]
    W_shapes.append((1, hidden_size))  # output layer, no bias

    b_shapes = [(sizes[i + 1],) for i in range(depth)]  # hidden layers only

    return W_shapes, b_shapes

def reconstruct_Ws_bs(posterior, W_shapes, b_shapes):
    Ws_samples = [jnp.asarray(posterior[f"W{i}"]) for i in range(len(W_shapes))]
    bs_samples = [jnp.asarray(posterior[f"b{i}"]) for i in range(len(b_shapes))]
    return Ws_samples, bs_samples

W_shapes, b_shapes = nn_param_shapes(nn_config)

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
    "H0": dist.Uniform(10., 200.),
    "bottomsmooth": dist.LogUniform(0.01, 1.),
    "m_high": dist.Uniform(50., 200.),
    "m_low": dist.Uniform(0.4, 1.4),
    "topsmooth": dist.LogUniform(0.001, 1.),
    "beta": dist.Uniform(-4., 12.),
    "Ws": norm(loc=0., scale=3.5),
    "bs": norm(loc=0., scale=4.0),
    "gamma": dist.Uniform(0., 12.),
    "kappa": dist.Uniform(0., 6.),
    "zp": dist.Uniform(0., 4.),
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
mass = neural_density(**nn_config)
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

def chimera_model(hyperlike, priors, W_shapes, b_shapes):
    hyperparams = {}
    for name, prior_dist in priors.items():
        if name == 'Ws':
            Ws = []
            for i, shape in enumerate(W_shapes):
                W = numpyro.sample(f"W{i}", prior_dist.expand(shape).to_event(len(shape)))
                Ws.append(W)
            hyperparams[name] = Ws
        elif name == 'bs':
            bs = []
            for i, shape in enumerate(b_shapes):
                b = numpyro.sample(f"b{i}", prior_dist.expand(shape).to_event(len(shape)))
                bs.append(b)
            hyperparams[name] = bs
        else:
            hyperparams[name] = numpyro.sample(name, prior_dist)
    loglike = hyperlike(**hyperparams)
    numpyro.factor("log_likelihood", loglike)

model_kwargs = dict(hyperlike=hyperlike, priors=priors, W_shapes=W_shapes, b_shapes=b_shapes)

result = pt_nuts(
    chimera_model,
    model_kwargs=model_kwargs,
    n_temperatures=8,
    n_chains_per_temperature=1,
    n_warmup=2500,
    n_samples=5000,
    swap_every=1,
    max_num_doublings=6,       # analogous to the old max_tree_depth=6
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
