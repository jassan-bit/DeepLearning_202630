"""Predeclared stationary Gaussian crossed DGPs, no market inputs.

All Gaussian AR features initialize from independent N(0,1), their exact
joint stationary law because each feature uses its own independent shocks.
Common factors have common phi; distinct model/idio factors may use phi_m.
"""
import numpy as np
from crossed_rmse_v2_frozen import require


def bases_for(case):
    bases = np.asarray(case['bases'], dtype=float)
    require(bases.shape == (case['M'],) and np.isfinite(bases).all()
            and (bases > 0).all(), 'Positive declared risk bases required')
    return bases


def feature_phis(case):
    R, M = case['R'], case['M']
    model_phi = np.asarray(case.get('model_phi', [case['phi']]*M), dtype=float)
    require(model_phi.shape == (M,) and (np.abs(model_phi) < 1).all(), 'Invalid model AR coefficients')
    return np.concatenate([[case['phi']], model_phi,
                           np.repeat(case['phi'], R), np.tile(model_phi, R)])


def stationary_gaussian_features(rng, phis, T):
    phis = np.asarray(phis, dtype=float)
    require(phis.ndim == 1 and (np.abs(phis) < 1).all(), 'Stationary AR required')
    state = rng.normal(size=len(phis))
    out = np.empty((T, len(phis)))
    shock_scale = np.sqrt(1-phis*phis)
    for t in range(T):
        # Each coordinate is independent in stationarity; shared cross-model
        # dependence enters only through explicitly shared factors below.
        state = phis*state+shock_scale*rng.normal(size=len(phis))
        out[t] = state
    return out


def make_squared_errors(case, rng):
    R, T, M = case['R'], case['T'], case['M']
    bases = bases_for(case)
    sigma = np.asarray(case['seed_sigma'], dtype=float)
    if sigma.ndim == 0:
        sigma = np.repeat(sigma, M)
    require(sigma.shape == (M,) and (sigma >= 0).all(), 'Invalid seed scale vector')
    latent = np.sqrt(.5)*rng.normal(size=(R, 1))+np.sqrt(.5)*rng.normal(size=(R, M))
    amplitude = bases[None, :]*np.exp(sigma[None, :]*latent-sigma[None, :]**2/2)
    weights = np.asarray(case['weights'], dtype=float)
    require(weights.shape == (4,) and (weights >= 0).all(), 'Four nonnegative weights required')
    if weights.sum() == 0:
        z = np.ones((R, T, M))
    else:
        require(abs(float(weights.sum())-1) < 1e-12, 'Factor variances must sum to one')
        values = stationary_gaussian_features(rng, feature_phis(case), T)
        global_time = values[:, 0]
        model_time = values[:, 1:1+M]
        seed_time = values[:, 1+M:1+M+R]
        idio = values[:, 1+M+R:].reshape(T, R, M).transpose(1, 0, 2)
        z = (np.sqrt(weights[0])*global_time[None, :, None]
             +np.sqrt(weights[1])*model_time[None, :, :]
             +np.sqrt(weights[2])*seed_time.T[:, :, None]
             +np.sqrt(weights[3])*idio)
    q = (amplitude[:, None, :]*z)**2
    # Every z coordinate has population second moment 1, E amplitude=bases.
    # Population individual-seed RMSE=amplitude, so theta=bases_r-bases_c.
    # This truth is never changed to E of the finite-calendar sample root.
    return q, bases[1:]-bases[0], bases
