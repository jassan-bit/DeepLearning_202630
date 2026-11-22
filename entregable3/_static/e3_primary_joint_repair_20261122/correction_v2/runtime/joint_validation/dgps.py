"""Predeclared synthetic risk DGPs; exact population mean-seed RMSE truth.

The observed unit is the date. Seed risks describe a fixed fitted cohort;
no seed-times-date pseudo-replicates are generated or treated as independent.
"""
from __future__ import annotations

import numpy as np

SYSTEM_IDS=('input_only_mse','FCN-1D','ResNet-1D','Inception-1D','LSTM-autoencoder con predictor',
    'TimesNet','TSMixer','TiDE','TimeMixer','PatchTST','iTransformer','MLP16','LSTM32','GRU32',
    'Persistence','SeasonalNaive','ARIMA','additive_positive_head','floor_only',
    'floor_plus_mean_excess','global_mse')
DETERMINISTIC={'Persistence','SeasonalNaive','ARIMA','floor_only','floor_plus_mean_excess'}

SCENARIOS=(
    dict(id='complete_epa_iid',kind='null',phi=0.,panel_coupling='independent',noise=.55,truth_pattern='complete_epa'),
    dict(id='complete_epa_ar_shared',kind='null',phi=.5,panel_coupling='equicorrelation_0.8',noise=.55,truth_pattern='complete_epa'),
    dict(id='main_single_practical_boundary',kind='null',phi=.5,panel_coupling='equicorrelation_0.8',noise=.05,truth_pattern='main_boundary'),
    dict(id='two_null_additional_boundaries',kind='null',phi=.5,panel_coupling='independent',noise=.05,truth_pattern='additional_boundary'),
    dict(id='two_null_additional_negative_coupling',kind='null',phi=.85,panel_coupling='additional_pair_minus_0.8',noise=.55,truth_pattern='additional_boundary'),
    dict(id='conditional_cancellation',kind='null',phi=.5,panel_coupling='equicorrelation_0.8',noise=.55,truth_pattern='main_epa',conditional_cancellation=True),
    dict(id='variance_break_adverse',kind='null',phi=.5,panel_coupling='equicorrelation_0.8',noise=.55,truth_pattern='additional_boundary',variance_break=True),
    dict(id='positive_risk_ranking_reversal',kind='null',phi=.5,panel_coupling='equicorrelation_0.8',noise=.55,truth_pattern='ranking_reversal'),
    dict(id='strong_positive_control',kind='alternative',phi=.5,panel_coupling='equicorrelation_0.8',noise=.55,truth_pattern='all_positive'),
)


def panel_covariance(mode):
    if mode=='independent':
        return np.eye(4)
    if mode=='equicorrelation_0.8':
        return .8*np.ones((4,4))+.2*np.eye(4)
    if mode=='additional_pair_minus_0.8':
        result=np.eye(4);result[2,3]=result[3,2]=-.8
        return result
    raise ValueError('Unplanned panel coupling')


def true_risks(scenario,R,margin=.02,strong_gain=.25):
    if R<2 or R%2:
        raise ValueError('The declared seed cohort requires an even R>=2')
    sigma=np.ones(R)
    if scenario.get('conditional_cancellation'):
        sigma=np.r_[np.full(R//2,.5),np.full(R//2,1.5)]
    if scenario['truth_pattern']=='ranking_reversal':
        sigma=np.full(R,1.1)
    mu=np.empty((4,len(SYSTEM_IDS),R))
    mu[:,0]=sigma*sigma
    gains=np.full((4,len(SYSTEM_IDS)-1),strong_gain)
    pattern=scenario['truth_pattern']
    if pattern=='complete_epa':gains[:]=0.
    elif pattern=='main_boundary':gains[0,0]=margin
    elif pattern=='additional_boundary':gains[2:,0]=margin
    elif pattern=='main_epa':gains[0,0]=0.
    elif pattern not in ('ranking_reversal','all_positive'):
        raise ValueError('Unknown fixed truth pattern')
    for panel in range(4):
        for j,name in enumerate(SYSTEM_IDS[1:],1):
            local=sigma/(1-gains[panel,j-1])
            if name in DETERMINISTIC:
                local=np.full(R,sigma.mean()/(1-gains[panel,j-1]))
            mu[panel,j]=local*local
    if pattern=='ranking_reversal':
        mu[0,1]=np.r_[np.full(R//2,.1**2),np.full(R//2,2.**2)]
    theta=np.sqrt(mu).mean(axis=2)
    gains=(theta[:,1:]-theta[:,0,None])/theta[:,1:]
    # Equality at the practical boundary is a true null; avoid roundoff changing truth.
    positive=gains > margin+1e-12
    return mu,theta,gains,positive


def stationary_ar(rng,shape,phi,covariance=None):
    """Return exactly stationary variance-one Gaussian factors including t=-1."""
    values=rng.standard_normal(shape)
    if covariance is not None:
        values=np.linalg.cholesky(covariance)@values
    innovation=np.sqrt(1-phi*phi)
    for t in range(1,shape[-1]):
        values[...,t]=phi*values[...,t-1]+innovation*values[...,t]
    return values


def generate(scenario,R,T,seed,margin=.02):
    """Positive squared errors with E(loss_m,r,t)=mu_m,r, known analytically.

    exp(s*Z-s²/2) has mean one for Z~N(0,1). Thus theta_m is
    exactly mean_r(sqrt(mu_m,r)), including the variance-break stress.
    A counterfactual signed forecast error could be sqrt(loss)*Rademacher;
    the statistical analyzer only needs these squared errors.
    """
    if T<16:raise ValueError('Calendar too short for this declared DGP')
    rng=np.random.default_rng(seed)
    mu,theta,gains,positive=true_risks(scenario,R,margin)
    phi=scenario['phi']
    factors=stationary_ar(rng,(4,T+1),phi,panel_covariance(scenario['panel_coupling']))
    # Explicit restricted DGP scope, chosen for exact computational compression:
    # 20 rival labels remain in Holm120, but there are only a witness rule and
    # one repeated non-witness rule. Fitted seeds are perfectly dependent in
    # homogeneous cases; heterogeneous cases contain two fixed seed groups.
    # This is not a claim of 20 independent algorithms or ten independent fits.
    groups=np.asarray([0,1]+[2]*(len(SYSTEM_IDS)-2))
    heterogeneous=bool(scenario.get('conditional_cancellation') or
                       scenario['truth_pattern']=='ranking_reversal')
    seed_groups=np.repeat(np.arange(2),R//2) if heterogeneous else np.zeros(R,dtype=int)
    model=stationary_ar(rng,(4,3,T+1),phi)
    individual=stationary_ar(rng,(4,3,2 if heterogeneous else 1,T+1),phi)
    common=np.broadcast_to(factors[:,None,None,:],individual.shape).copy()
    if scenario.get('conditional_cancellation'):
        common[0,1]*=-1
    latent=np.sqrt(.6)*common+np.sqrt(.3)*model[:,:,None,:]+np.sqrt(.1)*individual
    scale=np.full(T,scenario['noise'])
    if scenario.get('variance_break'):
        scale[:T//2]=.25;scale[T//2:]=.9
    unit=np.exp(latent[...,1:]*scale[None,None,None,:]-.5*scale[None,None,None,:]**2)
    expanded=unit[:,groups][:,:,seed_groups]
    losses=mu[:,:,:,None]*expanded
    for j,name in enumerate(SYSTEM_IDS):
        if name in DETERMINISTIC:
            losses[:,j]=losses[:,j,:1]
    instruments=np.stack((np.ones((4,T)),factors[:,:T]),axis=-1)
    if not np.isfinite(losses).all() or (losses<=0).any():
        raise ValueError('Invalid generated squared errors')
    return dict(losses=losses,instruments=instruments,theta=theta,gains=gains,
        true_superiority=positive,mu=mu,
        witness_true_rmse_difference=float(theta[0,1]-theta[0,0]),
        witness_true_relative_gain=float(gains[0,0]))
