"""Verbatim functions from preserved validate_controlled.py; original source remains unchanged.
AI-assisted extraction; reference hashes are in LEGACY_REFERENCE.json.
"""
import numpy as np

def require(ok, message):
    if not ok: raise ValueError(message)


def bases_for(case):
    base=np.ones(case['M']);d=case['delta'];kind=case['theta']
    if kind=='equal':base[1:]=1.
    elif kind=='boundary':base[1:]=1+d
    elif kind=='interior':base[1:]=1+d-.05
    elif kind=='partial':base[1:10]=1+d;base[10:]=1+d+.20
    elif kind=='alternative':base[1:]=1+d+.20
    else:raise ValueError('Unknown declared truth scenario')
    return base


def unit_second_moment_AR(rng,phi,T,features,innovation,nu):
    """Unit variance at every step; t innovations use a 128-point burn.

    Gaussian initialization is exactly stationary for the Gaussian AR case.
    In the t5 case only the second moment is exactly stationary; the full
    stationary distribution is approximated by burn-in, explicitly not exact.
    """
    require(abs(phi)<1,'AR stationary second moment requires abs(phi)<1')
    state=rng.normal(size=features)
    out=np.empty((T,features));scale=np.sqrt(1-phi*phi)
    for t in range(T+128):
        if innovation=='gaussian':shock=rng.normal(size=features)
        elif innovation=='student_t':
            require(nu>4,'Declared heavy-tail fourth moment must exist')
            shock=rng.standard_t(nu,size=features)*np.sqrt((nu-2)/nu)
        else:raise ValueError('Unknown innovations')
        state=phi*state+scale*shock
        if t>=128:out[t-128]=state
    return out


def make_squared_errors(case,rng):
    S,T,M=case['R'],case['T'],case['M'];bases=bases_for(case)
    latent=np.sqrt(.5)*rng.normal(size=(S,1))+np.sqrt(.5)*rng.normal(size=(S,M))
    sigma=case['seed_sigma']
    amplitude=bases[None,:]*np.exp(sigma*latent-sigma*sigma/2)
    weights=np.asarray(case['weights'],dtype=float)
    require(weights.shape==(4,) and (weights>=0).all(),'Invalid factor weights')
    if weights.sum()==0:
        z=np.ones((S,T,M))
    else:
        require(abs(float(weights.sum())-1)<1e-12,'Factor variances must sum to one')
        values=unit_second_moment_AR(rng,case['phi'],T,1+M+S+S*M,
                                    case['innovations'],case.get('nu'))
        global_time=values[:,0]
        model_time=values[:,1:1+M]
        seed_time=values[:,1+M:1+M+S]
        idio=values[:,1+M+S:].reshape(T,S,M).transpose(1,0,2)
        z=(np.sqrt(weights[0])*global_time[None,:,None]
           +np.sqrt(weights[1])*model_time[None,:,:]
           +np.sqrt(weights[2])*seed_time.T[:,:,None]
           +np.sqrt(weights[3])*idio)
    q=(amplitude[:,None,:]*z)**2
    return q,bases[1:]-bases[0],bases

