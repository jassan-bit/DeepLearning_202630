# Primary mean-RMSE inference, new version 1

The earlier S2 pooled-MSE implementation remains unchanged. Its reference
arithmetic is identifiable as `engine/statistics.py` from commit
`c54a2fd3123a1ea8f98c2e56c2acfbe491c8672c`, SHA256
`a14e20b74f668404835f0f15e1fe555c5ac718ab886125b2dea20d531df06e95`.
This new implementation is not a renaming of its MSE tests. Source attribution,
AI assistance and the pre-freeze nuisance-normalization error/correction are
recorded explicitly. No forecasting model or earlier pilot rule is rewritten.

## 1. Exact inherited primary functional and hypotheses

For fixed fitted model/seed indices j,s and chronological dates t, let
`l[j,s,t] = (prediction[j,s,t]-y[t])²`, `mu[j,s] = E_t l[j,s,t]`, and

```
rho[j,s] = sqrt(mu[j,s])
theta[j] = mean_s rho[j,s].
```

The empirical statistic reestimates each seed risk over dates, takes each root,
then averages roots. It is not `sqrt(mean_s,t l)` and not an ensemble prediction
RMSE. The primary EPA contrast is `D[j]=theta[j]-theta[0]`. With the retained
relative margin m=.02, the practical contrast is
`W[j]=(1-m)*theta[j]-theta[0]`, not `(1-m)²` times a root quantity.

The tests have two-sided boundary nulls D=0 and W=0. A positive superiority
claim additionally requires positive directions, all tests, the prespecified
effect threshold and favorable intervals. Boundary calibration does not prove
arbitrary finite-sample validity over composite one-sided nulls; separately
declared adverse/boundary DGPs check the complete decision.

## 2. Assumptions and influence derivation

R is fixed. The finite collection of loss and predictable-instrument processes
is weakly stationary, has sufficient joint moments and satisfies a joint
functional CLT and relevant bootstrap consistency/mixing conditions. All seed
risks are strictly positive and bounded away from zero for this smooth
derivation; a perfect zero-risk seed is returned as unavailable, not regularized
with an invented epsilon. No R×T iid assumption or unseen training-population
uncertainty is introduced.

The derivative of a seed root is `1/(2*rho[j,s])`. Therefore

```
psi[j,t] = mean_s (l[j,s,t]-mu[j,s])/(2*rho[j,s])
sqrt(T)*(theta_hat[j]-theta[j])
    = T^(-1/2) sum_t psi[j,t] + o_p(1).
```

In the empirical version psi has temporal mean zero up to floating arithmetic.
The EPA influence is psi[j]-psi[0]; the practical influence is
`(1-m)*psi[j]-psi[0]`. This is a new smooth-functional extension of temporal
loss comparison, not an assertion that ordinary additive-MSE DM already tests
the inherited mean-root functional.

## 3. Three distinct procedures

### Scalar delta/HAC-HLN test

The scalar statistic divides the actual D_hat or W_hat by the HAC standard
error of its delta influence. Bartlett HAC is calculated with zero-padded
moving sums, including boundary windows; it is algebraically the Newey-West
lag-weighted covariance with divisor T. The HLN factor is retained and used
consistently in the scalar reference. With fixed-b lag ratio the reference is
a frozen Gaussian scalar FCLT pivot, not a claim of an exact Student law for a
nonlinear statistic. Finite behavior requires the separately frozen validation.

### Exact-functional stationary bootstrap-t

A common circular stationary time-index array resamples **all models and fixed
seeds together**. In every replicate, the implementation recomputes every seed
MSE, every root, the root average, the influence coefficients and its HAC.
The centered studentized replicate uses the difference of the resampled and
observed **root-average** contrast. It never bootstraps a pooled-MSE shortcut.
Degenerate studentizers count as infinite absolute tail values; they are not
discarded to improve a p-value or effective B.

The Monte Carlo p-value is `(1+tail_count)/(B+1)`. The fixed computational seed
and index hash are reported. Exact memoization reuses the same readonly PCG
index bytes for the same `(T,B,block_length,seed)`; it changes no output. If the
validation plan fixes inner seeds while varying outer DGP samples, its scope
is data variation conditional on that frozen computational algorithm, not a
claim integrating fresh inner randomness. This choice precedes formal MC.

### Nuisance-adjusted conditional functional GMM

Define `h[j,t]=theta[j]+psi[j,t]`. Its expectation is theta[j]. For predictable
instruments z[t], with a constant and a nonconstant column, define

```
g = E[z[t] * (a*h[j,t]-h[0,t])],  a=1 or 1-m.
```

The constant moment is exactly the EPA/practical primary contrast. The vector
conditional-functional null is **stronger** than that scalar null. For example,
the scalar may cancel while instrumented moments are nonzero; a conditional
rejection is not itself a directional superiority finding. The scalar tests
remain mandatory in the conjunction and protect the scalar primary claim.

The seed risks mu are estimated on the same time series, so treating h as a
fixed observed loss would omit nuisance uncertainty. The required derivative is

```
D[j,q,s] = -E[z[q,t]*(l[j,s,t]-mu[j,s])]
                /(4*R*mu[j,s]^(3/2)).
IF_g[t] = z[t]*(a*h[j,t]-h[0,t])-g
              + a*D[j] @ (l[j,:,t]-mu[j,:])
              - D[0] @ (l[0,:,t]-mu[0,:]).
```

The derivative uses a **temporal mean**, including division by T. Independent
weighted-contamination finite differences verify this expression. For the
constant instrument, the nuisance derivative is zero and IF_g's first column
equals the scalar delta influence. For nonconstant instruments it generally
does not vanish. The Wald statistic uses HAC of the complete IF_g and the
frozen Gaussian vector fixed-b reference; a singular covariance is unavailable,
not silently changed to a lower-dimensional test.

## 4. Effects and intervals: an explicit prospective semantic amendment

The retained .2 threshold now applies to
`D_hat / SD_t(psi[j,t]-psi[0,t])`. This is the standardized temporal
smooth-functional pseudo-loss contrast. It is not the previous standardized
MSE difference and is not a training-population Cohen effect. Keeping the
numeric threshold does **not** mean all statistical functionals stayed byte
identical; the endpoint-alignment change and its scope are versioned here.

The primary difference intervals are bootstrap-t intervals for D. Relative
gain is `1-theta[0]/theta[j]`; the log ratio is `log(theta[0]/theta[j])` and its
delta influence is `psi[0]/theta[0]-psi[j]/theta[j]`. Back-transformation is
`1-exp(log_ratio)`, **without the old pooled-MSE factor 1/2**. IC95 and IC99
remain marginal temporal intervals, not simultaneous family confidence bands.

## 5. Family and joint policy retained, not tightened to all three

For each panel, 20 rivals × 3 procedures × 2 nulls gives Holm120 at alpha=.025.
The main-all and at-least-two-of-three-additional policy is retained, as are
positive effects/directions and favorable IC95/99, and the additional adverse
EPA Holm180 screen. GMM alone is not used to establish directional inferiority.

The strict decision wrapper checks ordered 21-system identity, 120 slots,
three exact procedure IDs/two nulls, version, primary interval type, common
settings/margin, and the declared temporal/fixed-seed sizes. Incomplete test
fixtures require an explicit `require_full_family=False` and do not constitute
a complete-family validation.

If a population primary claim is false, acceptance requires the corresponding
false primary-panel claim. If a rival has true practical superiority in at most
one additional panel, obtaining two positives requires a false claim in at
least one of two fixed null panels. Under valid marginal false-claim bounds
q<=.025, a union bound gives at most .05 without panel independence. The
additional adverse screen can only reduce acceptance. **This logical statement
does not import the earlier q<=.05 marginal CP bounds**; the new frozen study
measures the complete four-panel decision directly, including conditional
cancellation, partial/practical nulls, dependence and adverse conditions.

## 6. Exact computational compression and restricted DGP scope

Only exactly equal loss traces are reused. Duplicated rival channels are
expanded before all statistical slots and Holm; no rival is removed. The
candidate remains a separate channel even if an identical rival exists.
Joint seed columns `[models,time]` are reduced only if every exact equivalence
class has the same multiplicity. Uniform root averaging is then algebraically
unchanged. Unequal multiplicities retain all original seeds; no approximate
weights or near-equality tolerance defines classes. Hash matches are checked
with exact array equality. Floating arithmetic is checked against the
uncompressed implementation and independent reference within stated tolerances.

Metrics, influence/GMM and the reported fixed seed count retain the original
cohort. A validation using repeated rival rules and perfectly dependent seed
groups therefore has a **restricted DGP scope**, not evidence for 20 independent
learners or ten independent training draws. Heterogeneous seed-risk cases must
still exercise the mean-root versus pooled-root distinction. Throughput
benchmarks inspect timing/shape only and use disjoint benchmark seeds; they are
not calibration outcomes or criteria-selection evidence.

## 7. What this can and cannot close

Correct implementation plus successful prespecified finite validation can
support primary-endpoint alignment and joint-decision validation **within the
declared scenarios/settings**. Neither a proof under assumptions, passing
software tests nor synthetic calibration demonstrates market superiority,
real-data power, unseen reserves, all adequacy/robustness conditions or R3.
No original adverse pilot, calibration attempt, alpha, candidate, source or
historical conclusion is replaced.
