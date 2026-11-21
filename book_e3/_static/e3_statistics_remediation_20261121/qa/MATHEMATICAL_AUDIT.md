# Independent review of the earlier crossed RMSE methods

AI assistance declared. This review reads the frozen code and original saved Monte Carlo results. It executes no new model, prediction, or controlled random replicate. New algorithms and validations must remain separately versioned.

## What is established

The original equal predictive ability case has **12 seeds, 128 dates, and 20 systems**. The range statistic considers 190 unordered model pairs. The saved records independently reproduce **27/128 initial global rejections** at nominal 10%. The failure is retained. The full raw log hash is `a687d180084b906631a625db371c137754e3f5f66180fabd3e758ab3328b1b7b`.

The original code uses `theta = mean_s(sqrt(mean_t q_rival)) - mean_s(sqrt(mean_t q_candidate))`. Its scientific target is the expectation over initialization of each model's time-population RMSE. Pooling squared errors over seeds before taking the root would change that target. Averaging point predictions first would change it again.

## Equal-risk truth is not invalidated by the square-root bias

In the controlled DGP, `q_smt = A_sm^2 Z_smt^2`, with seed amplitudes independent of the temporal process, `E(A_sm)=base_m`, and each model having the same marginal distribution of `Z`. At a finite calendar length T,

`E(mean_s sqrt(mean_t q_smt)) = base_m * c_T`,

where `c_T = E(sqrt(mean_t Z_t^2)) <= 1` by Jensen. Under global EPA, all bases equal one, so every finite-calendar expected metric equals the same `c_T`. Therefore finite-T Jensen bias does **not** make the old equal-risk null false and does not excuse its 27 rejections.

For a population margin boundary `base_rival-base_candidate=delta>0`, the finite-calendar mean difference has expectation `c_T*delta`, below the long-run population boundary. This creates a finite-calendar/population target distinction; it generally moves this particular boundary inward rather than explaining anti-conservatism under the equal-risk null. Neither argument proves bootstrap validity.

## A concrete dependence mechanism

For a stationary unit Gaussian AR(1), `Cov(Z_0^2,Z_k^2)=2*phi^(2k)`. Thus the temporal long-run variance divided by its marginal variance equals `(1+phi^2)/(1-phi^2)`. The old lag-eight Bartlett kernel weights and mean-block-twelve stationary bootstrap kernel omit or downweight positive dependence.

At `phi=.85`, the full population factor is **6.2072**. The Bartlett lag-eight factor is **4.2341**, a ratio of **0.6821**. The stationary mean-block-twelve factor is **4.9223**, a ratio of **0.7930**. These calculations concern the temporal component before finite-T demeaning and before adding seed and interaction terms. They are not exact total variance ratios for the nonlinear crossed estimator.

The independent reanalysis of the old strong-dependence results finds ratios of mean P2 estimated variance to across-replicate empirical variance ranging **0.576–0.945** over the 19 rivals. Thus adding seed and temporal components is not an empirical finite-sample upper variance bound. The old mathematical motivation for double-counting interaction assumes an appropriate underlying long-run variance; estimated finite-lag HAC can still understate it.

At the equal-risk case's `phi=.35`, the kernel ratios are about **0.9724** and **0.9795**. These small differences alone are not a complete explanation of its global rejection inflation. A small seed cohort, many studentized pairs, and finite-sample tails require further investigation.

## No unjustified claim of a coding bug

The inspected direction, maximum-range statistic, and worst-model elimination sign are internally consistent with loss minimization. Using the same pairwise estimated standard deviation for observed and centered bootstrap statistics is also part of the conventional MCS construction. It is not itself proof of an implementation error. Here the **crossed full RMSE functional**, finite seed cohort, and resampling design are proposed extensions of that construction.

A new bootstrap-t version that estimates its scale within each resample is a defensible development proposal. It is a changed method, and its computational and statistical validation must use its own source freeze, plan, seeds, and acceptance criteria. Failure to detect inflation in a small validation is not calibration certification. Structural zero scales need explicit conservative behavior; silently dropping unfavorable or degenerate draws is not acceptable.

## The third-method gap remains real

The original F2 row permits MCS, DM-HLN, Giacomini–White or Clark–West according to nesting, Friedman across series or origins, and stationary bootstrap. This does not make all of them interchangeable tests of the same target:

- MCS is a global equal-risk/model-set procedure. Exclusion does not automatically give a directional practical-margin p-value against the nominated candidate.
- Clark–West requires an appropriate nested forecast comparison, not merely two network architecture names.
- Giacomini–White tests conditional predictive ability with a specified information/instrument set and forecasting scheme; replacing unconditional mean individual RMSE by a different additive loss changes the hypothesis.
- Friedman requires justified series/origin blocks. Seeds are not automatically independent series, and overlapping origins or dependent markets do not become independent by being listed separately.

The guide's three tests target equal performance, with direction, practical relevance and interval conditions separately required. The previous project choice of three practical-margin nulls would be stronger than that literal requirement. Neither formulation permits inventing a third procedure by changing names, repeating the same p-value, or silently changing estimands.

## Retrospective adequacy limits

The fixed V4 diagnostic plan properly separates 88 auxiliary p-value slots from its primary comparison and from 44 pointwise HAC mean-residual intervals. It conditions on saved trajectories and does not pretend that five seeds meet the original ten-seed primary guard.

Overlapping volatility targets motivate dependence analysis, but an optimal **one-step conditional-mean** forecast still has martingale-difference errors. Target overlap is not an automatic explanation for rejected residual whiteness. Conversely, ARCH effects alone do not establish invalidity of a point conditional-mean forecast. The Ljung–Box classical reference is not automatically robust to heteroskedasticity, and Holm cannot repair invalid marginal p-values. These are exploratory adequacy diagnostics, not a simultaneous adequacy or R3 certificate.

## Reproducible review outputs

- `reaudit_saved_controlled_run.py` and `OLD_MC_REAUDIT.json`: deterministic reading of all 1,024 already saved records.
- `analytical_dependence_audit.py` and its JSON output: deterministic kernel calculations and exact seed bootstrap variance identity.
- `RETROSPECTIVE_PLAN_REVIEW.json`: initial plan review before outcomes, with implementation and public anchor checks separately pending.

## Result of the separately anchored new validation

The later version fixed code and its validation plan in public commit `7d304faa0dd583c6109a688ff179ec7e937f6561` before execution. It completed **640 fresh replicates and 639,360 crossed bootstrap draws in 187.684 seconds**. The legacy comparison used the same draws; it did not add another bootstrap experiment. Source files stayed unchanged. The reviewer checked raw records and independently inverted binomial probabilities for the reported Clopper–Pearson intervals; no fresh simulation was rerun for review.

The new global MCS bootstrap-t variant rejects in **21/256**, **4/128**, and **2/128** of the three equal-risk cases. All three pass the previously declared individual tolerance screen requiring the upper central 95% CP endpoint to be at most 0.13 for nominal 0.10. This does not certify exact level 0.10, a simultaneous guarantee across settings, or validity for a future market family.

The **two-method Holm38 primary subset is still not approved**. Its seed-only null family error is **14/128 = 10.9375%**, CP95 **6.1101–17.6701%**, with the lower endpoint above nominal 5%. Its strong-dependence case gives **9/128**, an inconclusive interval **3.2654–12.9279%**. The family is not reduced after this result. Decomposing rejections within that same fixed family identifies P1 as present in all 14 seed-only false-rejection events, while P2 is present in one. This decomposition does not authorize a new selectively reduced family.

In the synthetic alternative, global EPA rejection occurs in **112/128** replicates, but a unique-candidate confidence set occurs only in **39/128**. The two-method all-rival p-value rejection intersection occurs in **18/128**; it does not include every practical-margin, effect-size and adequacy condition of the complete scientific decision rule. None is a certificate of the final three-method power requirement.

The new retrospective execution also preserves **13 Mincer–Zarnowitz diagnostic rejections** in its fixed Holm88 family, with 44 pointwise conditional mean-error intervals and no invented predictive interval calibration. No nonrejection is upgraded to an adequacy pass. The original V3 NO-GO and V4 abandonment remain intact.

See `CALIBRATION_PREVALIDATION_REVIEW.json`, `CALIBRATION_POSTRUN_REVIEW.json`, `CALIBRATION_EXPORT_REVIEW.json`, `RETROSPECTIVE_IMPLEMENTATION_REVIEW.json`, and `RETROSPECTIVE_POSTRUN_REVIEW.json` for hashes, dates, counts and scope. `HOLM_RESOLUTION_AUDIT.json` additionally records why a first-rank bootstrap p-value resolution bound must not be misapplied to a mixed analytical/bootstrap Holm family.
