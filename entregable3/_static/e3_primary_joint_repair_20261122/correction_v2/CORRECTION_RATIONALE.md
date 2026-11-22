# V2 operational interval correction

This correction was designed after the frozen V1 study completed. It is not
an additional V1 replicate or an attempt to change its acceptance criteria.
The parent preserved the complete V1 sources, independent review, outcome
and export in `versions/mean_rmse_v1` before authorizing this patch.

V1's sole formal acceptance failure was the main practical-boundary nominal99
coverage:501/512 for each operational interval, simultaneous CP107 lower
0.948366<0.95. Nominal95 undercoverage was also retained as a diagnostic.
The correction replaces operational bootstrap-t percentile inversion with
the direct inversion of the already implemented scalar fixed-b MC+1 test.
It preserves both nominal95/99 levels, all three p-value procedures, both
nulls, all120/180 Holm slots, alpha.025, practical margin.02, effect threshold
.2 and the original four-panel main-all plus two-of-three policy.

`mc_plus_one_critical(reference, level)` returns a metadata dictionary. For
M reference draws and exact rational alpha=1-level, its zero-based cutoff
index is M-ceil(alpha*(M+1))+1. This is the largest closed non-rejection
boundary for p=(1+#reference>=abs(T))/(M+1) and strict p<alpha. If no rejection
is possible at that resolution, critical_value and finite CI are None.
`fixed_b_inverted_ci(point,se,hln,critical)` receives that dictionary and
returns point +/- critical_value*se/hln, or None when not regular/finite.

The difference CI uses the same EPA scalar HAC/HLN studentizer as its test.
The gain CI inverts the log theta ratio with its own delta influence and
HAC studentizer and maps `[L,U]` to `[1-exp(U),1-exp(L)]`. No pooled-root
factor1/2, confidence-level inflation, chosen hull or fallback is used.
This is exact inversion of the finite reference-count decision, under the
same asymptotic FCLT approximation, not an exact finite-sample market claim.

The four V1 bootstrap-t intervals remain unchanged as diagnostic output.
Strict production admission rejects the old interval method as operational.
Unit checks include exact nominal ranks, ties, just-outside p-values,
insufficient resolution, HLN division and equality of all V1/V2 p-values and
old diagnostic interval values. An independent reviewer checks the patch
before a separate V2 protocol can be frozen and published.

Version: `temporal-fixed-seeds-mean-rmse-v2`.
Operational interval method: `scalar_fixed_b_mc_plus_one_inversion_v2`.
No market data, model fitting or reserved test was used. AI implementation
and independent AI mathematical/software review are explicitly declared.
