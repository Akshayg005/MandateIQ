"""eval/sim2.py -- end-to-end fit smoke test (R1b).

The gate (reports/gates.md, R1b) requires a simulator that ACTUALLY
generates issuer/instrument/age effects, not merely one that carries the
columns. Phase A's whole finding was that a widened design fit against a
DGP that doesn't vary by a covariate reports a clean, honest null (0/18
CIs excluding zero) -- the correct behaviour there. This test is the
mirror-image regression guard for Phase B: if eval/sim2.py's hazard bonuses
were ever accidentally zeroed out or diluted below detectability, a fit
here would ALSO go all-null, and nothing else would catch that (the DGP
tests in test_sim2.py check outcome aggregates directly, not the fitted
model's own inferential output)."""
from __future__ import annotations


def test_fit_sim2_model_detects_at_least_one_real_effect():
    """A fit on a real (if modest) sim2 corpus must find at least one of
    the 18 issuer/instrument/age coefficients significant at 95% -- proof
    the DGP's effects are strong enough for the design-matrix machinery to
    actually detect, not just carry as unused columns."""
    from eval.sim2 import (
        assembled_sim2_frame, fit_sim2_model, _coefficient_table, SIM2_SEEDS,
    )

    # A handful of seeds, not the full 40 -- issuer_gamma's dead-hazard
    # bonus is large (see eval/sim2.py's DECISIONS.md-cited derivation), so
    # this is expected to already clear significance well before the full
    # corpus size a real report run uses.
    df = assembled_sim2_frame(seeds=SIM2_SEEDS[:6])
    model = fit_sim2_model(df)
    coef_df = _coefficient_table(model)

    n_excludes_zero = int(coef_df["excludes_zero"].sum())
    assert n_excludes_zero >= 1, (
        "eval/sim2.py's DGP produced ZERO significant issuer/instrument/age "
        "coefficients on a 6-seed fit -- the whole point of a second "
        "simulator (unlike Phase A's honest null) is that these covariates "
        "carry real, detectable signal. Check the hazard bonus constants "
        "(_ISSUER_DEAD_BONUS_LOGIT / _INSTRUMENT_DEAD_BONUS_LOGIT / "
        "age_recovery_bonus_logit_per_year) haven't been diluted."
    )
    # The two effects deliberately built largest (see module docstring's
    # DECISIONS.md-cited margin calculation) should specifically survive:
    # issuer_gamma's DEAD coefficient and mandate_age_years' RECOVERED
    # coefficient.
    gamma_dead = coef_df[
        (coef_df["outcome"] == "DEAD") & (coef_df["column"] == "issuer_issuer_gamma")
    ]
    assert len(gamma_dead) == 1
    assert bool(gamma_dead["excludes_zero"].iloc[0]), (
        "issuer_gamma's DEAD coefficient (the single largest deliberately-"
        "built effect) did not clear 95% significance even on a 6-seed fit"
    )
