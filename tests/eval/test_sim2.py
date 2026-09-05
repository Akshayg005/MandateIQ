"""eval/sim2.py -- non-frozen, covariate-varying simulator for Phase B model defensibility.

This simulator generates mandates with varying issuer_id, instrument_type, and
mandate_age_days covariates that the frozen simulator never generates. The
data-generating process produces measured differences in cause-specific hazards
by these covariates so a competing-risks model can measure defensible per-covariate
coefficients (see reports/model_defensibility.md's Phase B section).

Key invariants tested here:
1. issuer_gamma has a materially elevated dead-hazard vs the reference issuer
2. upi_autopay has a materially elevated dead-hazard vs debit/credit instruments
3. generate_corpus() episodes pass all legal checks (increasing on_day, etc.)
4. build_sim2_features() produces the right shape and constant-per-mandate values
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestSim2Mandate:
    """Sim2Mandate is a frozen dataclass with issuer/instrument/age covariates."""

    def test_sim2_mandate_has_required_fields(self):
        """Import Sim2Mandate and verify it has the required fields."""
        from eval.sim2 import Sim2Mandate
        from src.core.types import Cause

        m = Sim2Mandate(
            mandate_id="s1:M001",
            cycle_id=1,
            amount_paise=100_000,
            ceiling_paise=200_000,
            category="subscription",
            household_id="HH001",
            initial_cause=Cause.CANT_PAY_NOW,
            issuer_id="issuer_alpha",
            instrument_type="upi_autopay",
            mandate_age_days=180,
        )
        assert m.mandate_id == "s1:M001"
        assert m.cycle_id == 1
        assert m.amount_paise == 100_000
        assert m.ceiling_paise == 200_000
        assert m.category == "subscription"
        assert m.household_id == "HH001"
        assert m.initial_cause == Cause.CANT_PAY_NOW
        assert m.issuer_id == "issuer_alpha"
        assert m.instrument_type == "upi_autopay"
        assert m.mandate_age_days == 180

    def test_sim2_mandate_is_frozen(self):
        """Sim2Mandate must be immutable (frozen=True)."""
        from eval.sim2 import Sim2Mandate
        from src.core.types import Cause

        m = Sim2Mandate(
            mandate_id="s1:M001",
            cycle_id=1,
            amount_paise=100_000,
            ceiling_paise=200_000,
            category="subscription",
            household_id="HH001",
            initial_cause=Cause.CANT_PAY_NOW,
            issuer_id="issuer_alpha",
            instrument_type="upi_autopay",
            mandate_age_days=180,
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            m.issuer_id = "issuer_beta"


class TestSimulator2:
    """Simulator2 class: deterministic mandate generation and attempt simulation."""

    def test_simulator2_init_generates_mandates_deterministically(self):
        """Simulator2(seed=X) must generate identical mandates for identical seed."""
        from eval.sim2 import Simulator2

        sim1 = Simulator2(seed=1000)
        sim2 = Simulator2(seed=1000)

        assert len(sim1.mandates) == len(sim2.mandates)
        for m1, m2 in zip(sim1.mandates, sim2.mandates):
            assert m1.mandate_id == m2.mandate_id
            assert m1.amount_paise == m2.amount_paise
            assert m1.initial_cause == m2.initial_cause
            assert m1.issuer_id == m2.issuer_id
            assert m1.instrument_type == m2.instrument_type
            assert m1.mandate_age_days == m2.mandate_age_days

    def test_simulator2_different_seeds_produce_different_mandates(self):
        """Simulator2(seed=X) must produce different mandates for different seeds."""
        from eval.sim2 import Simulator2

        sim1 = Simulator2(seed=1000)
        sim2 = Simulator2(seed=2000)

        # At least some mandates should differ
        differences = sum(
            1 for m1, m2 in zip(sim1.mandates, sim2.mandates)
            if m1.mandate_id != m2.mandate_id or m1.issuer_id != m2.issuer_id
        )
        assert differences > 0, "Different seeds should produce different mandates"

    def test_simulator2_mandates_have_required_fields(self):
        """Every mandate from Simulator2 must have issuer_id/instrument_type/age."""
        from eval.sim2 import Simulator2

        sim = Simulator2(seed=1000)
        assert len(sim.mandates) > 0

        for mandate in sim.mandates:
            assert mandate.mandate_id is not None
            assert mandate.cycle_id == 1
            assert mandate.amount_paise > 0
            assert mandate.ceiling_paise >= mandate.amount_paise
            assert mandate.category in (
                "subscription",
                "insurance_premium",
                "mutual_fund",
                "credit_card_bill",
            )
            assert mandate.initial_cause is not None
            assert mandate.issuer_id is not None
            assert mandate.instrument_type is not None
            assert mandate.mandate_age_days >= 0

    def test_simulator2_attempt_returns_attempt_result(self):
        """Simulator2.attempt() must return AttemptResult with correct fields."""
        from eval.sim2 import Simulator2, AttemptResult

        sim = Simulator2(seed=1000)
        mandate = sim.mandates[0]

        result = sim.attempt(mandate.mandate_id, slot=2, on_day=1)

        assert isinstance(result, AttemptResult)
        assert result.mandate_id == mandate.mandate_id
        assert result.slot == 2
        assert result.on_day == 1
        assert result.outcome is not None
        assert hasattr(result, "iatrogenic_insufficient_funds")

    def test_simulator2_attempt_validates_slot_order(self):
        """Simulator2.attempt() must raise on out-of-order slot attempts."""
        from eval.sim2 import Simulator2

        sim = Simulator2(seed=1000)
        mandate = sim.mandates[0]

        sim.attempt(mandate.mandate_id, slot=2, on_day=1)

        # Attempting slot 4 before slot 3 should raise
        with pytest.raises(ValueError, match="out of order|expected slot"):
            sim.attempt(mandate.mandate_id, slot=4, on_day=2)

    def test_simulator2_attempt_validates_increasing_days(self):
        """Simulator2.attempt() must raise if on_day is not strictly increasing."""
        from eval.sim2 import Simulator2

        sim = Simulator2(seed=1000)
        mandate = sim.mandates[0]

        sim.attempt(mandate.mandate_id, slot=2, on_day=5)

        # Attempting on a day <= 5 should raise
        with pytest.raises(ValueError, match="not after|is not strictly|is not increasing"):
            sim.attempt(mandate.mandate_id, slot=3, on_day=5)


class TestIssuerAndInstrumentLevels:
    """Test that ISSUER_LEVELS and INSTRUMENT_LEVELS are defined correctly."""

    def test_issuer_levels_defined(self):
        """ISSUER_LEVELS must be a tuple of 4 issuer names."""
        from eval.sim2 import ISSUER_LEVELS

        assert isinstance(ISSUER_LEVELS, tuple)
        assert len(ISSUER_LEVELS) == 4
        assert all(isinstance(x, str) for x in ISSUER_LEVELS)

    def test_instrument_levels_defined(self):
        """INSTRUMENT_LEVELS must be ('upi_autopay', 'debit_card', 'credit_card')."""
        from eval.sim2 import INSTRUMENT_LEVELS

        assert isinstance(INSTRUMENT_LEVELS, tuple)
        assert len(INSTRUMENT_LEVELS) == 3
        assert INSTRUMENT_LEVELS[0] == "upi_autopay"
        assert "debit_card" in INSTRUMENT_LEVELS
        assert "credit_card" in INSTRUMENT_LEVELS


class TestIssuerDeadHazardDifference:
    """Issuer hazard variation: issuer_gamma must have higher dead-hazard than reference."""

    def test_issuer_gamma_dead_rate_exceeds_reference_issuer(self):
        """Over many attempts, issuer_gamma must show materially higher DEAD rate.

        Aggregates outcomes across 150 seeds (~30,000 mandate-attempts),
        stratifying by issuer, and requires an absolute difference >= 5pp.

        SEED COUNT, DERIVED NOT GUESSED (DECISIONS.md, 2026-09-04, R1b review
        pass): the true aggregate gap is ~6.7-7.8pp (analytic marginalisation
        over the DGP's own cause_mix/age distribution, cross-checked by direct
        simulation). The statistics review found the ORIGINAL 20-seed window
        (seeds 1000-1019) measured only +5.51pp -- just 0.26 SD above the 5pp
        floor, an empirically confirmed ~7% flake rate across 30 disjoint
        20-seed windows. 150 seeds pushes the standard error low enough (from
        an empirical window-SD of ~1.93pp at 20 seeds, scaling by
        sqrt(20/150)) to put the measured gap ~2.5+ SE above the floor --
        confirmed directly: this seed range measures +7.83pp, not merely
        estimated to."""
        from eval.sim2 import Simulator2, ISSUER_LEVELS
        from src.core.types import Outcome

        # Identify the reference (first) and gamma issuers
        reference_issuer = ISSUER_LEVELS[0]
        gamma_issuer = "issuer_gamma"
        assert gamma_issuer in ISSUER_LEVELS, (
            f"issuer_gamma not found in ISSUER_LEVELS {ISSUER_LEVELS}"
        )

        # Generate aggregates: track (dead_count, total_count) per issuer
        issuer_outcomes = {}

        for seed in range(1000, 1150):  # 150 seeds -- see docstring for why
            sim = Simulator2(seed=seed)

            for mandate in sim.mandates:
                issuer = mandate.issuer_id
                if issuer not in issuer_outcomes:
                    issuer_outcomes[issuer] = {"dead": 0, "total": 0}

                # Attempt slot 2 only (deterministic outcome relative to cause/issuer/instrument)
                result = sim.attempt(mandate.mandate_id, slot=2, on_day=1)
                issuer_outcomes[issuer]["total"] += 1
                if result.outcome == Outcome.DEAD:
                    issuer_outcomes[issuer]["dead"] += 1

        # Compute dead rates for reference and gamma
        ref_dead_rate = (
            issuer_outcomes[reference_issuer]["dead"]
            / issuer_outcomes[reference_issuer]["total"]
        )
        gamma_dead_rate = (
            issuer_outcomes[gamma_issuer]["dead"] / issuer_outcomes[gamma_issuer]["total"]
        )

        # Gamma must have a materially higher dead rate (>= 5pp difference)
        assert gamma_dead_rate > ref_dead_rate, (
            f"issuer_gamma dead rate ({gamma_dead_rate:.2%}) must exceed "
            f"reference ({ref_dead_rate:.2%})"
        )
        assert gamma_dead_rate - ref_dead_rate >= 0.05, (
            f"Dead rate difference ({gamma_dead_rate - ref_dead_rate:.2%}) "
            f"is not economically meaningful (need >= 5pp)"
        )


class TestInstrumentDeadHazardDifference:
    """Instrument hazard variation: upi_autopay must have higher dead-hazard."""

    def test_upi_autopay_dead_rate_exceeds_debit_credit_cards(self):
        """Over many attempts, upi_autopay must show materially higher DEAD rate.

        Same aggregation strategy as the issuer test, and the same 150-seed
        rationale (see that test's docstring, and DECISIONS.md 2026-09-04
        R1b review pass): the original 20-seed window measured +5.69pp,
        0.69pp above the 5pp floor against an empirical window-SD of
        ~1.36pp -- real, but not the "wide margin" it was once documented
        as. 150 seeds measures +7.18pp here."""
        from eval.sim2 import Simulator2, INSTRUMENT_LEVELS
        from src.core.types import Outcome

        upi_autopay = "upi_autopay"
        assert upi_autopay in INSTRUMENT_LEVELS

        # Card instruments (non-reference)
        card_instruments = [i for i in INSTRUMENT_LEVELS if i != upi_autopay]

        instrument_outcomes = {}

        for seed in range(1000, 1150):  # 150 seeds -- see docstring for why
            sim = Simulator2(seed=seed)

            for mandate in sim.mandates:
                instrument = mandate.instrument_type
                if instrument not in instrument_outcomes:
                    instrument_outcomes[instrument] = {"dead": 0, "total": 0}

                result = sim.attempt(mandate.mandate_id, slot=2, on_day=1)
                instrument_outcomes[instrument]["total"] += 1
                if result.outcome == Outcome.DEAD:
                    instrument_outcomes[instrument]["dead"] += 1

        upi_dead_rate = (
            instrument_outcomes[upi_autopay]["dead"]
            / instrument_outcomes[upi_autopay]["total"]
        )

        # Average dead rate across card instruments
        card_dead_rates = [
            instrument_outcomes[card]["dead"] / instrument_outcomes[card]["total"]
            for card in card_instruments
        ]
        avg_card_dead_rate = np.mean(card_dead_rates)

        # UPI must have materially higher dead rate (>= 5pp difference)
        assert upi_dead_rate > avg_card_dead_rate, (
            f"upi_autopay dead rate ({upi_dead_rate:.2%}) must exceed "
            f"avg card rate ({avg_card_dead_rate:.2%})"
        )
        assert upi_dead_rate - avg_card_dead_rate >= 0.05, (
            f"UPI vs card dead rate difference ({upi_dead_rate - avg_card_dead_rate:.2%}) "
            f"is not economically meaningful (need >= 5pp)"
        )


class TestGenerateCorpus:
    """generate_corpus() produces episodes with legal attempt schedules."""

    def test_generate_corpus_returns_list_of_episodes(self):
        """generate_corpus() must return a list of episode-like objects."""
        from eval.sim2 import generate_corpus

        corpus = generate_corpus(seeds=(1000, 1001), rng_seed=42)

        assert isinstance(corpus, list)
        assert len(corpus) > 0

    def test_generate_corpus_episodes_have_required_fields(self):
        """Each episode must have .mandate, .attempts, .censor_reason fields."""
        from eval.sim2 import generate_corpus

        corpus = generate_corpus(seeds=(1000, 1001), rng_seed=42)

        for episode in corpus:
            assert hasattr(episode, "mandate")
            assert hasattr(episode, "attempts")
            assert hasattr(episode, "censor_reason")

    def test_generate_corpus_mandate_ids_are_namespaced_by_seed(self):
        """Episode mandates must have mandate_id = f's{seed}:...'."""
        from eval.sim2 import generate_corpus

        corpus = generate_corpus(seeds=(2000, 3000), rng_seed=42)

        mandate_ids = [ep.mandate.mandate_id for ep in corpus]
        assert any("s2000:" in mid for mid in mandate_ids), (
            "No mandate_id found with s2000: prefix"
        )
        assert any("s3000:" in mid for mid in mandate_ids), (
            "No mandate_id found with s3000: prefix"
        )

    def test_generate_corpus_attempts_have_strictly_increasing_days(self):
        """All attempts within an episode must have on_day strictly increasing."""
        from eval.sim2 import generate_corpus

        corpus = generate_corpus(seeds=(1000, 1001), rng_seed=42)

        for episode in corpus:
            days = [attempt.on_day for attempt in episode.attempts]
            # Days must be strictly increasing
            for i in range(1, len(days)):
                assert days[i] > days[i - 1], (
                    f"Episode {episode.mandate.mandate_id}: days not strictly "
                    f"increasing: {days}"
                )

    def test_generate_corpus_first_attempt_day_is_at_least_one(self):
        """First attempt in each episode must be on day >= 1 (commitment lag)."""
        from eval.sim2 import generate_corpus

        corpus = generate_corpus(seeds=(1000, 1001), rng_seed=42)

        for episode in corpus:
            if episode.attempts:
                first_day = episode.attempts[0].on_day
                assert first_day >= 1, (
                    f"Episode {episode.mandate.mandate_id}: first attempt on day "
                    f"{first_day}, expected >= 1"
                )

    def test_generate_corpus_slots_match_attempt_order(self):
        """Attempt slots must be 2, 3, 4 in order (slot 1 is given)."""
        from eval.sim2 import generate_corpus

        corpus = generate_corpus(seeds=(1000, 1001), rng_seed=42)

        for episode in corpus:
            expected_slot = 2
            for attempt in episode.attempts:
                assert attempt.slot == expected_slot, (
                    f"Episode {episode.mandate.mandate_id}: expected slot "
                    f"{expected_slot}, got {attempt.slot}"
                )
                expected_slot += 1


class TestBuildSim2Features:
    """build_sim2_features() assembles a feature matrix from corpus and person-period data."""

    def test_build_sim2_features_output_shape(self):
        """Output must have one row per input row, with issuer/instrument/age columns."""
        from eval.sim2 import generate_corpus, build_sim2_features, Simulator2
        from src.model.person_period import build

        corpus = generate_corpus(seeds=(1000,), rng_seed=42)
        pp_df = build(corpus)

        # build_sim2_features signature: pass corpus as an argument so it can
        # look up mandate metadata
        features_df = build_sim2_features(pp_df, corpus=corpus)

        assert len(features_df) == len(pp_df), (
            f"Feature frame has {len(features_df)} rows, pp_df has {len(pp_df)}"
        )

    def test_build_sim2_features_has_required_columns(self):
        """Output must include row_id, issuer_id, instrument_type, mandate_age_days."""
        from eval.sim2 import generate_corpus, build_sim2_features
        from src.model.person_period import build

        corpus = generate_corpus(seeds=(1000,), rng_seed=42)
        pp_df = build(corpus)
        features_df = build_sim2_features(pp_df, corpus=corpus)

        required_cols = ["row_id", "issuer_id", "instrument_type", "mandate_age_days"]
        for col in required_cols:
            assert col in features_df.columns, f"Missing column: {col}"

    def test_build_sim2_features_constant_per_mandate(self):
        """issuer/instrument/age must be constant across all slots of one mandate."""
        from eval.sim2 import generate_corpus, build_sim2_features
        from src.model.person_period import build

        corpus = generate_corpus(seeds=(1000,), rng_seed=42)
        pp_df = build(corpus)
        features_df = build_sim2_features(pp_df, corpus=corpus)

        # row_id format is f"{mandate_id}:{cycle_id}:{slot}" (src/core/ids.py's
        # row_id()) and mandate_id ITSELF contains a colon once namespaced
        # (f"s{seed}:{mandate_id}") -- a plain split(":")[0] would collapse
        # every mandate under one seed into a single group and silently pass
        # this test for the wrong reason. rsplit from the right strips exactly
        # the trailing ":cycle_id:slot" and keeps the real mandate_id intact.
        features_df["mandate_id"] = (
            features_df["row_id"].str.rsplit(":", n=2, expand=True)[0]
        )

        for mandate_id in features_df["mandate_id"].unique():
            mandate_rows = features_df[features_df["mandate_id"] == mandate_id]

            # All rows for this mandate must have the same issuer/instrument/age
            assert mandate_rows["issuer_id"].nunique() == 1, (
                f"Mandate {mandate_id} has varying issuer_id"
            )
            assert mandate_rows["instrument_type"].nunique() == 1, (
                f"Mandate {mandate_id} has varying instrument_type"
            )
            assert mandate_rows["mandate_age_days"].nunique() == 1, (
                f"Mandate {mandate_id} has varying mandate_age_days"
            )

    def test_build_sim2_features_values_are_valid(self):
        """issuer_id/instrument_type values must match corpus; age must be >= 0."""
        from eval.sim2 import (
            generate_corpus,
            build_sim2_features,
            ISSUER_LEVELS,
            INSTRUMENT_LEVELS,
        )
        from src.model.person_period import build

        corpus = generate_corpus(seeds=(1000,), rng_seed=42)
        pp_df = build(corpus)
        features_df = build_sim2_features(pp_df, corpus=corpus)

        valid_issuers = set(ISSUER_LEVELS)
        valid_instruments = set(INSTRUMENT_LEVELS)

        assert features_df["issuer_id"].isin(valid_issuers).all(), (
            "Found issuer_id values outside ISSUER_LEVELS"
        )
        assert features_df["instrument_type"].isin(valid_instruments).all(), (
            "Found instrument_type values outside INSTRUMENT_LEVELS"
        )
        assert (features_df["mandate_age_days"] >= 0).all(), (
            "Found negative mandate_age_days"
        )


import dataclasses
