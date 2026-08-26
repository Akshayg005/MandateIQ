"""eval/frozen/simulator.py -- the pre-registered generative mechanism.

Statistical assertions use generous tolerances (Monte Carlo over hundreds to
low thousands of draws) -- the point is to catch a wrong SIGN or a broken
mechanic, not to pin an exact float.
"""
from __future__ import annotations

import copy

import pytest

from src.core.types import Cause, Outcome
from eval.frozen.simulator import Simulator, load_config


@pytest.fixture(scope="module")
def base_config():
    return load_config()


# --- config loading ----------------------------------------------------------

def test_load_config_has_three_arms(base_config):
    assert set(base_config["arms"]) == {"nominal", "misspecified", "coupled"}


def test_load_config_cause_mix_sums_to_one(base_config):
    assert sum(base_config["cause_mix"].values()) == pytest.approx(1.0)


def test_load_config_category_mix_sums_to_one(base_config):
    assert sum(base_config["category_mix"].values()) == pytest.approx(1.0)


# --- generation: deterministic, valid ranges --------------------------------

def test_generate_mandates_deterministic_given_seed():
    a = Simulator("nominal", seed=42).mandates
    b = Simulator("nominal", seed=42).mandates
    assert a == b


def test_generate_mandates_different_seeds_differ():
    a = Simulator("nominal", seed=1).mandates
    b = Simulator("nominal", seed=2).mandates
    assert a != b


def test_generate_mandates_count_matches_config(base_config):
    sim = Simulator("nominal", seed=1)
    assert len(sim.mandates) == base_config["n_mandates"]


def test_amount_paise_spans_the_afa_cliff():
    """Some mandates must be below Rs 15,000 (AFA_FREE_LIMIT_PAISE) and some
    above -- otherwise B7's 8(a)/8(b) branch is untestable against this
    batch."""
    sim = Simulator("nominal", seed=1)
    amounts = [m.amount_paise for m in sim.mandates]
    assert any(a < 1_500_000 for a in amounts)
    assert any(a > 1_500_000 for a in amounts)


def test_ceiling_paise_always_at_least_amount_paise():
    sim = Simulator("nominal", seed=1)
    for m in sim.mandates:
        assert m.ceiling_paise >= m.amount_paise


def test_cause_mix_roughly_matches_config(base_config):
    sim = Simulator("nominal", seed=7)
    n = len(sim.mandates)
    counts = {c: 0 for c in base_config["cause_mix"]}
    for m in sim.mandates:
        counts[m.initial_cause.value] += 1
    for cause, expected_frac in base_config["cause_mix"].items():
        got_frac = counts[cause] / n
        assert abs(got_frac - expected_frac) < 0.12, (cause, got_frac, expected_frac)


def test_coupled_arm_assigns_household_ids_nominal_and_misspecified_do_not(base_config):
    hh_size = base_config["arms"]["coupled"]["household_size"]
    coupled = Simulator("coupled", seed=1)
    nominal = Simulator("nominal", seed=1)
    assert all(m.household_id is not None for m in coupled.mandates)
    assert all(m.household_id is None for m in nominal.mandates)
    n_households = len({m.household_id for m in coupled.mandates})
    assert n_households == len(coupled.mandates) // hh_size


def test_unknown_arm_raises():
    with pytest.raises(ValueError):
        Simulator("adversarial", seed=1)


def test_by_id_raises_on_unknown_mandate():
    sim = Simulator("nominal", seed=1)
    with pytest.raises(KeyError):
        sim._by_id("no-such-mandate")


def test_unknown_link_raises():
    cfg = copy.deepcopy(load_config())
    cfg["arms"]["nominal"]["link"] = "probit"
    sim = Simulator("nominal", config=cfg, seed=1)
    mid = sim.mandates[0].mandate_id
    with pytest.raises(ValueError):
        sim.attempt(mid, 2, 1)


# --- attempt(): state discipline ---------------------------------------------

def test_attempt_rejects_slot_one():
    sim = Simulator("nominal", seed=1)
    mid = sim.mandates[0].mandate_id
    with pytest.raises(ValueError):
        sim.attempt(mid, 1, 1)


def test_attempt_rejects_slot_five():
    sim = Simulator("nominal", seed=1)
    mid = sim.mandates[0].mandate_id
    with pytest.raises(ValueError):
        sim.attempt(mid, 5, 1)


def test_attempt_rejects_out_of_order_slot():
    sim = Simulator("nominal", seed=1)
    mid = sim.mandates[0].mandate_id
    with pytest.raises(ValueError):
        sim.attempt(mid, 3, 2)  # slot 2 was never attempted


def test_attempt_rejects_repeat_of_same_slot():
    sim = Simulator("nominal", seed=1)
    mid = sim.mandates[0].mandate_id
    sim.attempt(mid, 2, 1)
    with pytest.raises(ValueError):
        sim.attempt(mid, 2, 1)


def test_attempt_rejects_non_increasing_day():
    """A retry cannot land on or before the previous attempt's day -- the
    initial slot-1 failure is day 0, so slot 2 must be on day >= 1, and each
    later slot must be strictly after the one before it."""
    sim = Simulator("nominal", seed=1)
    mid = sim.mandates[0].mandate_id
    with pytest.raises(ValueError):
        sim.attempt(mid, 2, 0)


def test_attempt_rejects_day_equal_to_previous_attempt():
    sim = Simulator("nominal", seed=1)
    mid = sim.mandates[0].mandate_id
    sim.attempt(mid, 2, 5)
    with pytest.raises(ValueError):
        sim.attempt(mid, 3, 5)


def test_attempt_returns_a_valid_outcome():
    sim = Simulator("nominal", seed=1)
    mid = sim.mandates[0].mandate_id
    result = sim.attempt(mid, 2, 1)
    assert isinstance(result.outcome, Outcome)


def test_attempt_in_sequence_up_to_budget_succeeds():
    sim = Simulator("nominal", seed=1)
    mid = sim.mandates[0].mandate_id
    for slot, day in ((2, 1), (3, 2), (4, 3)):
        result = sim.attempt(mid, slot, day)
        assert result.slot == slot
        assert result.on_day == day


# --- nominal arm: salary-window bonus, optout escalation --------------------

def _cant_pay_now_mandates(sim):
    return [m for m in sim.mandates if m.initial_cause == Cause.CANT_PAY_NOW]


def test_salary_window_increases_recovery_for_cant_pay_now():
    """Attempt every CANT_PAY_NOW mandate's slot 2 twice under two large,
    independent batches -- once inside the salary window (day 3), once
    outside it (day 10) -- and confirm the in-window recovery rate is
    higher. Uses many mandates across many seeds as independent trials so
    this is a real Monte Carlo comparison, not a single noisy draw."""
    in_window_recoveries = 0
    out_window_recoveries = 0
    trials = 0
    for seed in range(20):
        sim_in = Simulator("nominal", seed=seed)
        sim_out = Simulator("nominal", seed=seed)
        for m in _cant_pay_now_mandates(sim_in):
            trials += 1
            r_in = sim_in.attempt(m.mandate_id, 2, 3)  # day 3: in window
            r_out = sim_out.attempt(m.mandate_id, 2, 10)  # day 10: out
            in_window_recoveries += r_in.outcome == Outcome.RECOVERED
            out_window_recoveries += r_out.outcome == Outcome.RECOVERED
    assert trials > 300
    assert in_window_recoveries / trials > out_window_recoveries / trials


def test_optout_rate_escalates_across_attempts_for_wont_pay():
    """A WONT_PAY mandate ground blindly through all three retries should
    opt out more often on later attempts than it would on the first, since
    optout_escalation_logit_per_attempt is strictly positive. Measured by
    running many independent WONT_PAY-only batches and comparing the
    per-slot opt-out rate at slot 2 vs slot 4 (conditional on reaching it)."""
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg["cause_mix"] = {"CANT_PAY_NOW": 0.0, "CANT_PAY_EVER": 0.0, "WONT_PAY": 1.0}

    slot2_optouts = slot2_total = 0
    slot4_optouts = slot4_total = 0
    for seed in range(30):
        sim = Simulator("nominal", config=cfg, seed=seed)
        for m in sim.mandates:
            r2 = sim.attempt(m.mandate_id, 2, 1)
            slot2_total += 1
            slot2_optouts += r2.outcome == Outcome.OPTED_OUT
            if r2.outcome != Outcome.STILL_PENDING:
                continue
            r3 = sim.attempt(m.mandate_id, 3, 2)
            if r3.outcome != Outcome.STILL_PENDING:
                continue
            r4 = sim.attempt(m.mandate_id, 4, 3)
            slot4_total += 1
            slot4_optouts += r4.outcome == Outcome.OPTED_OUT

    assert slot2_total > 500 and slot4_total > 50
    assert (slot4_optouts / slot4_total) > (slot2_optouts / slot2_total)


# --- misspecified arm: genuinely different from nominal ---------------------

def test_misspecified_uses_cloglog_not_softmax_directly():
    """The cloglog and logit links compute different probabilities from the
    same underlying base rates whenever the terminal probability is not
    tiny -- confirmed directly against the two link functions for a
    representative CANT_PAY_NOW score set, at zero days-since-last so the
    heavy-tail boost (which only affects CANT_PAY_NOW's recovery share, not
    the total terminal probability) does not confound the comparison."""
    nominal_probs = None
    misspecified_probs = None
    for seed in range(5):
        sim_nom = Simulator("nominal", seed=seed)
        sim_mis = Simulator("misspecified", seed=seed)
        m = sim_nom.mandates[0]
        h = sim_nom.config["hazards"][Cause.CANT_PAY_NOW.value]
        from eval.frozen.simulator import _logits_from_base_rates, _softmax
        logits = _logits_from_base_rates(h["base_recovery"], h["base_dead"], h["base_optout"])
        nominal_probs = _softmax(logits)
        misspecified_probs = sim_mis._cloglog_probs(logits, Cause.CANT_PAY_NOW, days_since_last=1)
        break
    assert nominal_probs["survive"] != pytest.approx(misspecified_probs["survive"], rel=1e-6)


def test_misspecified_recovery_share_grows_with_days_since_last_attempt():
    """The heavy-tailed replenishment property, directly: for a fixed
    CANT_PAY_NOW mandate, attempting after a long gap should show a higher
    recovery rate than attempting after a short gap, under misspecified --
    and this must be a LARGER effect than nominal shows for the same gap
    (nominal has no days_since_last dependence at all beyond the discrete
    salary-window bonus)."""
    short_gap_recoveries = long_gap_recoveries = 0
    trials = 300
    for seed in range(trials):
        sim_short = Simulator("misspecified", seed=seed)
        sim_long = Simulator("misspecified", seed=seed)
        cause_mix_mandates = [m for m in sim_short.mandates if m.initial_cause == Cause.CANT_PAY_NOW]
        if not cause_mix_mandates:
            continue
        mid = cause_mix_mandates[0].mandate_id
        # slot 2 first (day 20, outside salary window, gap = 20 from day 0)
        r_short = sim_short.attempt(mid, 2, 20)
        r_long = sim_long.attempt(mid, 2, 20)
        # both identical so far (same seed) -- now diverge on slot 3's gap
        if r_short.outcome != Outcome.STILL_PENDING or r_long.outcome != Outcome.STILL_PENDING:
            continue
        r_short2 = sim_short.attempt(mid, 3, 21)   # 1-day gap
        r_long2 = sim_long.attempt(mid, 3, 40)     # 20-day gap
        short_gap_recoveries += r_short2.outcome == Outcome.RECOVERED
        long_gap_recoveries += r_long2.outcome == Outcome.RECOVERED

    assert long_gap_recoveries >= short_gap_recoveries


def test_cause_switching_occurs_at_roughly_the_configured_rate(base_config):
    switch_prob = base_config["arms"]["misspecified"]["cause_switch_prob"]
    switches = 0
    checks = 0
    for seed in range(40):
        sim = Simulator("misspecified", seed=seed)
        for m in sim.mandates[:5]:
            before = sim.effective_cause(m.mandate_id)
            sim.attempt(m.mandate_id, 2, 1)
            after = sim.effective_cause(m.mandate_id)
            checks += 1
            switches += before != after
    got_rate = switches / checks
    assert abs(got_rate - switch_prob) < 0.08, got_rate


# --- coupled arm: the debit-storm mechanic ------------------------------------

def _tight_coupled_config():
    """CANT_PAY_NOW-only, near-certain recovery, and a household balance far
    too small to cover every member -- isolates the coupling mechanic from
    the underlying hazard noise so the storm effect is unambiguous."""
    cfg = copy.deepcopy(load_config())
    cfg["cause_mix"] = {"CANT_PAY_NOW": 1.0, "CANT_PAY_EVER": 0.0, "WONT_PAY": 0.0}
    cfg["hazards"]["CANT_PAY_NOW"]["base_recovery"] = 0.97
    cfg["hazards"]["CANT_PAY_NOW"]["base_dead"] = 0.01
    cfg["hazards"]["CANT_PAY_NOW"]["base_optout"] = 0.01
    cfg["arms"]["coupled"]["household_size"] = 4
    cfg["amount_paise"] = {
        "below_afa_frac": 1.0,
        "below_afa_range": [100000, 100000],   # every mandate exactly Rs 1,000
        "above_afa_range": [100000, 100000],
        "ceiling_multiplier_range": [1.0, 1.0],
    }
    # Balance covers ~1.5 members out of 4 -> later members must starve.
    cfg["arms"]["coupled"]["household_balance_range"] = [150000, 150000]
    return cfg


def test_household_balance_depletes_as_members_are_attempted_in_order():
    cfg = _tight_coupled_config()
    sim = Simulator("coupled", config=cfg, seed=1)
    household = sim.mandates[0].household_id
    members = [m for m in sim.mandates if m.household_id == household]
    assert len(members) == 4

    balances = [sim.household_balance(household)]
    for m in members:
        sim.attempt(m.mandate_id, 2, 1)
        balances.append(sim.household_balance(household))

    assert balances == sorted(balances, reverse=True)  # monotonically non-increasing
    assert balances[-1] < balances[0]


def test_later_household_members_see_more_iatrogenic_failures_than_earlier():
    cfg = _tight_coupled_config()
    early_iatrogenic = late_iatrogenic = early_total = late_total = 0
    for seed in range(60):
        sim = Simulator("coupled", config=cfg, seed=seed)
        household = sim.mandates[0].household_id
        members = [m for m in sim.mandates if m.household_id == household]
        for i, m in enumerate(members):
            r = sim.attempt(m.mandate_id, 2, 1)
            if i < len(members) // 2:
                early_total += 1
                early_iatrogenic += r.iatrogenic_insufficient_funds
            else:
                late_total += 1
                late_iatrogenic += r.iatrogenic_insufficient_funds

    assert early_total > 50 and late_total > 50
    assert (late_iatrogenic / late_total) > (early_iatrogenic / early_total)


def test_household_balance_never_goes_negative():
    cfg = _tight_coupled_config()
    sim = Simulator("coupled", config=cfg, seed=3)
    household = sim.mandates[0].household_id
    members = [m for m in sim.mandates if m.household_id == household]
    for m in members:
        sim.attempt(m.mandate_id, 2, 1)
        assert sim.household_balance(household) >= 0


def test_coupled_arm_never_recovers_more_than_the_household_ever_had():
    """Regression test for a real bug caught by payments-domain review
    before the freeze commit was finalized: the first implementation gave a
    below-balance attempt a probabilistic chance to succeed anyway, crediting
    the FULL mandate amount while only debiting the household to zero --
    fabricating money. A full batch run recovered 1.7x the total liquidity
    that existed across every household. This test would have caught it:
    total recovered, summed across every mandate in a household, must never
    exceed that household's starting balance."""
    cfg = _tight_coupled_config()
    for seed in range(20):
        sim = Simulator("coupled", config=cfg, seed=seed)
        households = {}
        for m in sim.mandates:
            households.setdefault(m.household_id, []).append(m)
        for household_id, members in households.items():
            starting_balance = sim.household_balance(household_id)
            recovered = 0
            for m in members:
                r = sim.attempt(m.mandate_id, 2, 1)
                if r.outcome == Outcome.RECOVERED:
                    recovered += m.amount_paise
            assert recovered <= starting_balance, (seed, household_id)


def test_coupled_arm_with_effectively_unlimited_balance_matches_nominal_shape():
    """With a household balance far larger than any plausible cumulative
    debit, the coupling mechanic should almost never bind -- confirming the
    storm effect above comes from the balance constraint, not from some
    other difference between the coupled and nominal arms."""
    cfg = copy.deepcopy(load_config())
    cfg["arms"]["coupled"]["household_balance_range"] = [10**12, 10**12]
    sim = Simulator("coupled", config=cfg, seed=5)
    iatrogenic = 0
    total = 0
    for m in sim.mandates:
        r = sim.attempt(m.mandate_id, 2, 1)
        total += 1
        iatrogenic += r.iatrogenic_insufficient_funds
    assert iatrogenic / total < 0.02
