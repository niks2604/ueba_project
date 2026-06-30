"""Tests for the Evaluator (ranking + precision/recall metrics)."""

import copy

import numpy as np
import pandas as pd
import pytest

from config import EvaluationConfig
from evaluator import Evaluator, EvaluationResult


TEST_WINDOW_START = 5


def _eval_config(aggregation="sum"):
    return EvaluationConfig(
        top_k_values=[1, 2, 3],
        primary_top_k=2,
        attack_types_tracked=["exfil", "lateral", "cred_theft"],
        test_window_start=TEST_WINDOW_START,
        aggregation=aggregation,
    )


def _toy_scenario():
    """
    6 users, days 0-7, test window starts day 5.

    Users 0 and 1 are attackers (exfil, lateral). Scores are arranged
    so the user-level ranking is exactly user 0 > 1 > 2 > ... > 5,
    which makes the metrics analytically predictable.
    """
    rows = []
    scores = []
    for user in range(6):
        attacker = user in (0, 1)
        atype = {0: "exfil", 1: "lateral"}.get(user, "")
        for day in range(8):
            in_window = day >= TEST_WINDOW_START
            rows.append({
                "user_id": user,
                "day": day,
                "is_anomaly": attacker and in_window,
                "anomaly_type": atype if (attacker and in_window) else "",
            })
            # Training rows score ~0; test rows decrease with user id.
            scores.append((10 - user) if in_window else 0.01)
    pdf = pd.DataFrame(rows)
    return pdf, np.array(scores, dtype=float)


VICTIMS = {"exfil": [0], "lateral": [1], "cred_theft": []}


def test_constructor_rejects_unknown_aggregation():
    with pytest.raises(ValueError, match="aggregation"):
        Evaluator(_eval_config(aggregation="median"))


def test_score_length_mismatch_raises():
    pdf, scores = _toy_scenario()
    ev = Evaluator(_eval_config())
    with pytest.raises(ValueError, match="does not match"):
        ev.evaluate(pdf=pdf, scores=scores[:-1], victims=VICTIMS)


def test_returns_result_object():
    pdf, scores = _toy_scenario()
    result = Evaluator(_eval_config()).evaluate(pdf=pdf, scores=scores, victims=VICTIMS)
    assert isinstance(result, EvaluationResult)


def test_n_attackers_total_counts_victims():
    pdf, scores = _toy_scenario()
    result = Evaluator(_eval_config()).evaluate(pdf=pdf, scores=scores, victims=VICTIMS)
    assert result.n_attackers_total == 2


def test_ranking_is_sorted_descending():
    pdf, scores = _toy_scenario()
    result = Evaluator(_eval_config()).evaluate(pdf=pdf, scores=scores, victims=VICTIMS)
    rankings = result.user_rankings
    assert list(rankings["user_id"]) == [0, 1, 2, 3, 4, 5]
    assert list(rankings["rank"]) == [1, 2, 3, 4, 5, 6]
    assert rankings["agg_score"].is_monotonic_decreasing


def test_only_test_window_rows_counted():
    # Aggregation must ignore training-window rows (day < window start).
    pdf, scores = _toy_scenario()
    result = Evaluator(_eval_config("sum")).evaluate(pdf=pdf, scores=scores, victims=VICTIMS)
    # user 0: sum of three test rows of score 10 = 30 (training 0.01 excluded).
    top = result.user_rankings.iloc[0]
    assert top["agg_score"] == pytest.approx(30.0)


def test_precision_and_recall_values():
    pdf, scores = _toy_scenario()
    result = Evaluator(_eval_config()).evaluate(pdf=pdf, scores=scores, victims=VICTIMS)
    # Top-2 are users 0 & 1, both attackers.
    assert result.precision_at_k[2] == pytest.approx(1.0)
    assert result.precision_at_k[1] == pytest.approx(1.0)
    # Top-3 adds user 2 (clean): 2 of 3 are attackers.
    assert result.precision_at_k[3] == pytest.approx(2 / 3)
    # Both attackers caught in top-2 → full recall.
    assert result.recall_at_k[2] == pytest.approx(1.0)
    assert result.recall_at_k[1] == pytest.approx(0.5)


def test_caught_by_type():
    pdf, scores = _toy_scenario()
    result = Evaluator(_eval_config()).evaluate(pdf=pdf, scores=scores, victims=VICTIMS)
    caught = result.caught_by_type
    assert caught["exfil"] == pytest.approx(1.0)
    assert caught["lateral"] == pytest.approx(1.0)
    # No victims of this type → 0.0, not a crash.
    assert caught["cred_theft"] == pytest.approx(0.0)


def test_recall_zero_when_no_attackers():
    pdf, scores = _toy_scenario()
    pdf = pdf.copy()
    pdf["is_anomaly"] = False
    result = Evaluator(_eval_config()).evaluate(
        pdf=pdf, scores=scores, victims={"exfil": [], "lateral": [], "cred_theft": []}
    )
    assert result.n_attackers_total == 0
    assert result.recall_at_k[2] == 0.0


def test_to_dict_is_flat_and_complete():
    pdf, scores = _toy_scenario()
    result = Evaluator(_eval_config()).evaluate(pdf=pdf, scores=scores, victims=VICTIMS)
    flat = result.to_dict()
    assert flat["precision_at_2"] == pytest.approx(1.0)
    assert flat["recall_at_2"] == pytest.approx(1.0)
    assert flat["caught_exfil"] == pytest.approx(1.0)
    assert all(isinstance(v, float) for v in flat.values())


def test_summary_line_mentions_primary_k():
    pdf, scores = _toy_scenario()
    result = Evaluator(_eval_config()).evaluate(pdf=pdf, scores=scores, victims=VICTIMS)
    line = result.summary_line(2)
    assert "P@2" in line and "R@2" in line


@pytest.mark.parametrize("aggregation", ["sum", "max", "mean"])
def test_all_aggregations_run(aggregation):
    pdf, scores = _toy_scenario()
    result = Evaluator(_eval_config(aggregation)).evaluate(
        pdf=pdf, scores=scores, victims=VICTIMS
    )
    # Ranking is unchanged here (per-user score is constant across days),
    # so attackers still top the list under every aggregation.
    assert result.precision_at_k[2] == pytest.approx(1.0)
