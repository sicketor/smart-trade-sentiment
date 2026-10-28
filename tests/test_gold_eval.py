"""eval.gold report math (no models, no network)."""
import numpy as np

from eval.gold import _report


def test_report_perfect_and_scales():
    y = np.array([-2, -1, 0, 1, 2])
    r = _report(y, y.copy(), y / 2.0)
    assert r["acc5"] == 1.0 and r["mae5"] == 0.0 and r["f1_dir"] == 1.0
    assert r["spearman"] == 1.0
    assert r["mean_pred"] == r["mean_gold"] == 0.0   # both on the -1..1 scale


def test_report_neutral_bias_shows_in_means():
    y = np.array([2, 2, 1, 1, 0])
    r = _report(y, np.zeros(5, dtype=int), np.zeros(5))
    assert r["acc5"] == 0.2 and r["mean_pred"] == 0.0 and r["mean_gold"] == 0.6
