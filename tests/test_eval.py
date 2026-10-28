import numpy as np
import pytest

from eval import datasets as D
from eval.metrics import accuracy, confusion, ece, macro_f1, mae, to3


def test_macro_f1():
    # class -1: F=2/3 ; class 0: F=2/3 ; class 1: F=1
    assert macro_f1([-1, -1, 0, 1], [-1, 0, 0, 1]) == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)


def test_accuracy_mae():
    assert accuracy([1, 0, -1, 1], [1, 0, 0, 0]) == 0.5
    assert mae([-2, 0, 2], [-1, 0, 0]) == pytest.approx(1.0)


def test_ece_tiny():
    p = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.7, 0.3]])
    y = np.array([0, 1, 1, 0])
    # separate bins: |1-.9|=.1, wrong at .6 -> .6, |1-.8|=.2, |1-.7|=.3 -> mean .3
    assert ece(p, y) == pytest.approx(0.3)
    assert ece(np.array([[1.0, 0.0]]), np.array([0])) == 0.0


def test_to3_confusion():
    r = to3(np.array([[0.1, 0.2, 0.3, 0.3, 0.1]]))
    assert r.tolist()[0] == pytest.approx([0.3, 0.3, 0.4])
    assert confusion([-1, 0, 1, 1], [-1, 1, 1, 0], (-1, 0, 1)) == [[1, 0, 0], [0, 0, 1], [0, 1, 1]]


def test_fiqa_bins():
    cases = {-1.0: -2, -0.5: -2, -0.49: -1, -0.11: -1, -0.1: 0, 0.0: 0, 0.1: 0, 0.11: 1, 0.49: 1, 0.5: 2, 0.9: 2}
    for s, l in cases.items():
        assert D.fiqa_label5(s) == l


def test_normalize_and_errors():
    assert D.normalize("  Hello \n  WORLD ") == "hello world"
    with pytest.raises(ValueError):
        D.load("nope", "test")


@pytest.mark.network
@pytest.mark.parametrize("name", D.DATASETS)
def test_dataset_mapping(name):
    test = D.load(name, "test")
    train = D.load(name, "train")
    assert test and train
    for e in test:
        assert e.label3 in (-1, 0, 1) and e.text and e.channel in ("news", "social")
        assert e.dataset == name
    assert {e.label3 for e in test} == {-1, 0, 1}
    if name == "fiqa":
        assert all(e.label3 == (e.label5 > 0) - (e.label5 < 0) for e in test)
    else:
        assert all(e.label5 is None for e in test)
    a = D.load(name, "test", limit=20, seed=1)
    assert len(a) == 20 and a == D.load(name, "test", limit=20, seed=1)


# --- 5-class probabilistic scores and uncertainty (hand-computed) ---------------------------------
from eval.metrics import bootstrap_ci, brier5, ece_eqmass, ece_floor, log_loss5, mcnemar, paired_bootstrap, rps

ONE0 = [0, 0, 1, 0, 0]
FAR = [1, 0, 0, 0, 0]
UNI = [0.2] * 5


def test_rps():
    assert rps([ONE0], [0]) == 0.0
    assert rps([FAR], [2]) == pytest.approx(1.0)          # CDF gaps 1,1,1,1 -> 4 / 4
    assert rps([UNI], [0]) == pytest.approx(0.1)          # gaps .2,.4,.4,.2 -> .4 / 4
    assert rps([ONE0, FAR, UNI], [0, 2, 0]) == pytest.approx(1.1 / 3)


def test_log_loss5_and_brier5():
    assert log_loss5([UNI], [1]) == pytest.approx(np.log(5))
    assert log_loss5([FAR], [2]) == pytest.approx(-np.log(1e-12))   # clipped
    assert brier5([ONE0], [0]) == 0.0
    assert brier5([UNI], [0]) == pytest.approx(0.8)       # 4 * .04 + .64
    assert brier5([FAR, UNI], [2, 0]) == pytest.approx((2.0 + 0.8) / 2)


def test_ece_eqmass():
    p = [[0.9, 0.025, 0.025, 0.025, 0.025],   # conf .9 correct
         [0.05, 0.8, 0.05, 0.05, 0.05],       # conf .8 wrong
         [0.1, 0.1, 0.6, 0.1, 0.1],           # conf .6 correct
         [0.15, 0.15, 0.15, 0.15, 0.4]]       # conf .4 correct
    y = [-2, 2, 0, 2]
    # bins by confidence: {.4, .6}: |1 - .5| * .5 ; {.8, .9}: |.5 - .85| * .5
    assert ece_eqmass(p, y, bins=2) == pytest.approx(0.25 + 0.175)
    assert ece_eqmass(p, y, bins=10) == pytest.approx((0.6 + 0.8 + 0.4 + 0.1) / 4)   # bins capped at n


def test_ece_floor():
    assert ece_floor([ONE0] * 20) == 0.0                  # certain predictions: sampled labels always agree
    p = np.tile(UNI, (50, 1))
    f = ece_floor(p, n_sim=200, seed=1)
    assert 0 < f < 1 and f == ece_floor(p, n_sim=200, seed=1)


def test_mcnemar():
    assert mcnemar([1, 1, 1, 0], [0, 0, 0, 0]) == pytest.approx(0.25)    # 3 vs 0 discordant: 2 / 8
    assert mcnemar([1] * 5 + [0], [0] * 5 + [0]) == pytest.approx(0.0625)  # 2 / 32
    assert mcnemar([1, 0], [0, 1]) == 1.0
    assert mcnemar([1, 0, 1], [1, 0, 1]) == 1.0


def test_bootstrap():
    y = np.arange(30)
    assert bootstrap_ci(lambda a: 7.0, y) == (7.0, 7.0)
    lo, hi = bootstrap_ci(np.mean, y, n=500)
    assert lo < y.mean() < hi
    r = paired_bootstrap(lambda a: float(a.mean()), (y,), (y,), n=200)
    assert r == {"diff": 0.0, "lo": 0.0, "hi": 0.0}
    r = paired_bootstrap(lambda a: float(a.mean()), (y + 1.0,), (y,), n=200)   # same rows -> exact shift
    assert [r["diff"], r["lo"], r["hi"]] == pytest.approx([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        bootstrap_ci(np.mean, y, y[:3])
