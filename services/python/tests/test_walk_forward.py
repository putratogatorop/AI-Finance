# tests/test_walk_forward.py
import sys

sys.path.insert(0, ".")



def test_walk_forward_basic():
    from src.ml.walk_forward import walk_forward_splits

    splits = list(walk_forward_splits(
        n_samples=50000,
        train_size=26208,
        val_size=2880,
        test_size=2880,
        step_size=1344,
        embargo=16,
    ))

    assert len(splits) > 10

    # First split
    train, val, test = splits[0]
    assert train == (0, 26208)
    assert val == (26224, 29104)   # 26208 + 16 embargo = 26224
    assert test == (29120, 32000)  # 29104 + 16 embargo = 29120

    # No overlap
    assert train[1] + 16 <= val[0]
    assert val[1] + 16 <= test[0]


def test_walk_forward_sliding():
    from src.ml.walk_forward import walk_forward_splits

    splits = list(walk_forward_splits(
        n_samples=50000,
        train_size=26208,
        val_size=2880,
        test_size=2880,
        step_size=1344,
        embargo=16,
    ))

    # Each split slides by step_size
    _, _, test0 = splits[0]
    _, _, test1 = splits[1]
    assert test1[0] - test0[0] == 1344


def test_walk_forward_no_future_leak():
    from src.ml.walk_forward import walk_forward_splits

    splits = list(walk_forward_splits(
        n_samples=50000,
        train_size=26208,
        val_size=2880,
        test_size=2880,
        step_size=1344,
        embargo=16,
    ))

    for train, val, test in splits:
        assert train[1] + 16 <= val[0]
        assert val[1] + 16 <= test[0]
        assert test[1] <= 50000


def test_walk_forward_count():
    from src.ml.walk_forward import walk_forward_splits

    splits = list(walk_forward_splits(
        n_samples=104528,
        train_size=26208,
        val_size=2880,
        test_size=2880,
        step_size=1344,
        embargo=16,
    ))

    assert 20 <= len(splits) <= 60
