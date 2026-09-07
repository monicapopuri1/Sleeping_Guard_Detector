"""Fix 6: DebouncedFlag unit tests -- pure, no state machine involved."""

from guard_monitor.state_machine.debounce import DebouncedFlag


def test_starts_false():
    flag = DebouncedFlag(true_hold_sec=8, false_hold_sec=5)
    assert flag.state is False


def test_true_requires_full_hold():
    flag = DebouncedFlag(true_hold_sec=8, false_hold_sec=5)
    flag.update(0, True)
    assert flag.update(4, True) is False
    assert flag.update(8, True) is True


def test_brief_flicker_never_flips():
    flag = DebouncedFlag(true_hold_sec=8, false_hold_sec=5)
    flag.update(0, True)
    flag.update(4, False)  # flicker before hold elapses
    flag.update(4.5, True)
    # pending timer restarted at 4.5 (raw flipped away and back);
    # 8s from there is 12.5, not yet at 11
    assert flag.update(11, True) is False
    assert flag.update(12.5, True) is True


def test_false_clears_after_hold():
    flag = DebouncedFlag(true_hold_sec=8, false_hold_sec=5)
    flag.update(0, True)
    flag.update(8, True)
    assert flag.state is True
    flag.update(9, False)
    assert flag.update(13, False) is True  # 4s of false, still latched
    assert flag.update(14, False) is False  # 5s of false, clears


def test_reset():
    flag = DebouncedFlag(true_hold_sec=8, false_hold_sec=5)
    flag.update(0, True)
    flag.update(8, True)
    assert flag.state is True
    flag.reset()
    assert flag.state is False
