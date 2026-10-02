"""Smoke test for the GHZ + measurement-counts example (ghz_counts.py)."""

import simq


def build_ghz():
    builder = simq.CircuitBuilder(3)
    builder.h(0)
    builder.cx(0, 1)
    builder.cx(1, 2)
    return builder.build()


def test_ghz_state_probabilities():
    sim = simq.Simulator()
    probs = sim.run(build_ghz()).probabilities
    assert len(probs) == 8
    assert abs(probs[0] - 0.5) < 0.01  # |000>
    assert abs(probs[7] - 0.5) < 0.01  # |111>
    assert abs(sum(probs) - 1.0) < 1e-9
    for i in (1, 2, 3, 4, 5, 6):
        assert abs(probs[i]) < 0.01


def test_ghz_counts_only_000_and_111():
    config = simq.SimulatorConfig(shots=1024)
    sim = simq.Simulator(config)
    counts = sim.run_with_shots(build_ghz(), shots=1024)
    assert set(counts) <= {"000", "111"}
    assert counts.get("000", 0) + counts.get("111", 0) == 1024
