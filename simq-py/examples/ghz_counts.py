"""GHZ state + measurement counts (Python twin of the README quickstart).

Builds the 3-qubit GHZ state (h(0); cx(0,1); cx(1,2)), samples 1024
shots, and prints the counts sorted by frequency: roughly 50/50
between `000` and `111`, nothing else.
"""

import simq


def main():
    builder = simq.CircuitBuilder(3)
    builder.h(0)
    builder.cx(0, 1)
    builder.cx(1, 2)
    circuit = builder.build()

    config = simq.SimulatorConfig(shots=1024)
    simulator = simq.Simulator(config)
    counts = simulator.run_with_shots(circuit, shots=1024)

    for bitstring, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"{bitstring}: {count}")


if __name__ == "__main__":
    main()
