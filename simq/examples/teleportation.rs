//! Quantum teleportation with the fluent API
//!
//! Run with: cargo run -p simq --example teleportation
//!
//! Teleportation moves an arbitrary qubit state to a distant qubit using
//! one shared Bell pair and two classical bits — no quantum channel between
//! sender and receiver. The circuit:
//! 1. prepare any state on qubit 0 (here RY(0.7)|0⟩),
//! 2. share a Bell pair on qubits 1–2,
//! 3. rotate qubits 0–1 into the Bell basis (`cnot(0,1); h(0)`),
//! 4. apply the corrections.
//!
//! SimQ has no mid-circuit measurement yet, so instead of measuring qubits
//! 0–1 and classically conditioning X/Z on the outcomes, this uses the
//! standard unitary equivalent from the deferred-measurement principle:
//! the classically-controlled X becomes `cnot(1,2)` and the
//! classically-controlled Z becomes `cz(0,2)`. Same statistics, no
//! classical feed-forward needed.

use simq::QuantumCircuit;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let angle = 0.7f64;
    // RY(0.7)|0⟩ has P(1) = sin²(0.7/2) = sin²(0.35).
    let expected_p1 = (angle / 2.0).sin().powi(2);

    // Reference: the prepared state's own statistics.
    let mut prep = QuantumCircuit::new(1);
    prep.ry(angle, 0);
    let p1_before = prep.probabilities()?[1];
    println!("Prepared state P(q0=1): {p1_before:.6} (theory: {expected_p1:.6})");
    assert!((p1_before - expected_p1).abs() < 1e-10);

    // Teleportation circuit.
    let mut qc = QuantumCircuit::new(3);
    qc.ry(angle, 0); // 1. arbitrary state on qubit 0
    qc.h(1).cnot(1, 2); // 2. Bell pair on qubits 1-2
    qc.cnot(0, 1).h(0); // 3. Bell-basis rotation on qubits 0-1
    qc.cnot(1, 2).cz(0, 2); // 4. corrections (deferred measurement)

    // Qubit 2's reduced statistics must reproduce the prepared state:
    // P(q2=1) = sin²(0.35), while qubits 0-1 are left maximally mixed.
    let probs = qc.probabilities()?;
    let p_q2_1: f64 = probs
        .iter()
        .enumerate()
        .filter(|(i, _)| (i >> 2) & 1 == 1)
        .map(|(_, &p)| p)
        .sum();
    println!("Teleported  P(q2=1): {p_q2_1:.6} (theory: {expected_p1:.6})");
    assert!(
        (p_q2_1 - expected_p1).abs() < 1e-10,
        "qubit 2 does not carry the teleported state"
    );

    // Qubits 0 and 1 each look completely random — the original state is
    // gone from the sender's side (no cloning happened).
    for q in [0, 1] {
        let p1: f64 = probs
            .iter()
            .enumerate()
            .filter(|(i, _)| (i >> q) & 1 == 1)
            .map(|(_, &p)| p)
            .sum();
        println!("Sender qubit {q} P(1): {p1:.6} (maximally mixed: 0.5)");
        assert!((p1 - 0.5).abs() < 1e-10);
    }
    println!("\nTeleportation succeeded: qubit 2 reproduces the prepared statistics.");

    Ok(())
}
