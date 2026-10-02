//! Grover's search on 3 qubits
//!
//! Run with: cargo run -p simq --example grover
//!
//! Grover's algorithm finds a marked item in an unstructured database of
//! N = 2^n entries with ~√N queries instead of ~N. On 3 qubits (8 entries)
//! two iterations already push the marked state above 94% probability.
//!
//! The two building blocks:
//! - **Oracle**: flips the phase of the marked state only. A phase flip is
//!   a bit flip conjugated by Hadamards, and H·CCX·H = CCZ, so the oracle
//!   is X-gates (to map the marked state onto |111⟩), H·Toffoli·H on the
//!   target, X-gates to undo.
//! - **Diffusion** ("inversion about the mean"): H on all qubits, X on all
//!   qubits, CCZ, X on all, H on all. This reflects every amplitude about
//!   the average, amplifying the marked state the oracle singled out.

use simq::QuantumCircuit;

/// Phase-flip the |111⟩ state: H·CCX·H = CCZ.
fn ccz(qc: &mut QuantumCircuit) {
    qc.h(2).toffoli(0, 1, 2).h(2);
}

/// Oracle for the marked state |101⟩: X-conjugate qubit 1 (whose marked
/// bit is 0) so the marked state looks like |111⟩, apply CCZ, undo.
fn oracle(qc: &mut QuantumCircuit) {
    qc.x(1);
    ccz(qc);
    qc.x(1);
}

/// Diffusion operator: inversion about the mean on all 3 qubits.
fn diffusion(qc: &mut QuantumCircuit) {
    qc.h(0).h(1).h(2);
    qc.x(0).x(1).x(2);
    ccz(qc);
    qc.x(0).x(1).x(2);
    qc.h(0).h(1).h(2);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Marked state |101⟩ = index 5 (qubit 0 set, qubit 1 clear, qubit 2 set).
    let marked = 5;

    // Uniform superposition, then 2 Grover iterations.
    let mut qc = QuantumCircuit::new(3);
    qc.h(0).h(1).h(2);
    for _ in 0..2 {
        oracle(&mut qc);
        diffusion(&mut qc);
    }

    // Exact probability vs theory: after k iterations P = sin²((2k+1)·θ)
    // with θ = asin(1/√N), N = 8.
    let theta = (1.0f64 / 8.0).sqrt().asin();
    let theory = ((2.0 * 2.0 + 1.0) * theta).sin().powi(2);
    let p_marked = qc.probabilities()?[marked];
    println!("P(|101⟩) after 2 iterations: {p_marked:.6} (theory: {theory:.6})");
    assert!((p_marked - theory).abs() < 1e-10, "Grover probability does not match theory");
    assert!(p_marked > 0.94, "2 iterations should exceed 94%");

    // Sampling view: the marked state dominates the counts.
    let counts = qc.simulate_with_shots(1024)?.measurements.unwrap();
    println!("\nCounts over 1024 shots:");
    for (bitstring, count) in counts.sorted() {
        println!("  |{bitstring}⟩: {count}");
    }

    Ok(())
}
