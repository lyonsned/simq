//! Quantum Fourier Transform (QFT) example
//!
//! Run with: cargo run -p simq --example qft
//!
//! The QFT is the quantum analogue of the discrete Fourier transform and
//! the engine inside phase estimation and Shor's algorithm. This example:
//! 1. builds the textbook QFT circuit (`h` + controlled phases + reversal swaps),
//! 2. checks its output amplitudes against the analytic DFT of a basis input,
//! 3. shows the "aha" moment: a periodic superposition collapses onto sharp
//!    frequency peaks after the QFT.

use simq::core::Complex64;
use simq::QuantumCircuit;
use std::f64::consts::PI;

/// Append the QFT on `n` qubits (qubits `0..n`) to `qc`.
///
/// Textbook pattern (Nielsen & Chuang, Fig. 5.1), mirrored for SimQ's
/// little-endian wires: the textbook draws the most significant qubit on
/// the top wire, but here qubit 0 is the *least* significant bit, so the
/// "first" qubit of the textbook circuit is qubit `n - 1`. For each qubit
/// `j` from top to bottom: a Hadamard, then controlled phases
/// `cp(π/2^(j-l), l, j)` from every lower qubit `l < j`. The trailing swaps
/// undo the bit-reversed output order the raw circuit produces, so the
/// result is the plain DFT: QFT|x⟩ = Σ_y exp(2πixy/N)/√N |y⟩ with `x` and
/// `y` read as little-endian integers.
fn qft(qc: &mut QuantumCircuit, n: usize) {
    for j in (0..n).rev() {
        qc.h(j);
        for l in 0..j {
            qc.cp(PI / 2f64.powi((j - l) as i32), l, j);
        }
    }
    for j in 0..n / 2 {
        qc.swap(j, n - 1 - j);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 4;
    let dim = 1 << n;

    // --- Part 1: QFT of a basis state matches the analytic DFT ---
    // Prepare |5⟩ = |0101⟩, i.e. X on qubits 0 and 2.
    let mut qc = QuantumCircuit::new(n);
    qc.x(0).x(2);
    qft(&mut qc, n);

    let state = qc.simulate()?.state.to_dense_vec();
    // Analytic DFT: amplitude[y] = exp(2πi·x·y/N) / √N with x = 5, N = 16.
    let x = 5.0;
    let mut max_err: f64 = 0.0;
    for (y, amp) in state.iter().enumerate() {
        let phase = 2.0 * PI * x * y as f64 / dim as f64;
        let expected = Complex64::new(phase.cos(), phase.sin()) / (dim as f64).sqrt();
        max_err = max_err.max((amp - expected).norm());
    }
    println!("QFT|5⟩ max amplitude error vs analytic DFT: {max_err:.2e}");
    assert!(max_err < 1e-10, "QFT does not match the DFT");

    // A basis state spreads uniformly: every outcome has probability 1/16.
    let probs = qc.probabilities()?;
    let uniform = probs.iter().all(|&p| (p - 1.0 / dim as f64).abs() < 1e-10);
    assert!(uniform, "basis-state QFT should be uniform");
    println!("QFT of a basis state is uniform over all {dim} outcomes, as expected.");

    // --- Part 2: the "aha" moment — period finding ---
    // H on qubits 2 and 3 prepares (|0⟩ + |4⟩ + |8⟩ + |12⟩)/2, a signal
    // with period 4. Its Fourier transform is nonzero only at multiples
    // of N/period = 4: outcomes {0, 4, 8, 12}, each with probability 1/4.
    let mut periodic = QuantumCircuit::new(n);
    periodic.h(2).h(3);
    qft(&mut periodic, n);

    let freqs = periodic.probabilities()?;
    println!("\nQFT of period-4 superposition:");
    for (y, p) in freqs.iter().enumerate() {
        if *p > 1e-10 {
            println!("  |{y:04b}⟩ (y={y}): p = {p:.4}");
        }
    }
    for (y, p) in freqs.iter().enumerate() {
        let expected = if y % 4 == 0 { 0.25 } else { 0.0 };
        assert!((p - expected).abs() < 1e-10, "y={y}: expected p={expected}, got p={p}");
    }
    println!("Probability concentrates exactly on multiples of 4 — period finding works.");

    Ok(())
}
