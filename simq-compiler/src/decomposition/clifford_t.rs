//! Clifford+T decomposition for fault-tolerant quantum computing
//!
//! This module provides decomposition of arbitrary quantum gates into the Clifford+T
//! gate set, which is the standard basis for fault-tolerant quantum computation.
//!
//! # The Clifford+T Gate Set
//!
//! The Clifford+T basis consists of:
//! - **H** (Hadamard): Creates superposition
//! - **S** (Phase): √Z rotation by π/2
//! - **T** (π/8 gate): √S rotation by π/4
//! - **CNOT**: Two-qubit entanglement
//!
//! This gate set is universal for quantum computation and is particularly important
//! for fault-tolerant quantum computing because:
//!
//! 1. Clifford gates (H, S, CNOT) can be implemented fault-tolerantly with low overhead
//! 2. T gates are expensive but can be implemented using magic state distillation
//! 3. The T-count (number of T gates) determines the resource requirements
//!
//! # Decomposition Methods
//!
//! ## Exact Synthesis
//!
//! For certain angles (multiples of π/8), exact Clifford+T synthesis is possible:
//! - **Gridsynth**: Optimal single-qubit synthesis (Ross & Selinger 2016)
//! - **Number-theoretic methods**: Uses continued fractions and algebraic number theory
//!
//! ## Approximate Synthesis
//!
//! For arbitrary angles, we use approximation algorithms:
//! - **Solovay-Kitaev**: Universal approximation, O(log^c(1/ε)) gates for precision ε
//! - **Quantum Shannon Decomposition**: Systematic approach with known bounds
//!
//! # T-Count Optimization
//!
//! The goal is to minimize the number of T gates while maintaining fidelity:
//! - T-count is the primary cost metric in fault-tolerant quantum computing
//! - T-depth (number of layers of T gates) affects circuit runtime
//! - Tradeoffs exist between T-count, T-depth, and auxiliary gates
//!
//! # References
//!
//! - Ross & Selinger, "Optimal ancilla-free Clifford+T approximation of z-rotations" (2016)
//! - Kliuchnikov, Maslov, Mosca, "Fast and efficient exact synthesis of single-qubit unitaries" (2013)
//! - Amy, Maslov, Mosca, "Polynomial-time T-depth optimization of Clifford+T circuits" (2014)
//! - Solovay-Kitaev theorem for universal approximation

use crate::decomposition::{
    Decomposer, DecompositionConfig, DecompositionMetadata, DecompositionResult,
};
use crate::matrix_computation::{is_unitary_2x2, Matrix2};
use num_complex::Complex64;
use simq_core::{Gate, QuantumError, Result};
use std::f64::consts::PI;
use std::sync::Arc;

const EPSILON: f64 = 1e-10;

/// Configuration for Clifford+T synthesis
#[derive(Debug, Clone)]
pub struct GridSynthConfig {
    /// Target approximation error (epsilon)
    pub epsilon: f64,

    /// Maximum number of gates to generate
    pub max_gates: usize,

    /// Whether to optimize for T-count
    pub optimize_t_count: bool,

    /// Whether to optimize for T-depth
    pub optimize_t_depth: bool,
}

impl Default for GridSynthConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-10,
            max_gates: 10000,
            optimize_t_count: true,
            optimize_t_depth: false,
        }
    }
}

/// Clifford+T decomposer
pub struct CliffordTDecomposer {
    config: GridSynthConfig,
}

impl CliffordTDecomposer {
    /// Create a new Clifford+T decomposer with default configuration
    pub fn new() -> Self {
        Self {
            config: GridSynthConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: GridSynthConfig) -> Self {
        Self { config }
    }

    /// Decompose a rotation around Z-axis into Clifford+T gates
    ///
    /// Succeeds only for angles that are multiples of π/4 (within
    /// `config.epsilon`), which decompose exactly:
    /// - Rz(π/4) = T
    /// - Rz(π/2) = S
    /// - Rz(π) = Z = S²
    ///
    /// # Errors
    ///
    /// Returns an error for any other angle: exact Clifford+T synthesis of
    /// arbitrary rotations (gridsynth) is not implemented, and silently
    /// substituting the nearest π/4 multiple would change the circuit's
    /// semantics (angle error up to π/8). Callers that explicitly want that
    /// approximation should use [`decompose_rz_approx`](Self::decompose_rz_approx),
    /// which reports the incurred angle error.
    pub fn decompose_rz(&self, angle: f64) -> Result<Vec<CliffordTGate>> {
        // Normalize angle to [0, 2π)
        let normalized_angle = angle.rem_euclid(2.0 * PI);

        // Check for exact multiples of π/4
        let k = (normalized_angle / (PI / 4.0)).round();
        if (normalized_angle - k * PI / 4.0).abs() < self.config.epsilon {
            return Ok(self.exact_rz_pi_over_4(k as i32));
        }

        Err(QuantumError::ValidationError(format!(
            "Rz({angle}) is not a multiple of π/4 (within epsilon = {}); exact Clifford+T \
             synthesis of arbitrary angles is not implemented. Use decompose_rz_approx() to \
             explicitly opt into nearest-π/4 approximation with a reported angle error.",
            self.config.epsilon
        )))
    }

    /// Approximate Rz(θ) by the nearest multiple of π/4
    ///
    /// This is an explicit opt-in to lossy decomposition. Returns the gate
    /// sequence together with the signed angle error θ_normalized − k·π/4
    /// actually incurred (up to ±π/8), so callers can decide whether the
    /// approximation is acceptable.
    pub fn decompose_rz_approx(&self, angle: f64) -> (Vec<CliffordTGate>, f64) {
        let normalized_angle = angle.rem_euclid(2.0 * PI);
        let k = (normalized_angle / (PI / 4.0)).round();
        let angle_error = normalized_angle - k * PI / 4.0;
        (self.exact_rz_pi_over_4(k as i32), angle_error)
    }

    /// Exact decomposition for Rz(k·π/4) where k is an integer
    fn exact_rz_pi_over_4(&self, k: i32) -> Vec<CliffordTGate> {
        // Normalize k to [0, 8)
        let k = k.rem_euclid(8);

        match k {
            0 => vec![], // Identity
            1 => vec![CliffordTGate::T],
            2 => vec![CliffordTGate::S],
            3 => vec![CliffordTGate::T, CliffordTGate::S],
            4 => vec![CliffordTGate::Z],
            5 => vec![CliffordTGate::TDagger],
            6 => vec![CliffordTGate::SDagger],
            7 => vec![CliffordTGate::TDagger, CliffordTGate::SDagger],
            _ => vec![],
        }
    }

    /// Decompose arbitrary single-qubit unitary into Clifford+T gates
    ///
    /// Uses the decomposition: U = Rz(α)Ry(β)Rz(γ)
    /// Then approximates each Ry and Rz using Clifford+T gates.
    pub fn decompose_single_qubit(&self, matrix: &Matrix2) -> Result<Vec<CliffordTGate>> {
        if !is_unitary_2x2(matrix) {
            return Err(QuantumError::ValidationError("Matrix is not unitary".to_string()));
        }

        // Extract Euler angles (ZYZ decomposition)
        // U = Rz(α)Ry(β)Rz(γ)

        let gamma = 2.0 * matrix[0][0].norm().acos();

        let alpha = if gamma.abs() < EPSILON {
            0.0
        } else {
            let sin_half = (gamma / 2.0).sin();
            (matrix[1][0] / Complex64::new(0.0, sin_half)).arg()
        };

        let delta = if gamma.abs() < EPSILON {
            (matrix[0][1] / Complex64::new(0.0, 1.0)).arg()
        } else {
            let sin_half = (gamma / 2.0).sin();
            (matrix[0][1] / Complex64::new(0.0, -sin_half)).arg()
        };

        let mut gates = Vec::new();

        // Decompose first Rz(alpha)
        gates.extend(self.decompose_rz(alpha)?);

        // Decompose Ry(gamma) = H Rz(gamma) H
        if gamma.abs() > EPSILON {
            gates.push(CliffordTGate::H);
            gates.extend(self.decompose_rz(gamma)?);
            gates.push(CliffordTGate::H);
        }

        // Decompose last Rz(delta)
        gates.extend(self.decompose_rz(delta)?);

        Ok(gates)
    }

    /// Solovay-Kitaev algorithm for universal approximation
    ///
    /// Recursively approximates any single-qubit unitary using Clifford+T gates.
    /// Achieves precision ε using O(log^c(1/ε)) gates where c ≈ 3.97.
    ///
    /// Only `depth == 0` (direct exact synthesis for π/4 multiples) is
    /// implemented. Higher depths error instead of silently returning the
    /// depth-0 result with unmet precision.
    pub fn solovay_kitaev(&self, matrix: &Matrix2, depth: usize) -> Result<Vec<CliffordTGate>> {
        if depth == 0 {
            // Base case: use direct decomposition
            return self.decompose_single_qubit(matrix);
        }

        Err(QuantumError::ValidationError(format!(
            "Solovay-Kitaev recursion (depth {depth}) is not implemented; \
             only depth 0 direct synthesis of π/4-multiple rotations is supported"
        )))
    }

    /// Count T gates in a gate sequence
    pub fn count_t_gates(gates: &[CliffordTGate]) -> usize {
        gates
            .iter()
            .filter(|g| matches!(g, CliffordTGate::T | CliffordTGate::TDagger))
            .count()
    }

    /// Count T-depth (number of layers of T gates)
    pub fn count_t_depth(gates: &[CliffordTGate]) -> usize {
        let mut depth = 0;
        let mut in_t_layer = false;

        for gate in gates {
            match gate {
                CliffordTGate::T | CliffordTGate::TDagger => {
                    if !in_t_layer {
                        depth += 1;
                        in_t_layer = true;
                    }
                },
                _ => {
                    in_t_layer = false;
                },
            }
        }

        depth
    }

    /// Optimize gate sequence to reduce T-count
    ///
    /// Applies circuit identities and optimizations:
    /// - T† T = I
    /// - SS = Z
    /// - HSHS = X
    pub fn optimize_t_count(&self, gates: &[CliffordTGate]) -> Vec<CliffordTGate> {
        let mut optimized = Vec::new();

        for gate in gates {
            match (optimized.last(), gate) {
                // T† T = I (cancel)
                (Some(&CliffordTGate::TDagger), CliffordTGate::T)
                | (Some(&CliffordTGate::T), CliffordTGate::TDagger) => {
                    optimized.pop();
                },

                // S† S = I (cancel)
                (Some(&CliffordTGate::SDagger), CliffordTGate::S)
                | (Some(&CliffordTGate::S), CliffordTGate::SDagger) => {
                    optimized.pop();
                },

                // HH = I (cancel)
                (Some(&CliffordTGate::H), CliffordTGate::H) => {
                    optimized.pop();
                },

                // SS = Z
                (Some(&CliffordTGate::S), CliffordTGate::S) => {
                    optimized.pop();
                    optimized.push(CliffordTGate::Z);
                },

                // Otherwise, add gate
                _ => {
                    optimized.push(*gate);
                },
            }
        }

        optimized
    }

    /// Convert intermediate Clifford+T gates to concrete gate objects.
    pub fn ct_gates_to_gates(ct_gates: &[CliffordTGate]) -> Vec<Arc<dyn Gate>> {
        use simq_gates::{
            Hadamard, PauliX, PauliY, PauliZ, SGate, SGateDagger, TGate, TGateDagger,
        };

        ct_gates
            .iter()
            .filter_map(|g| -> Option<Arc<dyn Gate>> {
                match g {
                    CliffordTGate::H => Some(Arc::new(Hadamard)),
                    CliffordTGate::S => Some(Arc::new(SGate)),
                    CliffordTGate::SDagger => Some(Arc::new(SGateDagger)),
                    CliffordTGate::T => Some(Arc::new(TGate)),
                    CliffordTGate::TDagger => Some(Arc::new(TGateDagger)),
                    CliffordTGate::X => Some(Arc::new(PauliX)),
                    CliffordTGate::Y => Some(Arc::new(PauliY)),
                    CliffordTGate::Z => Some(Arc::new(PauliZ)),
                    // Identity needs no physical gate; drop it so an
                    // identity decomposition honestly yields zero gates.
                    CliffordTGate::I => None,
                    // CNOT is two-qubit and cannot appear in the
                    // single-qubit path; drop rather than misrepresent.
                    CliffordTGate::CNOT => None,
                }
            })
            .collect()
    }
}

impl Default for CliffordTDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

impl Decomposer for CliffordTDecomposer {
    fn decompose(
        &self,
        gate: &dyn Gate,
        config: &DecompositionConfig,
    ) -> Result<DecompositionResult> {
        if gate.num_qubits() != 1 {
            return Err(QuantumError::ValidationError(
                "Clifford+T decomposition currently supports single-qubit gates only".to_string(),
            ));
        }

        // Get gate matrix
        let matrix = gate.matrix().ok_or_else(|| {
            QuantumError::ValidationError("Gate does not provide matrix representation".to_string())
        })?;

        if matrix.len() != 4 {
            return Err(QuantumError::ValidationError("Invalid matrix size".to_string()));
        }

        // Convert to Matrix2 format
        let matrix_2x2: Matrix2 = [
            [
                Complex64::new(matrix[0].re, matrix[0].im),
                Complex64::new(matrix[1].re, matrix[1].im),
            ],
            [
                Complex64::new(matrix[2].re, matrix[2].im),
                Complex64::new(matrix[3].re, matrix[3].im),
            ],
        ];

        // Decompose to Clifford+T gates
        let mut ct_gates = self.decompose_single_qubit(&matrix_2x2)?;

        // Optimize if requested
        if config.optimization_level > 0 {
            ct_gates = self.optimize_t_count(&ct_gates);
        }

        let t_count = Self::count_t_gates(&ct_gates);
        let t_depth = Self::count_t_depth(&ct_gates);

        // Convert to concrete gate objects. Empty is only valid for the
        // identity (no gates needed); otherwise an empty sequence with
        // near-unit fidelity would silently delete the gate.
        let gates = Self::ct_gates_to_gates(&ct_gates);
        if gates.is_empty() && !ct_gates.is_empty() {
            return Err(QuantumError::ValidationError(
                "Clifford+T conversion produced no gates for a non-empty sequence".to_string(),
            ));
        }

        Ok(DecompositionResult {
            gates,
            fidelity: 1.0 - self.config.epsilon,
            depth: ct_gates.len(),
            gate_count: ct_gates.len(),
            two_qubit_count: 0,
            metadata: DecompositionMetadata {
                strategy: format!("Clifford+T (T-count: {}, T-depth: {})", t_count, t_depth),
                optimized: config.optimization_level > 0,
                optimization_passes: config.optimization_level as usize,
                original_gate_count: 1,
            },
        })
    }

    /// Check whether this decomposer may handle the gate.
    ///
    /// Shape check only (1 qubit + matrix present). `decompose` can still
    /// return `Err` for rotations that are not multiples of π/4, which have
    /// no exact Clifford+T synthesis; callers must handle that case.
    fn can_decompose(&self, gate: &dyn Gate) -> bool {
        gate.num_qubits() == 1 && gate.matrix().is_some()
    }

    fn name(&self) -> &str {
        "CliffordT"
    }

    fn estimate_cost(&self, gate: &dyn Gate) -> Option<usize> {
        if gate.num_qubits() == 1 {
            // Rough estimate: average 5-10 gates for arbitrary single-qubit
            Some(7)
        } else {
            None
        }
    }
}

/// Clifford+T gate enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliffordTGate {
    /// Hadamard gate
    H,

    /// S gate (√Z)
    S,

    /// S† gate
    SDagger,

    /// T gate (√S)
    T,

    /// T† gate
    TDagger,

    /// Pauli-X gate
    X,

    /// Pauli-Y gate
    Y,

    /// Pauli-Z gate
    Z,

    /// Identity (sometimes needed for bookkeeping)
    I,

    /// CNOT (two-qubit gate)
    CNOT,
}

impl CliffordTGate {
    /// Check if this is a Clifford gate (not T)
    pub fn is_clifford(&self) -> bool {
        !matches!(self, CliffordTGate::T | CliffordTGate::TDagger)
    }

    /// Check if this is a T gate (expensive in fault-tolerant QC)
    pub fn is_t_gate(&self) -> bool {
        matches!(self, CliffordTGate::T | CliffordTGate::TDagger)
    }

    /// Get the name of this gate
    pub fn name(&self) -> &str {
        match self {
            CliffordTGate::H => "H",
            CliffordTGate::S => "S",
            CliffordTGate::SDagger => "S†",
            CliffordTGate::T => "T",
            CliffordTGate::TDagger => "T†",
            CliffordTGate::X => "X",
            CliffordTGate::Y => "Y",
            CliffordTGate::Z => "Z",
            CliffordTGate::I => "I",
            CliffordTGate::CNOT => "CNOT",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_rz_decomposition() {
        let decomposer = CliffordTDecomposer::new();

        // Rz(π/4) = T
        let gates = decomposer.decompose_rz(PI / 4.0).unwrap();
        assert_eq!(gates, vec![CliffordTGate::T]);

        // Rz(π/2) = S
        let gates = decomposer.decompose_rz(PI / 2.0).unwrap();
        assert_eq!(gates, vec![CliffordTGate::S]);

        // Rz(π) = Z
        let gates = decomposer.decompose_rz(PI).unwrap();
        assert_eq!(gates, vec![CliffordTGate::Z]);
    }

    #[test]
    fn test_t_count() {
        let gates = vec![
            CliffordTGate::H,
            CliffordTGate::T,
            CliffordTGate::S,
            CliffordTGate::T,
            CliffordTGate::TDagger,
        ];

        assert_eq!(CliffordTDecomposer::count_t_gates(&gates), 3);
    }

    #[test]
    fn test_t_depth() {
        let gates = vec![
            CliffordTGate::T,
            CliffordTGate::T, // Same layer
            CliffordTGate::H, // Clifford
            CliffordTGate::T, // New layer
        ];

        assert_eq!(CliffordTDecomposer::count_t_depth(&gates), 2);
    }

    #[test]
    fn test_optimization() {
        let decomposer = CliffordTDecomposer::new();

        let gates = vec![
            CliffordTGate::T,
            CliffordTGate::TDagger, // Should cancel
            CliffordTGate::H,
            CliffordTGate::H, // Should cancel
        ];

        let optimized = decomposer.optimize_t_count(&gates);
        assert!(optimized.is_empty());
    }

    #[test]
    fn test_is_clifford() {
        assert!(CliffordTGate::H.is_clifford());
        assert!(CliffordTGate::S.is_clifford());
        assert!(!CliffordTGate::T.is_clifford());
        assert!(!CliffordTGate::TDagger.is_clifford());
    }

    #[test]
    fn test_is_t_gate() {
        assert!(CliffordTGate::T.is_t_gate());
        assert!(CliffordTGate::TDagger.is_t_gate());
        assert!(!CliffordTGate::H.is_t_gate());
        assert!(!CliffordTGate::S.is_t_gate());
    }

    #[derive(Debug)]
    struct MockGate {
        name: String,
        n_qubits: usize,
        matrix: Option<Vec<Complex64>>,
    }

    impl Gate for MockGate {
        fn name(&self) -> &str {
            &self.name
        }
        fn num_qubits(&self) -> usize {
            self.n_qubits
        }
        fn matrix(&self) -> Option<Vec<Complex64>> {
            self.matrix.clone()
        }
    }

    fn matrix2_to_flat(m: &Matrix2) -> Vec<Complex64> {
        vec![m[0][0], m[0][1], m[1][0], m[1][1]]
    }

    fn identity_matrix2() -> Matrix2 {
        [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ]
    }

    fn hadamard_matrix2() -> Matrix2 {
        let f = std::f64::consts::FRAC_1_SQRT_2;
        [
            [Complex64::new(f, 0.0), Complex64::new(f, 0.0)],
            [Complex64::new(f, 0.0), Complex64::new(-f, 0.0)],
        ]
    }

    #[test]
    fn test_exact_rz_pi_over_4_all_residues() {
        let decomposer = CliffordTDecomposer::new();

        // k=0 => identity (empty vec)
        assert_eq!(decomposer.exact_rz_pi_over_4(0), vec![]);
        // k=1 => T
        assert_eq!(decomposer.exact_rz_pi_over_4(1), vec![CliffordTGate::T]);
        // k=2 => S
        assert_eq!(decomposer.exact_rz_pi_over_4(2), vec![CliffordTGate::S]);
        // k=3 => T, S
        assert_eq!(decomposer.exact_rz_pi_over_4(3), vec![CliffordTGate::T, CliffordTGate::S]);
        // k=4 => Z
        assert_eq!(decomposer.exact_rz_pi_over_4(4), vec![CliffordTGate::Z]);
        // k=5 => TDagger
        assert_eq!(decomposer.exact_rz_pi_over_4(5), vec![CliffordTGate::TDagger]);
        // k=6 => SDagger
        assert_eq!(decomposer.exact_rz_pi_over_4(6), vec![CliffordTGate::SDagger]);
        // k=7 => TDagger, SDagger
        assert_eq!(
            decomposer.exact_rz_pi_over_4(7),
            vec![CliffordTGate::TDagger, CliffordTGate::SDagger]
        );

        // Negative k wraps around via rem_euclid: -1 -> 7
        assert_eq!(
            decomposer.exact_rz_pi_over_4(-1),
            vec![CliffordTGate::TDagger, CliffordTGate::SDagger]
        );

        // k=8 wraps to 0
        assert_eq!(decomposer.exact_rz_pi_over_4(8), vec![]);
    }

    #[test]
    fn test_decompose_rz_arbitrary_angle_errors() {
        let decomposer = CliffordTDecomposer::new();
        // An angle that is not a k*pi/4 multiple within epsilon must be
        // rejected rather than silently snapped to the nearest multiple.
        let angle = PI / 4.0 + 0.2;
        assert!(decomposer.decompose_rz(angle).is_err());
    }

    #[test]
    fn test_decompose_rz_approx_reports_error() {
        let decomposer = CliffordTDecomposer::new();
        let angle = PI / 4.0 + 0.2;
        let (gates, angle_error) = decomposer.decompose_rz_approx(angle);
        // Snaps to the nearest pi/4 multiple...
        let k = (angle / (PI / 4.0)).round() as i32;
        assert_eq!(gates, decomposer.exact_rz_pi_over_4(k));
        // ...and reports exactly how far off that is.
        assert!((angle_error - 0.2).abs() < 1e-12, "angle_error = {angle_error}");

        // Exact angles report zero error
        let (gates_exact, err_exact) = decomposer.decompose_rz_approx(PI / 2.0);
        assert_eq!(gates_exact, vec![CliffordTGate::S]);
        assert!(err_exact.abs() < 1e-12);
    }

    #[test]
    fn test_decompose_single_qubit_identity_gamma_near_zero() {
        // Identity matrix triggers the gamma.abs() < EPSILON branch inside decompose_single_qubit
        let decomposer = CliffordTDecomposer::new();
        let id = identity_matrix2();
        let result = decomposer.decompose_single_qubit(&id);
        assert!(result.is_ok());
    }

    #[test]
    fn test_decompose_single_qubit_non_unitary_errors() {
        let decomposer = CliffordTDecomposer::new();
        let not_unitary: Matrix2 = [
            [Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        let result = decomposer.decompose_single_qubit(&not_unitary);
        assert!(result.is_err());
    }

    #[test]
    fn test_solovay_kitaev_base_case_and_recursive_error() {
        let decomposer = CliffordTDecomposer::new();
        let h = hadamard_matrix2();

        // depth == 0 -> base case (direct decomposition)
        let base = decomposer.solovay_kitaev(&h, 0).unwrap();
        assert!(!base.is_empty());

        // depth > 0 -> recursion unimplemented, must error rather than
        // silently return the depth-0 result with unmet precision.
        assert!(decomposer.solovay_kitaev(&h, 3).is_err());
    }

    #[test]
    fn test_optimize_t_count_s_dagger_s_and_ss_cancellation() {
        let decomposer = CliffordTDecomposer::new();

        // S† S should cancel (and its symmetric S S† case)
        let gates = vec![CliffordTGate::SDagger, CliffordTGate::S];
        assert!(decomposer.optimize_t_count(&gates).is_empty());

        let gates2 = vec![CliffordTGate::S, CliffordTGate::SDagger];
        assert!(decomposer.optimize_t_count(&gates2).is_empty());

        // S S should become Z
        let gates3 = vec![CliffordTGate::S, CliffordTGate::S];
        assert_eq!(decomposer.optimize_t_count(&gates3), vec![CliffordTGate::Z]);

        // A gate with no matching optimization rule (default arm) should just be appended
        let gates4 = vec![CliffordTGate::X, CliffordTGate::Y];
        assert_eq!(decomposer.optimize_t_count(&gates4), vec![CliffordTGate::X, CliffordTGate::Y]);
    }

    #[test]
    fn test_decomposer_trait_rejects_non_single_qubit_gate() {
        let decomposer = CliffordTDecomposer::new();
        let config = DecompositionConfig::default();
        let two_qubit_gate = MockGate {
            name: "CNOT".to_string(),
            n_qubits: 2,
            matrix: None,
        };
        let result = decomposer.decompose(&two_qubit_gate, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_decomposer_trait_rejects_gate_without_matrix() {
        let decomposer = CliffordTDecomposer::new();
        let config = DecompositionConfig::default();
        let gate_no_matrix = MockGate {
            name: "H".to_string(),
            n_qubits: 1,
            matrix: None,
        };
        let result = decomposer.decompose(&gate_no_matrix, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_decomposer_trait_rejects_invalid_matrix_size() {
        let decomposer = CliffordTDecomposer::new();
        let config = DecompositionConfig::default();
        let gate_bad_matrix = MockGate {
            name: "H".to_string(),
            n_qubits: 1,
            matrix: Some(vec![Complex64::new(1.0, 0.0)]),
        };
        let result = decomposer.decompose(&gate_bad_matrix, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_decomposer_trait_succeeds_and_optimizes() {
        let decomposer = CliffordTDecomposer::new();
        let config = DecompositionConfig {
            optimization_level: 1,
            ..Default::default()
        };

        let gate = MockGate {
            name: "H".to_string(),
            n_qubits: 1,
            matrix: Some(matrix2_to_flat(&hadamard_matrix2())),
        };
        let result = decomposer.decompose(&gate, &config).unwrap();
        assert!(result.metadata.optimized);
        assert!(result.metadata.strategy.contains("Clifford+T"));
        assert_eq!(result.two_qubit_count, 0);
    }

    #[test]
    fn test_decomposer_trait_can_decompose_name_and_estimate_cost() {
        let decomposer = CliffordTDecomposer::new();
        assert_eq!(decomposer.name(), "CliffordT");

        let single_with_matrix = MockGate {
            name: "H".to_string(),
            n_qubits: 1,
            matrix: Some(matrix2_to_flat(&hadamard_matrix2())),
        };
        assert!(decomposer.can_decompose(&single_with_matrix));
        assert_eq!(decomposer.estimate_cost(&single_with_matrix), Some(7));

        let single_without_matrix = MockGate {
            name: "H".to_string(),
            n_qubits: 1,
            matrix: None,
        };
        assert!(!decomposer.can_decompose(&single_without_matrix));

        let two_qubit = MockGate {
            name: "CNOT".to_string(),
            n_qubits: 2,
            matrix: None,
        };
        assert!(!decomposer.can_decompose(&two_qubit));
        assert_eq!(decomposer.estimate_cost(&two_qubit), None);
    }

    #[test]
    fn test_clifford_t_gate_name_all_variants() {
        let expected = [
            (CliffordTGate::H, "H"),
            (CliffordTGate::S, "S"),
            (CliffordTGate::SDagger, "S†"),
            (CliffordTGate::T, "T"),
            (CliffordTGate::TDagger, "T†"),
            (CliffordTGate::X, "X"),
            (CliffordTGate::Y, "Y"),
            (CliffordTGate::Z, "Z"),
            (CliffordTGate::I, "I"),
            (CliffordTGate::CNOT, "CNOT"),
        ];

        for (gate, name) in expected {
            assert_eq!(gate.name(), name, "unexpected name for {:?}", gate);
        }
    }
}
