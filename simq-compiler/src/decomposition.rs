//! Gate decomposition for quantum circuits
//!
//! This module provides comprehensive gate decomposition functionality for translating
//! quantum gates into different basis gate sets. Decomposition is essential for:
//!
//! - **Hardware compilation**: Mapping gates to native gate sets of quantum processors
//! - **Circuit optimization**: Expressing gates in fault-tolerant bases (e.g., Clifford+T)
//! - **Gate synthesis**: Breaking down complex gates into elementary operations
//! - **Cross-platform compatibility**: Translating between different quantum frameworks
//!
//! # Architecture
//!
//! The decomposition system is organized into several specialized modules:
//!
//! - [`basis`]: Defines basis gate sets (IBM, Google, IonQ, Clifford+T, etc.)
//! - [`single_qubit`]: Single-qubit gate decompositions (ZYZ, ZXZ, U3, etc.)
//! - [`two_qubit`]: Two-qubit gate decompositions (CNOT synthesis, CZ, SWAP)
//! - [`multi_qubit`]: Multi-controlled gate decompositions (Toffoli, Fredkin)
//! - [`clifford_t`]: Clifford+T decomposition for fault-tolerant quantum computing
//!
//! # Decomposition Strategies
//!
//! ## Single-Qubit Gates
//!
//! Any single-qubit unitary U can be decomposed using Euler angle decompositions:
//!
//! - **ZYZ**: U = e^(iα) Rz(β) Ry(γ) Rz(δ)
//! - **ZXZ**: U = e^(iα) Rz(β) Rx(γ) Rz(δ)
//! - **XYX**: U = e^(iα) Rx(β) Ry(γ) Rx(δ)
//! - **U3**: U = Rz(φ) Ry(θ) Rz(λ) (IBM's native gate)
//!
//! For specific basis gates like {H, T}:
//! - Solovay-Kitaev algorithm for approximation
//! - Gridsynth/Ross-Selinger for exact synthesis
//!
//! ## Two-Qubit Gates
//!
//! Two-qubit gates can be decomposed into CNOT + single-qubit gates:
//!
//! - **Canonical decomposition**: Any 2-qubit unitary requires at most 3 CNOTs
//! - **CZ decomposition**: CZ = H₁ CNOT H₁
//! - **SWAP decomposition**: SWAP = CNOT₀₁ CNOT₁₀ CNOT₀₁
//! - **iSWAP decomposition**: Uses CNOT + rotations
//!
//! ## Multi-Qubit Gates
//!
//! - **Toffoli (CCX)**: 6 CNOTs + T gates (Clifford+T)
//! - **Fredkin (CSWAP)**: Decomposed into Toffoli + CNOTs
//! - **Multi-controlled gates**: Linear or logarithmic decomposition using ancillas
//!
//! # Example
//!
//! ```ignore
//! use simq_compiler::decomposition::{DecompositionConfig, BasisGateSet, decompose_circuit};
//!
//! // Configure decomposition to IBM basis
//! let config = DecompositionConfig {
//!     basis: BasisGateSet::IBM,
//!     optimization_level: 2,
//!     fidelity_threshold: 0.9999,
//!     ..Default::default()
//! };
//!
//! // Decompose circuit
//! let decomposed = decompose_circuit(&circuit, &config)?;
//! ```
//!
//! # Performance
//!
//! - **Compile-time optimization**: Common angles use cached matrices
//! - **High fidelity**: Numerical precision < 10^-10 for exact decompositions
//! - **Optimized sequences**: Minimizes gate count and circuit depth
//!
//! # References
//!
//! - Nielsen & Chuang, "Quantum Computation and Quantum Information"
//! - Barenco et al., "Elementary gates for quantum computation" (1995)
//! - Shende, Bullock, Markov, "Synthesis of quantum-logic circuits" (2006)
//! - Solovay-Kitaev theorem for universal approximation
//! - Ross & Selinger, "Optimal ancilla-free Clifford+T approximation" (2016)

pub mod basis;
pub mod clifford_t;
pub mod multi_qubit;
pub mod single_qubit;
pub mod two_qubit;

use num_complex::Complex64;
use simq_core::{Gate, QuantumError, Result};
use std::sync::Arc;

// Re-export main types
pub use basis::{BasisGate, BasisGateSet};
pub use clifford_t::{CliffordTDecomposer, GridSynthConfig};
pub use multi_qubit::MultiQubitDecomposer;
pub use single_qubit::{EulerBasis, SingleQubitDecomposer};
pub use two_qubit::{EntanglementGate, TwoQubitDecomposer};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for gate decomposition
#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    /// Target basis gate set
    pub basis: BasisGateSet,

    /// Optimization level (0-3)
    /// - 0: No optimization, direct decomposition
    /// - 1: Basic optimization (merge adjacent rotations)
    /// - 2: Advanced optimization (circuit identities, commutation)
    /// - 3: Aggressive optimization (numerical search, Solovay-Kitaev)
    pub optimization_level: u8,

    /// Maximum circuit depth (None = unlimited)
    pub max_depth: Option<usize>,

    /// Minimum fidelity threshold (0.0 to 1.0)
    /// Decomposition fails if fidelity < threshold
    pub fidelity_threshold: f64,

    /// Maximum number of gates in decomposition (None = unlimited)
    pub max_gates: Option<usize>,

    /// Use ancilla qubits for optimization
    pub allow_ancillas: bool,

    /// Number of ancilla qubits available
    pub num_ancillas: usize,

    /// Precision for Clifford+T synthesis (epsilon)
    pub clifford_t_epsilon: f64,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            basis: BasisGateSet::Universal,
            optimization_level: 1,
            max_depth: None,
            fidelity_threshold: 0.9999,
            max_gates: None,
            allow_ancillas: false,
            num_ancillas: 0,
            clifford_t_epsilon: 1e-10,
        }
    }
}

// ============================================================================
// Decomposition Result
// ============================================================================

/// Result of gate decomposition
///
/// Note on qubit indices: `gates` carries gate *types* only. When a
/// decomposition is expressed with per-gate qubit positions (e.g.
/// `TwoQubitGateInstruction::CNOT { control, target }` or
/// `MultiQubitInstruction`), those indices are not preserved in the
/// returned objects and must be replayed positionally against the
/// instruction sequence by the caller. In particular, direction matters:
/// SWAP = `[CNOT01, CNOT10, CNOT01]` becomes `[CNot, CNot, CNot]` with the
/// direction living in the instruction order.
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    /// Sequence of gates in the decomposition
    pub gates: Vec<Arc<dyn Gate>>,

    /// Fidelity of the decomposition (1.0 = perfect)
    pub fidelity: f64,

    /// Circuit depth of the decomposition
    pub depth: usize,

    /// Total gate count
    pub gate_count: usize,

    /// Number of two-qubit gates (important metric)
    pub two_qubit_count: usize,

    /// Metadata about the decomposition
    pub metadata: DecompositionMetadata,
}

/// Metadata about the decomposition process
#[derive(Debug, Clone, Default)]
pub struct DecompositionMetadata {
    /// Which decomposition strategy was used
    pub strategy: String,

    /// Whether optimization was applied
    pub optimized: bool,

    /// Number of optimization passes applied
    pub optimization_passes: usize,

    /// Original gate count before optimization
    pub original_gate_count: usize,
}

// ============================================================================
// Main Decomposer Trait
// ============================================================================

/// Trait for gate decomposition strategies
pub trait Decomposer: Send + Sync {
    /// Decompose a gate into a sequence of basis gates
    fn decompose(
        &self,
        gate: &dyn Gate,
        config: &DecompositionConfig,
    ) -> Result<DecompositionResult>;

    /// Check if this decomposer can handle the given gate
    fn can_decompose(&self, gate: &dyn Gate) -> bool;

    /// Get the name of this decomposition strategy
    fn name(&self) -> &str;

    /// Estimate the cost of decomposition (gate count)
    fn estimate_cost(&self, _gate: &dyn Gate) -> Option<usize> {
        None
    }
}

// ============================================================================
// Universal Decomposer
// ============================================================================

/// Universal decomposer that dispatches to specialized decomposers
pub struct UniversalDecomposer {
    config: DecompositionConfig,
}

impl UniversalDecomposer {
    /// Create a new universal decomposer with given configuration
    pub fn new(config: DecompositionConfig) -> Self {
        Self { config }
    }

    /// Decompose a gate using the appropriate strategy
    pub fn decompose_gate(&self, gate: &dyn Gate) -> Result<DecompositionResult> {
        let num_qubits = gate.num_qubits();

        match num_qubits {
            1 => {
                let decomposer = SingleQubitDecomposer::new(EulerBasis::ZYZ);
                decomposer.decompose(gate, &self.config)
            },
            2 => {
                let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
                decomposer.decompose(gate, &self.config)
            },
            _ => {
                let decomposer = MultiQubitDecomposer::new();
                decomposer.decompose(gate, &self.config)
            },
        }
    }

    /// Decompose a sequence of gates
    pub fn decompose_gates(&self, gates: &[Arc<dyn Gate>]) -> Result<Vec<DecompositionResult>> {
        gates
            .iter()
            .map(|g| self.decompose_gate(g.as_ref()))
            .collect()
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Compute fidelity between two unitary matrices
///
/// Fidelity F = |Tr(U†V)| / d, where d is the matrix dimension
pub fn compute_fidelity(u: &[Vec<Complex64>], v: &[Vec<Complex64>]) -> Result<f64> {
    if u.len() != v.len() || u.is_empty() {
        return Err(QuantumError::ValidationError("Matrix dimensions mismatch".to_string()));
    }

    let dim = u.len();
    let mut trace = Complex64::new(0.0, 0.0);

    for i in 0..dim {
        for j in 0..dim {
            if j >= u[i].len() || j >= v[i].len() {
                return Err(QuantumError::ValidationError("Invalid matrix structure".to_string()));
            }
            trace += u[j][i].conj() * v[j][i];
        }
    }

    Ok((trace.norm() / dim as f64).min(1.0))
}

/// Validate that a decomposition meets the fidelity threshold
pub fn validate_decomposition(
    original: &[Vec<Complex64>],
    decomposed: &[Vec<Complex64>],
    threshold: f64,
) -> Result<f64> {
    let fidelity = compute_fidelity(original, decomposed)?;

    if fidelity < threshold {
        return Err(QuantumError::ValidationError(format!(
            "Decomposition fidelity {} below threshold {}",
            fidelity, threshold
        )));
    }

    Ok(fidelity)
}

/// Optimize a gate sequence by merging adjacent rotations
pub fn optimize_gate_sequence(gates: Vec<Arc<dyn Gate>>, level: u8) -> Vec<Arc<dyn Gate>> {
    if level == 0 {
        return gates;
    }

    // TODO: Implement gate fusion and optimization
    // For now, return gates as-is
    gates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = DecompositionConfig::default();
        assert_eq!(config.optimization_level, 1);
        assert!(config.fidelity_threshold > 0.999);
    }

    #[test]
    fn test_fidelity_identical() {
        let m1 = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        let m2 = m1.clone();

        let fidelity = compute_fidelity(&m1, &m2).unwrap();
        assert!((fidelity - 1.0).abs() < 1e-10);
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

    #[test]
    fn test_decomposer_default_estimate_cost_returns_none() {
        // A minimal Decomposer impl that only overrides the required methods,
        // exercising the trait's default estimate_cost() implementation.
        struct DummyDecomposer;
        impl Decomposer for DummyDecomposer {
            fn decompose(
                &self,
                _gate: &dyn Gate,
                _config: &DecompositionConfig,
            ) -> Result<DecompositionResult> {
                unimplemented!()
            }
            fn can_decompose(&self, _gate: &dyn Gate) -> bool {
                true
            }
            fn name(&self) -> &str {
                "Dummy"
            }
        }

        let dummy = DummyDecomposer;
        let gate = MockGate {
            name: "X".to_string(),
            n_qubits: 1,
            matrix: None,
        };
        assert_eq!(dummy.estimate_cost(&gate), None);
    }

    #[test]
    fn test_universal_decomposer_dispatches_single_qubit() {
        let config = DecompositionConfig::default();
        let decomposer = UniversalDecomposer::new(config);
        let gate = MockGate {
            name: "H".to_string(),
            n_qubits: 1,
            matrix: Some(vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ]),
        };
        let result = decomposer.decompose_gate(&gate).unwrap();
        // Identity matrix decomposes to zero gates; the invariant is that
        // gate_count tracks the real emitted sequence.
        assert_eq!(result.gate_count, result.gates.len());
    }

    #[test]
    fn test_universal_decomposer_dispatches_two_qubit() {
        let config = DecompositionConfig::default();
        let decomposer = UniversalDecomposer::new(config);
        let mut matrix = vec![Complex64::new(0.0, 0.0); 16];
        for i in 0..4 {
            matrix[i * 4 + i] = Complex64::new(1.0, 0.0);
        }
        let gate = MockGate {
            name: "CZ".to_string(),
            n_qubits: 2,
            matrix: Some(matrix),
        };
        let result = decomposer.decompose_gate(&gate).unwrap();
        // CZ -> H CNOT H: one entangling gate, real non-empty sequence.
        assert!(!result.gates.is_empty());
        assert_eq!(result.two_qubit_count, 1);
    }

    #[test]
    fn test_universal_decomposer_generic_two_qubit_errors() {
        let config = DecompositionConfig::default();
        let decomposer = UniversalDecomposer::new(config);
        let mut matrix = vec![Complex64::new(0.0, 0.0); 16];
        for i in 0..4 {
            matrix[i * 4 + i] = Complex64::new(1.0, 0.0);
        }
        let gate = MockGate {
            name: "CUSTOM2Q".to_string(),
            n_qubits: 2,
            matrix: Some(matrix),
        };
        assert!(decomposer.decompose_gate(&gate).is_err());
    }

    #[test]
    fn test_universal_decomposer_dispatches_multi_qubit() {
        let config = DecompositionConfig::default();
        let decomposer = UniversalDecomposer::new(config);
        let gate = MockGate {
            name: "Toffoli".to_string(),
            n_qubits: 3,
            matrix: None,
        };
        let result = decomposer.decompose_gate(&gate).unwrap();
        assert!(result.gate_count > 0);
    }

    #[test]
    fn test_compute_fidelity_rejects_dimension_mismatch() {
        let m1 = vec![vec![Complex64::new(1.0, 0.0)]];
        let m2 = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        assert!(compute_fidelity(&m1, &m2).is_err());
    }

    #[test]
    fn test_compute_fidelity_rejects_empty_matrices() {
        let m1: Vec<Vec<Complex64>> = vec![];
        let m2: Vec<Vec<Complex64>> = vec![];
        assert!(compute_fidelity(&m1, &m2).is_err());
    }

    #[test]
    fn test_compute_fidelity_rejects_ragged_rows() {
        // Outer dims match (2), but row 0 of u is shorter than needed, tripping
        // the `j >= u[i].len()` guard inside the nested loop.
        let m1 = vec![
            vec![Complex64::new(1.0, 0.0)],
            vec![Complex64::new(1.0, 0.0)],
        ];
        let m2 = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        assert!(compute_fidelity(&m1, &m2).is_err());
    }

    #[test]
    fn test_validate_decomposition_passes_above_threshold() {
        let m1 = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        let m2 = m1.clone();
        let fidelity = validate_decomposition(&m1, &m2, 0.99).unwrap();
        assert!(fidelity >= 0.99);
    }

    #[test]
    fn test_validate_decomposition_fails_below_threshold() {
        let m1 = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        let m2 = vec![
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ];
        let result = validate_decomposition(&m1, &m2, 0.9999);
        assert!(result.is_err());
    }

    #[test]
    fn test_optimize_gate_sequence_level_zero_returns_unchanged() {
        let gates: Vec<Arc<dyn Gate>> = vec![Arc::new(MockGate {
            name: "H".to_string(),
            n_qubits: 1,
            matrix: None,
        })];
        let result = optimize_gate_sequence(gates.clone(), 0);
        assert_eq!(result.len(), gates.len());
    }

    #[test]
    fn test_optimize_gate_sequence_nonzero_level_currently_noop() {
        let gates: Vec<Arc<dyn Gate>> = vec![Arc::new(MockGate {
            name: "H".to_string(),
            n_qubits: 1,
            matrix: None,
        })];
        // TODO in production code: optimization not yet implemented, gates pass through unchanged
        let result = optimize_gate_sequence(gates.clone(), 2);
        assert_eq!(result.len(), gates.len());
    }
}
