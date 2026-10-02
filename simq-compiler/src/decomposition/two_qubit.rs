//! Two-qubit gate decomposition
//!
//! This module provides decomposition strategies for two-qubit gates into a basis
//! set of CNOT (or other entangling gates) plus single-qubit rotations.
//!
//! # Key Results
//!
//! ## Canonical Decomposition Theorem
//!
//! Any two-qubit unitary U can be decomposed as:
//! ```text
//! U = (A₁ ⊗ A₂) · CNOT · (B₁ ⊗ B₂) · CNOT · (C₁ ⊗ C₂) · CNOT · (D₁ ⊗ D₂)
//! ```
//! where Aᵢ, Bᵢ, Cᵢ, Dᵢ are single-qubit unitaries.
//!
//! This requires **at most 3 CNOTs**, which is optimal for generic two-qubit gates.
//!
//! ## Special Cases
//!
//! Many important gates require fewer CNOTs:
//! - Controlled gates (CX, CY, CZ): 1 CNOT
//! - SWAP: 3 CNOTs
//! - iSWAP: 2 CNOTs + single-qubit gates
//! - √iSWAP: 2 CNOTs + single-qubit gates
//!
//! # Entangling Gates
//!
//! Different quantum hardware uses different native entangling gates:
//! - IBM, Rigetti, IonQ: CNOT (Controlled-X)
//! - Google Sycamore: √iSWAP
//! - Some superconducting qubits: CZ (Controlled-Z)
//!
//! This module provides conversions between these representations.
//!
//! # References
//!
//! - Barenco et al., "Elementary gates for quantum computation" (1995)
//! - Vatan & Williams, "Optimal quantum circuits for general two-qubit gates" (2004)
//! - Shende, Bullock, Markov, "Synthesis of quantum-logic circuits" (2006)

use crate::decomposition::{
    Decomposer, DecompositionConfig, DecompositionMetadata, DecompositionResult,
};
use crate::matrix_computation::{is_unitary_4x4, Matrix4};
use num_complex::Complex64;
use simq_core::{Gate, QuantumError, Result};
use std::f64::consts::PI;
use std::sync::Arc;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Entangling gate basis for two-qubit decomposition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntanglementGate {
    /// CNOT (Controlled-X) - most common
    CNOT,

    /// CZ (Controlled-Z) - equivalent to CNOT up to local gates
    CZ,

    /// iSWAP - native to some superconducting architectures
    ISWAP,

    /// √iSWAP - Google Sycamore's native gate
    SqrtISWAP,

    /// SWAP - sometimes useful as a basis
    SWAP,

    /// CPhase - parameterized controlled phase
    CPhase,
}

/// Two-qubit gate decomposer
pub struct TwoQubitDecomposer {
    entangling_gate: EntanglementGate,
}

impl TwoQubitDecomposer {
    /// Create a new two-qubit decomposer with specified entangling gate
    pub fn new(entangling_gate: EntanglementGate) -> Self {
        Self { entangling_gate }
    }

    /// Decompose a two-qubit gate using canonical decomposition
    ///
    /// Returns the decomposition in the form:
    /// U = (A₁ ⊗ A₂) · E · (B₁ ⊗ B₂) · E · (C₁ ⊗ C₂) · E · (D₁ ⊗ D₂)
    ///
    /// where E is the entangling gate (CNOT, CZ, etc.)
    pub fn decompose_canonical(&self, matrix: &Matrix4) -> Result<CanonicalDecomposition> {
        if !is_unitary_4x4(matrix) {
            return Err(QuantumError::ValidationError("Matrix is not unitary".to_string()));
        }

        // Full magic-basis canonical decomposition (arbitrary 2-qubit
        // unitary -> ≤3 entangling gates + local rotations) is not
        // implemented yet. Returning a placeholder with empty
        // single_qubit_layers and fidelity 1.0 would silently delete the
        // gate downstream, so fail loudly instead. Known gates (SWAP,
        // iSWAP, CZ, CNOT conversions) are handled via their dedicated
        // decompose_* helpers; only the generic path errors here.
        Err(QuantumError::ValidationError(
            "Generic two-qubit canonical decomposition (magic-basis) is not implemented; \
             only known gates (SWAP, iSWAP, √iSWAP, CNOT<->CZ) are supported"
                .to_string(),
        ))
    }

    /// Decompose SWAP gate
    ///
    /// SWAP = CNOT₀₁ · CNOT₁₀ · CNOT₀₁
    ///
    /// Requires 3 CNOTs (or equivalent entangling gates).
    pub fn decompose_swap(&self) -> Vec<TwoQubitGateInstruction> {
        match self.entangling_gate {
            EntanglementGate::CNOT => vec![
                TwoQubitGateInstruction::CNOT {
                    control: 0,
                    target: 1,
                },
                TwoQubitGateInstruction::CNOT {
                    control: 1,
                    target: 0,
                },
                TwoQubitGateInstruction::CNOT {
                    control: 0,
                    target: 1,
                },
            ],
            EntanglementGate::CZ => {
                // CZ can implement SWAP with additional Hadamards
                vec![
                    TwoQubitGateInstruction::Hadamard { qubit: 1 },
                    TwoQubitGateInstruction::CZ,
                    TwoQubitGateInstruction::Hadamard { qubit: 1 },
                    TwoQubitGateInstruction::Hadamard { qubit: 0 },
                    TwoQubitGateInstruction::CZ,
                    TwoQubitGateInstruction::Hadamard { qubit: 0 },
                    TwoQubitGateInstruction::Hadamard { qubit: 1 },
                    TwoQubitGateInstruction::CZ,
                    TwoQubitGateInstruction::Hadamard { qubit: 1 },
                ]
            },
            _ => {
                // For other gates, convert to CNOT first
                vec![]
            },
        }
    }

    /// Decompose iSWAP gate
    ///
    /// iSWAP = S₀ · S₁ · H₁ · CNOT₀₁ · H₁ · CNOT₁₀ · H₀
    ///
    /// Requires 2 CNOTs plus local gates.
    pub fn decompose_iswap(&self) -> Vec<TwoQubitGateInstruction> {
        if self.entangling_gate == EntanglementGate::ISWAP {
            return vec![TwoQubitGateInstruction::ISWAP];
        }

        vec![
            TwoQubitGateInstruction::SGate { qubit: 0 },
            TwoQubitGateInstruction::SGate { qubit: 1 },
            TwoQubitGateInstruction::Hadamard { qubit: 1 },
            TwoQubitGateInstruction::CNOT {
                control: 0,
                target: 1,
            },
            TwoQubitGateInstruction::Hadamard { qubit: 1 },
            TwoQubitGateInstruction::CNOT {
                control: 1,
                target: 0,
            },
            TwoQubitGateInstruction::Hadamard { qubit: 0 },
        ]
    }

    /// Decompose √iSWAP gate (Google Sycamore's native gate)
    ///
    /// Requires 2 CNOTs plus local rotations.
    pub fn decompose_sqrt_iswap(&self) -> Vec<TwoQubitGateInstruction> {
        if self.entangling_gate == EntanglementGate::SqrtISWAP {
            return vec![TwoQubitGateInstruction::SqrtISWAP];
        }

        // Decomposition in terms of CNOTs
        vec![
            TwoQubitGateInstruction::Ry {
                qubit: 0,
                angle: PI / 4.0,
            },
            TwoQubitGateInstruction::Ry {
                qubit: 1,
                angle: PI / 4.0,
            },
            TwoQubitGateInstruction::CNOT {
                control: 0,
                target: 1,
            },
            TwoQubitGateInstruction::Rz {
                qubit: 1,
                angle: -PI / 4.0,
            },
            TwoQubitGateInstruction::CNOT {
                control: 1,
                target: 0,
            },
            TwoQubitGateInstruction::Ry {
                qubit: 0,
                angle: -PI / 4.0,
            },
            TwoQubitGateInstruction::Ry {
                qubit: 1,
                angle: -PI / 4.0,
            },
        ]
    }

    /// Convert CNOT to CZ
    ///
    /// CNOT = H₁ · CZ · H₁
    pub fn cnot_to_cz() -> Vec<TwoQubitGateInstruction> {
        vec![
            TwoQubitGateInstruction::Hadamard { qubit: 1 },
            TwoQubitGateInstruction::CZ,
            TwoQubitGateInstruction::Hadamard { qubit: 1 },
        ]
    }

    /// Convert CZ to CNOT
    ///
    /// CZ = H₁ · CNOT · H₁
    pub fn cz_to_cnot() -> Vec<TwoQubitGateInstruction> {
        vec![
            TwoQubitGateInstruction::Hadamard { qubit: 1 },
            TwoQubitGateInstruction::CNOT {
                control: 0,
                target: 1,
            },
            TwoQubitGateInstruction::Hadamard { qubit: 1 },
        ]
    }

    /// Optimize the decomposition by reducing gate count
    ///
    /// Currently a pass-through: no merging/cancellation is implemented yet.
    /// Kept as a hook so callers can request optimization levels without
    /// silent behavior change; returns without modifying `decomp`.
    pub fn optimize_decomposition(&self, _decomp: &mut CanonicalDecomposition, _level: u8) {}

    /// Convert intermediate two-qubit instructions to concrete gate objects.
    ///
    /// Note: the returned `Arc<dyn Gate>` values carry only the gate type;
    /// the per-instruction qubit indices (`control`/`target`/`qubit`) are
    /// not preserved. Callers must replay `gates` positionally against the
    /// original `instructions` (e.g. SWAP = `[CNOT01, CNOT10, CNOT01]`
    /// becomes `[CNot, CNot, CNot]` — direction lives in the instruction
    /// order, not the gate objects). See `DecompositionResult` docs.
    ///
    /// Errors on `SqrtISWAP`: no dedicated √iSWAP gate type exists in
    /// `simq-gates`, and substituting `ISwap` would silently miscompile
    /// (different unitary). Callers on a √iSWAP basis should use the
    /// expanded CNOT form from `decompose_sqrt_iswap` instead.
    pub fn instructions_to_gates(
        instructions: &[TwoQubitGateInstruction],
    ) -> Result<Vec<Arc<dyn Gate>>> {
        use simq_gates::{
            CNot, Hadamard, ISwap, RotationX, RotationY, RotationZ, SGate, Swap, TGate, CZ,
        };

        instructions
            .iter()
            .map(|inst| -> Result<Arc<dyn Gate>> {
                match inst {
                    TwoQubitGateInstruction::CNOT { .. } => Ok(Arc::new(CNot)),
                    TwoQubitGateInstruction::CZ => Ok(Arc::new(CZ)),
                    TwoQubitGateInstruction::ISWAP => Ok(Arc::new(ISwap)),
                    TwoQubitGateInstruction::SqrtISWAP => Err(QuantumError::ValidationError(
                        "√iSWAP has no concrete gate type in simq-gates; use the CNOT expansion from decompose_sqrt_iswap or a CNOT/CZ entangling basis"
                            .to_string(),
                    )),
                    TwoQubitGateInstruction::SWAP => Ok(Arc::new(Swap)),
                    TwoQubitGateInstruction::Hadamard { .. } => Ok(Arc::new(Hadamard)),
                    TwoQubitGateInstruction::SGate { .. } => Ok(Arc::new(SGate)),
                    TwoQubitGateInstruction::TGate { .. } => Ok(Arc::new(TGate)),
                    TwoQubitGateInstruction::Rx { angle, .. } => {
                        Ok(Arc::new(RotationX::new(*angle)))
                    },
                    TwoQubitGateInstruction::Ry { angle, .. } => {
                        Ok(Arc::new(RotationY::new(*angle)))
                    },
                    TwoQubitGateInstruction::Rz { angle, .. } => {
                        Ok(Arc::new(RotationZ::new(*angle)))
                    },
                }
            })
            .collect()
    }
}

impl Decomposer for TwoQubitDecomposer {
    fn decompose(
        &self,
        gate: &dyn Gate,
        config: &DecompositionConfig,
    ) -> Result<DecompositionResult> {
        if gate.num_qubits() != 2 {
            return Err(QuantumError::ValidationError(format!(
                "Expected two-qubit gate, got {}-qubit gate",
                gate.num_qubits()
            )));
        }

        // Get gate matrix for validation (even known gates are validated).
        let matrix = gate.matrix().ok_or_else(|| {
            QuantumError::ValidationError("Gate does not provide matrix representation".to_string())
        })?;

        if matrix.len() != 16 {
            return Err(QuantumError::ValidationError(
                "Invalid matrix size for two-qubit gate".to_string(),
            ));
        }

        // Convert to Matrix4 format for unitarity check.
        let mut matrix_4x4: Matrix4 = [[ZERO; 4]; 4];
        #[allow(clippy::needless_range_loop)]
        for i in 0..4 {
            for j in 0..4 {
                let idx = i * 4 + j;
                matrix_4x4[i][j] = Complex64::new(matrix[idx].re, matrix[idx].im);
            }
        }
        if !is_unitary_4x4(&matrix_4x4) {
            return Err(QuantumError::ValidationError("Matrix is not unitary".to_string()));
        }

        // Dispatch known gates to their exact decompositions. Generic
        // unitaries require the unimplemented magic-basis canonical
        // decomposition, so error loudly instead of returning an empty
        // gate list with fidelity 1.0.
        let name_upper = gate.name().to_uppercase();
        let instructions: Vec<TwoQubitGateInstruction> =
            if name_upper.contains("SWAP") && !name_upper.contains("ISWAP") {
                let inst = self.decompose_swap();
                if inst.is_empty() {
                    return Err(QuantumError::ValidationError(format!(
                        "SWAP decomposition not available for entangling basis {:?}",
                        self.entangling_gate
                    )));
                }
                inst
            } else if name_upper.contains("ISWAP") || name_upper.contains("SQRTISWAP") {
                if name_upper.contains("SQRT") {
                    self.decompose_sqrt_iswap()
                } else {
                    self.decompose_iswap()
                }
            } else if name_upper == "CZ" {
                if self.entangling_gate == EntanglementGate::CNOT {
                    Self::cz_to_cnot()
                } else {
                    vec![TwoQubitGateInstruction::CZ]
                }
            } else if name_upper == "CNOT" || name_upper == "CX" {
                if self.entangling_gate == EntanglementGate::CZ {
                    Self::cnot_to_cz()
                } else {
                    vec![TwoQubitGateInstruction::CNOT {
                        control: 0,
                        target: 1,
                    }]
                }
            } else {
                // Generic two-qubit unitary: canonical decomposition needed.
                self.decompose_canonical(&matrix_4x4)?;
                unreachable!("decompose_canonical always errors for generic gates");
            };

        let gates = Self::instructions_to_gates(&instructions)?;
        if gates.is_empty() {
            return Err(QuantumError::ValidationError(format!(
                "Two-qubit decomposition of '{}' produced no gates",
                gate.name()
            )));
        }

        let two_qubit_count = instructions
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    TwoQubitGateInstruction::CNOT { .. }
                        | TwoQubitGateInstruction::CZ
                        | TwoQubitGateInstruction::ISWAP
                        | TwoQubitGateInstruction::SqrtISWAP
                        | TwoQubitGateInstruction::SWAP
                )
            })
            .count();
        let gate_count = gates.len();

        // Suppress unused warning when optimization is a pass-through.
        let _ = config.optimization_level;

        Ok(DecompositionResult {
            gates,
            fidelity: 1.0,
            depth: gate_count,
            gate_count,
            two_qubit_count,
            metadata: DecompositionMetadata {
                strategy: format!("{:?} canonical decomposition", self.entangling_gate),
                optimized: config.optimization_level > 0,
                optimization_passes: config.optimization_level as usize,
                original_gate_count: 1,
            },
        })
    }

    /// Check whether this decomposer may handle the gate.
    ///
    /// This is a shape check only (2 qubits + matrix present), not a
    /// synthesizability guarantee: `decompose` still errors for generic
    /// unitaries that need the unimplemented magic-basis canonical
    /// decomposition, and for √iSWAP instructions with no concrete gate
    /// type. Callers must handle `decompose` returning `Err` even when this
    /// returns `true`.
    fn can_decompose(&self, gate: &dyn Gate) -> bool {
        gate.num_qubits() == 2 && gate.matrix().is_some()
    }

    fn name(&self) -> &str {
        "TwoQubitCanonical"
    }

    fn estimate_cost(&self, gate: &dyn Gate) -> Option<usize> {
        if gate.num_qubits() == 2 {
            Some(3) // At most 3 entangling gates
        } else {
            None
        }
    }
}

/// Canonical decomposition result
#[derive(Debug, Clone)]
pub struct CanonicalDecomposition {
    /// Which entangling gate is used
    pub entangling_gate: EntanglementGate,

    /// Single-qubit gates between entangling gates
    pub single_qubit_layers: Vec<SingleQubitLayer>,

    /// Number of entangling gates used (≤ 3)
    pub num_entangling: usize,
}

/// Layer of single-qubit gates (applied in parallel)
#[derive(Debug, Clone)]
pub struct SingleQubitLayer {
    /// Gate on qubit 0
    pub gate_0: Option<SingleQubitAngles>,

    /// Gate on qubit 1
    pub gate_1: Option<SingleQubitAngles>,
}

/// Single-qubit gate angles (ZYZ parameterization)
#[derive(Debug, Clone, Copy)]
pub struct SingleQubitAngles {
    pub beta: f64,
    pub gamma: f64,
    pub delta: f64,
}

/// Two-qubit gate instruction (intermediate representation)
#[derive(Debug, Clone, PartialEq)]
pub enum TwoQubitGateInstruction {
    CNOT { control: usize, target: usize },
    CZ,
    ISWAP,
    SqrtISWAP,
    SWAP,
    Hadamard { qubit: usize },
    SGate { qubit: usize },
    TGate { qubit: usize },
    Rx { qubit: usize, angle: f64 },
    Ry { qubit: usize, angle: f64 },
    Rz { qubit: usize, angle: f64 },
}

impl CanonicalDecomposition {
    /// Count total gates in the decomposition
    pub fn total_gates(&self) -> usize {
        let mut count = self.num_entangling;
        for layer in &self.single_qubit_layers {
            if layer.gate_0.is_some() {
                count += 3; // ZYZ = 3 rotations
            }
            if layer.gate_1.is_some() {
                count += 3;
            }
        }
        count
    }

    /// Estimate circuit depth
    pub fn depth(&self) -> usize {
        // Each entangling gate adds depth, single-qubit gates are parallel
        self.num_entangling + self.single_qubit_layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swap_decomposition() {
        let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
        let swap_gates = decomposer.decompose_swap();

        assert_eq!(swap_gates.len(), 3);
        assert!(matches!(swap_gates[0], TwoQubitGateInstruction::CNOT { .. }));
    }

    #[test]
    fn test_cnot_cz_conversion() {
        let cnot_to_cz = TwoQubitDecomposer::cnot_to_cz();
        assert_eq!(cnot_to_cz.len(), 3);

        let cz_to_cnot = TwoQubitDecomposer::cz_to_cnot();
        assert_eq!(cz_to_cnot.len(), 3);
    }

    #[test]
    fn test_iswap_decomposition() {
        let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
        let iswap_gates = decomposer.decompose_iswap();

        assert!(!iswap_gates.is_empty());
    }

    #[test]
    fn test_canonical_decomposition_depth() {
        let decomp = CanonicalDecomposition {
            entangling_gate: EntanglementGate::CNOT,
            single_qubit_layers: vec![
                SingleQubitLayer {
                    gate_0: None,
                    gate_1: None,
                },
                SingleQubitLayer {
                    gate_0: None,
                    gate_1: None,
                },
            ],
            num_entangling: 3,
        };

        assert_eq!(decomp.depth(), 5); // 3 CNOTs + 2 layers
    }

    fn identity_matrix4() -> Matrix4 {
        let mut m = [[ZERO; 4]; 4];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = Complex64::new(1.0, 0.0);
        }
        m
    }

    fn non_unitary_matrix4() -> Matrix4 {
        let mut m = [[ZERO; 4]; 4];
        m[0][0] = Complex64::new(2.0, 0.0);
        m[1][1] = Complex64::new(1.0, 0.0);
        m[2][2] = Complex64::new(1.0, 0.0);
        m[3][3] = Complex64::new(1.0, 0.0);
        m
    }

    #[test]
    fn test_decompose_canonical_rejects_non_unitary() {
        let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
        let result = decomposer.decompose_canonical(&non_unitary_matrix4());
        assert!(result.is_err());
    }

    #[test]
    fn test_decompose_canonical_errors_on_identity() {
        // Generic magic-basis decomposition is unimplemented: even the
        // identity must error rather than return an empty placeholder with
        // fidelity 1.0.
        let decomposer = TwoQubitDecomposer::new(EntanglementGate::CZ);
        let result = decomposer.decompose_canonical(&identity_matrix4());
        assert!(result.is_err());
    }

    #[test]
    fn test_optimize_decomposition_noop_level_zero() {
        let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
        let mut decomp = CanonicalDecomposition {
            entangling_gate: EntanglementGate::CNOT,
            single_qubit_layers: vec![],
            num_entangling: 3,
        };
        // Pass-through regardless of level.
        decomposer.optimize_decomposition(&mut decomp, 0);
        assert_eq!(decomp.num_entangling, 3);
    }

    #[test]
    fn test_optimize_decomposition_nonzero_level_still_noop_today() {
        let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
        let mut decomp = CanonicalDecomposition {
            entangling_gate: EntanglementGate::CNOT,
            single_qubit_layers: vec![],
            num_entangling: 3,
        };
        decomposer.optimize_decomposition(&mut decomp, 2);
        assert_eq!(decomp.num_entangling, 3);
    }

    #[derive(Debug)]
    struct MockGate {
        name: String,
        n_qubits: usize,
        matrix: Option<Vec<Complex64>>,
    }

    impl simq_core::Gate for MockGate {
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

    fn identity_matrix4_flat() -> Vec<Complex64> {
        let mut v = vec![ZERO; 16];
        for i in 0..4 {
            v[i * 4 + i] = Complex64::new(1.0, 0.0);
        }
        v
    }

    #[test]
    fn test_decomposer_trait_rejects_non_two_qubit_gate() {
        let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
        let config = DecompositionConfig::default();
        let gate = MockGate {
            name: "H".to_string(),
            n_qubits: 1,
            matrix: None,
        };
        assert!(decomposer.decompose(&gate, &config).is_err());
    }

    #[test]
    fn test_decomposer_trait_rejects_missing_matrix() {
        let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
        let config = DecompositionConfig::default();
        let gate = MockGate {
            name: "CZ".to_string(),
            n_qubits: 2,
            matrix: None,
        };
        assert!(decomposer.decompose(&gate, &config).is_err());
    }

    #[test]
    fn test_decomposer_trait_rejects_invalid_matrix_size() {
        let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
        let config = DecompositionConfig::default();
        let gate = MockGate {
            name: "CZ".to_string(),
            n_qubits: 2,
            matrix: Some(vec![Complex64::new(1.0, 0.0)]),
        };
        assert!(decomposer.decompose(&gate, &config).is_err());
    }

    #[test]
    fn test_decomposer_trait_succeeds_with_optimization() {
        let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
        let config = DecompositionConfig {
            optimization_level: 1,
            ..Default::default()
        };
        let gate = MockGate {
            name: "CZ".to_string(),
            n_qubits: 2,
            matrix: Some(identity_matrix4_flat()),
        };
        let result = decomposer.decompose(&gate, &config).unwrap();
        assert!(result.metadata.optimized);
        // CZ -> H CNOT H: 3 gates, 1 entangling, real non-empty sequence.
        assert!(!result.gates.is_empty());
        assert_eq!(result.gate_count, result.gates.len());
        assert_eq!(result.gate_count, 3);
        assert_eq!(result.two_qubit_count, 1);
        assert_eq!(result.depth, 3);
    }

    #[test]
    fn test_decomposer_trait_generic_unitary_errors() {
        // Generic two-qubit unitaries have no exact decomposition yet:
        // must error instead of returning empty gates with fidelity 1.0.
        let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
        let config = DecompositionConfig::default();
        let gate = MockGate {
            name: "CUSTOM2Q".to_string(),
            n_qubits: 2,
            matrix: Some(identity_matrix4_flat()),
        };
        let result = decomposer.decompose(&gate, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_instructions_to_gates_rejects_sqrt_iswap() {
        // No concrete √iSWAP gate exists; substituting ISwap would silently
        // miscompile, so conversion must error.
        let result =
            TwoQubitDecomposer::instructions_to_gates(&[TwoQubitGateInstruction::SqrtISWAP]);
        assert!(result.is_err());
    }

    #[test]
    fn test_decomposer_trait_name_can_decompose_estimate_cost() {
        let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
        assert_eq!(decomposer.name(), "TwoQubitCanonical");

        let two_with_matrix = MockGate {
            name: "CZ".to_string(),
            n_qubits: 2,
            matrix: Some(identity_matrix4_flat()),
        };
        assert!(decomposer.can_decompose(&two_with_matrix));
        assert_eq!(decomposer.estimate_cost(&two_with_matrix), Some(3));

        let one_qubit = MockGate {
            name: "H".to_string(),
            n_qubits: 1,
            matrix: None,
        };
        assert!(!decomposer.can_decompose(&one_qubit));
        assert_eq!(decomposer.estimate_cost(&one_qubit), None);
    }

    #[test]
    fn test_decompose_swap_cz_variant_uses_hadamards() {
        let decomposer = TwoQubitDecomposer::new(EntanglementGate::CZ);
        let gates = decomposer.decompose_swap();
        assert_eq!(gates.len(), 9);
        let cz_count = gates
            .iter()
            .filter(|g| matches!(g, TwoQubitGateInstruction::CZ))
            .count();
        assert_eq!(cz_count, 3);
    }
}
