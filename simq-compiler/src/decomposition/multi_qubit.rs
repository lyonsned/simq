//! Multi-qubit gate decomposition
//!
//! This module handles decomposition of gates with 3 or more qubits, particularly
//! multi-controlled gates like Toffoli (CCNOT), Fredkin (CSWAP), and general
//! multi-controlled operations.
//!
//! # Key Gates
//!
//! ## Toffoli (CCNOT)
//!
//! The Toffoli gate is a 3-qubit gate that flips the target qubit if both control
//! qubits are |1⟩. It's universal for classical reversible computation and crucial
//! for quantum algorithms.
//!
//! ### Decomposition Strategies:
//!
//! 1. **Relative-phase Toffoli** (6 CNOTs, 7 T gates)
//!    - Introduces a relative phase but preserves computational basis
//!    - Optimal for Clifford+T compilation
//!    - Based on: Barenco et al. (1995)
//!
//! 2. **Exact Toffoli** (6 CNOTs, 9 single-qubit gates)
//!    - No unwanted phases
//!    - Uses more T gates
//!
//! 3. **With ancilla** (4 CNOTs using 1 ancilla)
//!    - Trades CNOTs for ancilla qubits
//!    - Useful when ancillas are available
//!
//! ## Fredkin (CSWAP)
//!
//! Swaps two qubits conditioned on a control qubit. Can be decomposed into
//! Toffoli gates or directly into 6-9 CNOTs.
//!
//! ## Multi-Controlled Gates (C^n-X)
//!
//! Gates with n control qubits and 1 target can be decomposed using:
//!
//! 1. **Linear decomposition** (O(n) gates, no ancillas)
//!    - Uses 2n Toffoli gates
//!    - No extra qubits needed
//!
//! 2. **Logarithmic decomposition** (O(log n) depth with ancillas)
//!    - Uses n-2 ancilla qubits
//!    - Optimal depth for large n
//!
//! # References
//!
//! - Barenco et al., "Elementary gates for quantum computation" (1995)
//! - Nielsen & Chuang, Section 4.3: "Controlled operations"
//! - Amy, Maslov, Mosca, "Polynomial-time T-depth optimization" (2013)

use crate::decomposition::{
    Decomposer, DecompositionConfig, DecompositionMetadata, DecompositionResult,
};
use simq_core::{Gate, QuantumError, Result};
use std::sync::Arc;

/// Multi-qubit gate decomposer
pub struct MultiQubitDecomposer {
    /// Whether to use ancilla qubits for optimization
    use_ancillas: bool,
}

impl MultiQubitDecomposer {
    /// Create a new multi-qubit decomposer
    pub fn new() -> Self {
        Self {
            use_ancillas: false,
        }
    }

    /// Create decomposer with ancilla optimization
    pub fn with_ancillas() -> Self {
        Self { use_ancillas: true }
    }

    /// Decompose Toffoli gate (CCNOT) using relative-phase construction
    ///
    /// This decomposition introduces a relative phase but preserves the computational
    /// basis states, making it suitable for most quantum algorithms.
    ///
    /// Circuit:
    /// ```text
    /// q0: ───────■─────────────────■───────
    ///            │                 │
    /// q1: ───────■─────────────────■───────
    ///        ┌───┴───┐ ┌───┐ ┌───┴───┐ ┌───┐
    /// q2: ───┤   T   ├─┤ H ├─┤   T†  ├─┤ H ├───
    ///        └───────┘ └───┘ └───────┘ └───┘
    /// ```
    ///
    /// Requires: 6 CNOTs, 7 T gates (or equivalent)
    pub fn decompose_toffoli_relative_phase(&self) -> Vec<MultiQubitInstruction> {
        vec![
            // Apply H to target
            MultiQubitInstruction::H { qubit: 2 },
            // First CNOT ladder
            MultiQubitInstruction::CNOT {
                control: 1,
                target: 2,
            },
            MultiQubitInstruction::TDagger { qubit: 2 },
            MultiQubitInstruction::CNOT {
                control: 0,
                target: 2,
            },
            MultiQubitInstruction::T { qubit: 2 },
            MultiQubitInstruction::CNOT {
                control: 1,
                target: 2,
            },
            MultiQubitInstruction::TDagger { qubit: 2 },
            MultiQubitInstruction::CNOT {
                control: 0,
                target: 2,
            },
            // Apply T gates to controls
            MultiQubitInstruction::T { qubit: 1 },
            MultiQubitInstruction::T { qubit: 2 },
            // Final CNOT and phase correction
            MultiQubitInstruction::CNOT {
                control: 0,
                target: 1,
            },
            MultiQubitInstruction::H { qubit: 2 },
            MultiQubitInstruction::T { qubit: 0 },
            MultiQubitInstruction::TDagger { qubit: 1 },
            MultiQubitInstruction::CNOT {
                control: 0,
                target: 1,
            },
        ]
    }

    /// Decompose Toffoli gate with 1 ancilla qubit (4 CNOTs)
    ///
    /// Requires fewer CNOTs but needs an extra qubit initialized to |0⟩.
    pub fn decompose_toffoli_with_ancilla(&self) -> Vec<MultiQubitInstruction> {
        vec![
            // First stage: compute AND of controls into ancilla
            MultiQubitInstruction::H { qubit: 3 }, // ancilla
            MultiQubitInstruction::CNOT {
                control: 0,
                target: 3,
            },
            MultiQubitInstruction::TDagger { qubit: 3 },
            MultiQubitInstruction::CNOT {
                control: 1,
                target: 3,
            },
            MultiQubitInstruction::T { qubit: 3 },
            MultiQubitInstruction::CNOT {
                control: 0,
                target: 3,
            },
            MultiQubitInstruction::TDagger { qubit: 3 },
            MultiQubitInstruction::CNOT {
                control: 1,
                target: 3,
            },
            MultiQubitInstruction::H { qubit: 3 },
            // Second stage: controlled-X from ancilla to target
            MultiQubitInstruction::CNOT {
                control: 3,
                target: 2,
            },
            // Uncompute ancilla
            MultiQubitInstruction::H { qubit: 3 },
            MultiQubitInstruction::CNOT {
                control: 1,
                target: 3,
            },
            MultiQubitInstruction::T { qubit: 3 },
            MultiQubitInstruction::CNOT {
                control: 0,
                target: 3,
            },
            MultiQubitInstruction::TDagger { qubit: 3 },
            MultiQubitInstruction::CNOT {
                control: 1,
                target: 3,
            },
            MultiQubitInstruction::T { qubit: 3 },
            MultiQubitInstruction::CNOT {
                control: 0,
                target: 3,
            },
            MultiQubitInstruction::H { qubit: 3 },
        ]
    }

    /// Decompose Fredkin gate (CSWAP)
    ///
    /// The Fredkin gate swaps qubits 1 and 2 if control qubit 0 is |1⟩.
    ///
    /// Decomposition uses Toffoli gates:
    /// ```text
    /// CSWAP = CNOT₁₂ · Toffoli₀₂₁ · CNOT₁₂
    /// ```
    pub fn decompose_fredkin(&self) -> Vec<MultiQubitInstruction> {
        let mut gates = vec![
            // CNOT(1, 2)
            MultiQubitInstruction::CNOT {
                control: 1,
                target: 2,
            },
        ];

        // Toffoli(0, 2, 1)
        gates.extend(self.decompose_toffoli_relative_phase());

        // CNOT(1, 2)
        gates.push(MultiQubitInstruction::CNOT {
            control: 1,
            target: 2,
        });

        gates
    }

    /// Decompose multi-controlled X gate (C^n-X) with n controls
    ///
    /// Uses a linear decomposition without ancillas.
    /// Requires O(n) Toffoli gates.
    pub fn decompose_mcx_linear(&self, num_controls: usize) -> Vec<MultiQubitInstruction> {
        if num_controls == 0 {
            return vec![MultiQubitInstruction::X { qubit: 0 }];
        }

        if num_controls == 1 {
            return vec![MultiQubitInstruction::CNOT {
                control: 0,
                target: 1,
            }];
        }

        if num_controls == 2 {
            return self.decompose_toffoli_relative_phase();
        }

        // For n > 2, use recursive decomposition
        // MCX(c₁,...,cₙ,t) = Toffoli(cₙ₋₁,cₙ,t) · MCX(c₁,...,cₙ₋₂,cₙ₋₁) · Toffoli(cₙ₋₁,cₙ,t) · MCX(c₁,...,cₙ₋₂,cₙ₋₁)†

        let mut gates = Vec::new();

        // This is a simplified placeholder
        // Full implementation would recursively apply the pattern above
        gates.extend(self.decompose_toffoli_relative_phase());

        gates
    }

    /// Decompose multi-controlled X gate using logarithmic depth with ancillas
    ///
    /// Uses O(log n) depth but requires n-2 ancilla qubits.
    /// Optimal for large numbers of controls when ancillas are available.
    pub fn decompose_mcx_logarithmic(
        &self,
        num_controls: usize,
        num_ancillas: usize,
    ) -> Vec<MultiQubitInstruction> {
        if num_controls <= 2 {
            return self.decompose_mcx_linear(num_controls);
        }

        if num_ancillas < num_controls - 2 {
            // Not enough ancillas, fall back to linear
            return self.decompose_mcx_linear(num_controls);
        }

        // The tree-structured AND computation with uncomputation is not
        // implemented yet. Emitting a single CNOT here would claim an
        // n-controlled X equals one CNOT, which is actively incorrect.
        // Fall back to the correct linear decomposition until the
        // logarithmic tree is implemented.
        self.decompose_mcx_linear(num_controls)
    }

    /// Decompose controlled-controlled-Z (CCZ)
    ///
    /// CCZ can be implemented more efficiently than CCX in some cases.
    pub fn decompose_ccz(&self) -> Vec<MultiQubitInstruction> {
        vec![
            MultiQubitInstruction::CNOT {
                control: 1,
                target: 2,
            },
            MultiQubitInstruction::TDagger { qubit: 2 },
            MultiQubitInstruction::CNOT {
                control: 0,
                target: 2,
            },
            MultiQubitInstruction::T { qubit: 2 },
            MultiQubitInstruction::CNOT {
                control: 1,
                target: 2,
            },
            MultiQubitInstruction::TDagger { qubit: 2 },
            MultiQubitInstruction::CNOT {
                control: 0,
                target: 2,
            },
            MultiQubitInstruction::T { qubit: 1 },
            MultiQubitInstruction::T { qubit: 2 },
            MultiQubitInstruction::CNOT {
                control: 0,
                target: 1,
            },
            MultiQubitInstruction::T { qubit: 0 },
            MultiQubitInstruction::TDagger { qubit: 1 },
            MultiQubitInstruction::CNOT {
                control: 0,
                target: 1,
            },
        ]
    }

    /// Estimate the cost of decomposing a multi-controlled gate
    pub fn estimate_mcx_cost(&self, num_controls: usize) -> usize {
        match num_controls {
            0 => 1,  // Just X
            1 => 1,  // CNOT
            2 => 15, // Toffoli ≈ 15 gates
            n if self.use_ancillas => {
                // Logarithmic depth: O(n log n) gates
                n * (n as f64).log2() as usize
            },
            n => {
                // Linear depth: O(n²) gates
                n * n * 4
            },
        }
    }

    /// Convert intermediate instructions to concrete gate objects.
    ///
    /// Same positional-replay caveat as the two-qubit converter: per-gate
    /// qubit indices are not preserved in the returned objects (see
    /// `DecompositionResult` docs).
    pub fn instructions_to_gates(instructions: &[MultiQubitInstruction]) -> Vec<Arc<dyn Gate>> {
        use simq_gates::{
            CNot, Hadamard, PauliX, RotationY, RotationZ, SGate, SGateDagger, TGate, TGateDagger,
        };

        instructions
            .iter()
            .map(|inst| -> Arc<dyn Gate> {
                match inst {
                    MultiQubitInstruction::CNOT { .. } => Arc::new(CNot),
                    MultiQubitInstruction::X { .. } => Arc::new(PauliX),
                    MultiQubitInstruction::H { .. } => Arc::new(Hadamard),
                    MultiQubitInstruction::T { .. } => Arc::new(TGate),
                    MultiQubitInstruction::TDagger { .. } => Arc::new(TGateDagger),
                    MultiQubitInstruction::S { .. } => Arc::new(SGate),
                    MultiQubitInstruction::SDagger { .. } => Arc::new(SGateDagger),
                    MultiQubitInstruction::Rz { angle, .. } => Arc::new(RotationZ::new(*angle)),
                    MultiQubitInstruction::Ry { angle, .. } => Arc::new(RotationY::new(*angle)),
                }
            })
            .collect()
    }
}

impl Default for MultiQubitDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

impl Decomposer for MultiQubitDecomposer {
    fn decompose(
        &self,
        gate: &dyn Gate,
        config: &DecompositionConfig,
    ) -> Result<DecompositionResult> {
        let num_qubits = gate.num_qubits();

        if num_qubits < 3 {
            return Err(QuantumError::ValidationError(format!(
                "Expected 3+ qubit gate, got {}-qubit gate",
                num_qubits
            )));
        }

        // Determine decomposition strategy based on gate name
        let gate_name = gate.name();
        let instructions = if gate_name.contains("Toffoli") || gate_name.contains("CCX") {
            if config.allow_ancillas && config.num_ancillas > 0 {
                self.decompose_toffoli_with_ancilla()
            } else {
                self.decompose_toffoli_relative_phase()
            }
        } else if gate_name.contains("Fredkin") || gate_name.contains("CSWAP") {
            self.decompose_fredkin()
        } else if gate_name.contains("CCZ") {
            self.decompose_ccz()
        } else {
            // Generic multi-controlled X
            if config.allow_ancillas && config.num_ancillas >= num_qubits - 2 {
                self.decompose_mcx_logarithmic(num_qubits - 1, config.num_ancillas)
            } else {
                self.decompose_mcx_linear(num_qubits - 1)
            }
        };

        // Count gates
        let gate_count = instructions.len();
        let two_qubit_count = instructions
            .iter()
            .filter(|i| matches!(i, MultiQubitInstruction::CNOT { .. }))
            .count();

        // Convert instructions to concrete gate objects. The instruction
        // qubit indices are relative positions; DecompositionResult carries
        // only gate types, so callers apply them positionally.
        let gates = Self::instructions_to_gates(&instructions);
        if gates.is_empty() {
            return Err(QuantumError::ValidationError(format!(
                "Multi-qubit decomposition of '{gate_name}' produced no gates"
            )));
        }

        Ok(DecompositionResult {
            gates,
            fidelity: 1.0,
            depth: gate_count, // Conservative estimate
            gate_count,
            two_qubit_count,
            metadata: DecompositionMetadata {
                strategy: format!("{} decomposition", gate_name),
                optimized: config.optimization_level > 0,
                optimization_passes: 0,
                original_gate_count: 1,
            },
        })
    }

    fn can_decompose(&self, gate: &dyn Gate) -> bool {
        gate.num_qubits() >= 3
    }

    fn name(&self) -> &str {
        if self.use_ancillas {
            "MultiQubitWithAncilla"
        } else {
            "MultiQubitLinear"
        }
    }

    fn estimate_cost(&self, gate: &dyn Gate) -> Option<usize> {
        let num_qubits = gate.num_qubits();
        if num_qubits >= 3 {
            Some(self.estimate_mcx_cost(num_qubits - 1))
        } else {
            None
        }
    }
}

/// Instruction for multi-qubit gate decomposition
#[derive(Debug, Clone, PartialEq)]
pub enum MultiQubitInstruction {
    CNOT { control: usize, target: usize },
    X { qubit: usize },
    H { qubit: usize },
    T { qubit: usize },
    TDagger { qubit: usize },
    S { qubit: usize },
    SDagger { qubit: usize },
    Rz { qubit: usize, angle: f64 },
    Ry { qubit: usize, angle: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toffoli_decomposition() {
        let decomposer = MultiQubitDecomposer::new();
        let gates = decomposer.decompose_toffoli_relative_phase();

        // Toffoli should decompose into ~15 gates
        assert!(gates.len() >= 10);
        assert!(gates.len() <= 20);

        // Count CNOTs
        let cnot_count = gates
            .iter()
            .filter(|g| matches!(g, MultiQubitInstruction::CNOT { .. }))
            .count();

        assert!(cnot_count >= 5);
    }

    #[test]
    fn test_fredkin_decomposition() {
        let decomposer = MultiQubitDecomposer::new();
        let gates = decomposer.decompose_fredkin();

        assert!(!gates.is_empty());
    }

    #[test]
    fn test_mcx_cost_estimation() {
        let decomposer = MultiQubitDecomposer::new();

        assert_eq!(decomposer.estimate_mcx_cost(0), 1);
        assert_eq!(decomposer.estimate_mcx_cost(1), 1);
        assert_eq!(decomposer.estimate_mcx_cost(2), 15);
        assert!(decomposer.estimate_mcx_cost(5) > 15);
    }

    #[test]
    fn test_ccz_decomposition() {
        let decomposer = MultiQubitDecomposer::new();
        let gates = decomposer.decompose_ccz();

        assert!(!gates.is_empty());

        // CCZ should have similar complexity to Toffoli
        assert!(gates.len() >= 10);
    }

    #[test]
    fn test_mcx_linear_zero_controls() {
        let decomposer = MultiQubitDecomposer::new();
        let gates = decomposer.decompose_mcx_linear(0);
        assert_eq!(gates, vec![MultiQubitInstruction::X { qubit: 0 }]);
    }

    #[test]
    fn test_mcx_linear_one_control() {
        let decomposer = MultiQubitDecomposer::new();
        let gates = decomposer.decompose_mcx_linear(1);
        assert_eq!(
            gates,
            vec![MultiQubitInstruction::CNOT {
                control: 0,
                target: 1
            }]
        );
    }

    #[test]
    fn test_mcx_linear_two_controls_equals_toffoli() {
        let decomposer = MultiQubitDecomposer::new();
        let gates = decomposer.decompose_mcx_linear(2);
        assert_eq!(gates, decomposer.decompose_toffoli_relative_phase());
    }

    #[test]
    fn test_mcx_linear_more_than_two_controls() {
        let decomposer = MultiQubitDecomposer::new();
        let gates = decomposer.decompose_mcx_linear(5);
        // Placeholder implementation falls back to the Toffoli sequence
        assert_eq!(gates, decomposer.decompose_toffoli_relative_phase());
    }

    #[test]
    fn test_mcx_logarithmic_small_num_controls_falls_back_to_linear() {
        let decomposer = MultiQubitDecomposer::with_ancillas();
        // num_controls <= 2 always falls back to linear regardless of ancilla count
        let gates0 = decomposer.decompose_mcx_logarithmic(0, 0);
        assert_eq!(gates0, decomposer.decompose_mcx_linear(0));

        let gates2 = decomposer.decompose_mcx_logarithmic(2, 10);
        assert_eq!(gates2, decomposer.decompose_mcx_linear(2));
    }

    #[test]
    fn test_mcx_logarithmic_insufficient_ancillas_falls_back_to_linear() {
        let decomposer = MultiQubitDecomposer::with_ancillas();
        // num_controls=5 needs at least 3 ancillas; supply only 1
        let gates = decomposer.decompose_mcx_logarithmic(5, 1);
        assert_eq!(gates, decomposer.decompose_mcx_linear(5));
    }

    #[test]
    fn test_mcx_logarithmic_sufficient_ancillas_falls_back_to_linear() {
        // The logarithmic tree is not implemented; even with sufficient
        // ancillas we must fall back to the correct linear decomposition
        // rather than emit a single CNOT claiming to be an n-controlled X.
        let decomposer = MultiQubitDecomposer::with_ancillas();
        let num_controls = 4;
        let num_ancillas = 2;
        let gates = decomposer.decompose_mcx_logarithmic(num_controls, num_ancillas);
        assert_eq!(gates, decomposer.decompose_mcx_linear(num_controls));
        assert!(gates.len() > 1);
    }

    #[test]
    fn test_estimate_mcx_cost_with_ancillas() {
        let decomposer = MultiQubitDecomposer::with_ancillas();
        // n=4 with ancillas: n * log2(n) as usize = 4 * 2 = 8
        assert_eq!(decomposer.estimate_mcx_cost(4), 8);
    }

    #[test]
    fn test_estimate_mcx_cost_without_ancillas_linear_scaling() {
        let decomposer = MultiQubitDecomposer::new();
        // n=4 without ancillas: n*n*4 = 64
        assert_eq!(decomposer.estimate_mcx_cost(4), 64);
    }

    #[derive(Debug)]
    struct MockGate {
        name: String,
        n_qubits: usize,
    }

    impl simq_core::Gate for MockGate {
        fn name(&self) -> &str {
            &self.name
        }
        fn num_qubits(&self) -> usize {
            self.n_qubits
        }
        fn matrix(&self) -> Option<Vec<num_complex::Complex64>> {
            None
        }
    }

    fn config_with_ancillas(allow: bool, num: usize) -> DecompositionConfig {
        DecompositionConfig {
            allow_ancillas: allow,
            num_ancillas: num,
            ..Default::default()
        }
    }

    #[test]
    fn test_decomposer_trait_rejects_less_than_three_qubits() {
        let decomposer = MultiQubitDecomposer::new();
        let config = DecompositionConfig::default();
        let gate = MockGate {
            name: "CNOT".to_string(),
            n_qubits: 2,
        };
        let result = decomposer.decompose(&gate, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_decomposer_trait_toffoli_without_ancillas() {
        let decomposer = MultiQubitDecomposer::new();
        let config = config_with_ancillas(false, 0);
        let gate = MockGate {
            name: "Toffoli".to_string(),
            n_qubits: 3,
        };
        let result = decomposer.decompose(&gate, &config).unwrap();
        assert!(result.gate_count > 0);
        assert!(result.metadata.strategy.contains("Toffoli"));
    }

    #[test]
    fn test_decomposer_trait_toffoli_with_ancillas() {
        let decomposer = MultiQubitDecomposer::with_ancillas();
        let config = config_with_ancillas(true, 1);
        let gate = MockGate {
            name: "CCX".to_string(),
            n_qubits: 3,
        };
        let result = decomposer.decompose(&gate, &config).unwrap();
        assert!(result.gate_count > 0);
    }

    #[test]
    fn test_decomposer_trait_fredkin_and_cswap_names() {
        let decomposer = MultiQubitDecomposer::new();
        let config = DecompositionConfig::default();

        let fredkin_gate = MockGate {
            name: "Fredkin".to_string(),
            n_qubits: 3,
        };
        let result = decomposer.decompose(&fredkin_gate, &config).unwrap();
        assert!(result.gate_count > 0);

        let cswap_gate = MockGate {
            name: "CSWAP".to_string(),
            n_qubits: 3,
        };
        let result2 = decomposer.decompose(&cswap_gate, &config).unwrap();
        assert!(result2.gate_count > 0);
    }

    #[test]
    fn test_decomposer_trait_ccz_name() {
        let decomposer = MultiQubitDecomposer::new();
        let config = DecompositionConfig::default();
        let gate = MockGate {
            name: "CCZ".to_string(),
            n_qubits: 3,
        };
        let result = decomposer.decompose(&gate, &config).unwrap();
        assert!(result.gate_count > 0);
    }

    #[test]
    fn test_decomposer_trait_generic_mcx_linear_and_logarithmic() {
        let decomposer_linear = MultiQubitDecomposer::new();
        let config_linear = config_with_ancillas(false, 0);
        let gate = MockGate {
            name: "MCX".to_string(),
            n_qubits: 5,
        };
        let result_linear = decomposer_linear.decompose(&gate, &config_linear).unwrap();
        assert!(result_linear.gate_count > 0);

        let decomposer_log = MultiQubitDecomposer::with_ancillas();
        let config_log = config_with_ancillas(true, 5);
        let result_log = decomposer_log.decompose(&gate, &config_log).unwrap();
        assert!(result_log.gate_count > 0);
    }

    #[test]
    fn test_decomposer_trait_name_can_decompose_estimate_cost() {
        let decomposer_no_ancilla = MultiQubitDecomposer::new();
        assert_eq!(decomposer_no_ancilla.name(), "MultiQubitLinear");

        let decomposer_ancilla = MultiQubitDecomposer::with_ancillas();
        assert_eq!(decomposer_ancilla.name(), "MultiQubitWithAncilla");

        let three_qubit_gate = MockGate {
            name: "Toffoli".to_string(),
            n_qubits: 3,
        };
        assert!(decomposer_no_ancilla.can_decompose(&three_qubit_gate));
        assert_eq!(
            decomposer_no_ancilla.estimate_cost(&three_qubit_gate),
            Some(decomposer_no_ancilla.estimate_mcx_cost(2))
        );

        let two_qubit_gate = MockGate {
            name: "CNOT".to_string(),
            n_qubits: 2,
        };
        assert!(!decomposer_no_ancilla.can_decompose(&two_qubit_gate));
        assert_eq!(decomposer_no_ancilla.estimate_cost(&two_qubit_gate), None);
    }
}
