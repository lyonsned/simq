//! Circuit transpilation for backend compatibility
//!
//! The transpiler converts circuits to be compatible with specific backend
//! requirements by:
//! - Decomposing gates to native gate sets
//! - Mapping qubits for limited connectivity
//! - Inserting SWAP gates when needed
//! - Optimizing the resulting circuit
//!
//! # Example
//!
//! ```ignore
//! use simq_backend::{Transpiler, OptimizationLevel};
//!
//! let transpiler = Transpiler::new(OptimizationLevel::Medium);
//! let transpiled = transpiler.transpile(&circuit, &backend.capabilities());
//! ```

use crate::{BackendCapabilities, BackendError, GateDecomposer, Result, Router, RoutingStrategy};
use simq_core::{Circuit, QubitId};
use std::collections::HashMap;
use std::sync::Arc;

/// Transpiler for converting circuits to backend-specific formats
///
/// The transpiler performs several transformation passes:
/// 1. **Gate Decomposition**: Convert gates to native gate set
/// 2. **Qubit Mapping**: Map logical to physical qubits
/// 3. **SWAP Insertion**: Handle connectivity constraints
/// 4. **Optimization**: Reduce gate count and depth
pub struct Transpiler {
    /// Optimization level
    optimization_level: OptimizationLevel,

    /// Gate decomposition rules
    decomposition_rules: DecompositionRules,

    /// Whether to use approximations for gate decomposition
    allow_approximations: bool,
}

impl Transpiler {
    /// Create a new transpiler with the specified optimization level
    pub fn new(optimization_level: OptimizationLevel) -> Self {
        Self {
            optimization_level,
            decomposition_rules: DecompositionRules::default(),
            allow_approximations: false,
        }
    }

    /// Enable approximations for gate decomposition
    pub fn with_approximations(mut self, allow: bool) -> Self {
        self.allow_approximations = allow;
        self
    }

    /// Add custom decomposition rule
    pub fn add_decomposition_rule(&mut self, gate_name: &str, rule: DecompositionRule) {
        self.decomposition_rules
            .add_rule(gate_name.to_string(), rule);
    }

    /// Transpile a circuit for a specific backend
    ///
    /// # Arguments
    ///
    /// * `circuit` - The input circuit
    /// * `capabilities` - Target backend capabilities
    ///
    /// # Returns
    ///
    /// A transpiled circuit compatible with the backend
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Circuit requires more qubits than backend supports
    /// - Gates cannot be decomposed to native set
    /// - No valid qubit mapping exists
    pub fn transpile(
        &self,
        circuit: &Circuit,
        capabilities: &BackendCapabilities,
    ) -> Result<Circuit> {
        self.transpile_with_stats(circuit, capabilities)
            .map(|(transpiled, _swap_gates)| transpiled)
    }

    /// Transpile a circuit, also returning the number of SWAP gates inserted
    /// during routing (0 if the backend has no connectivity constraints).
    fn transpile_with_stats(
        &self,
        circuit: &Circuit,
        capabilities: &BackendCapabilities,
    ) -> Result<(Circuit, usize)> {
        // Validate circuit can fit on backend
        if circuit.num_qubits() > capabilities.max_qubits {
            return Err(BackendError::CapabilityExceeded(format!(
                "Circuit requires {} qubits, backend supports {}",
                circuit.num_qubits(),
                capabilities.max_qubits
            )));
        }

        let mut transpiled = circuit.clone();

        // Step 1: Decompose to native gates
        if self.optimization_level != OptimizationLevel::None {
            transpiled = self.decompose_to_native(&transpiled, capabilities)?;
        }

        // Step 2: Map to physical qubits (if connectivity constraints exist)
        let mut swap_gates = 0;
        if capabilities.connectivity.is_some() {
            let (routed, inserted_swaps) = self.map_and_route(&transpiled, capabilities)?;
            transpiled = routed;
            swap_gates = inserted_swaps;
        }

        // Step 3: Optimize based on level
        transpiled = match self.optimization_level {
            OptimizationLevel::None => transpiled,
            OptimizationLevel::Light => self.optimize_light(&transpiled)?,
            OptimizationLevel::Medium => self.optimize_medium(&transpiled)?,
            OptimizationLevel::Heavy => self.optimize_heavy(&transpiled)?,
        };

        Ok((transpiled, swap_gates))
    }

    /// Decompose gates to native gate set
    ///
    /// Converts all gates in the circuit to the backend's native gate set,
    /// using the decomposition rules appropriate for `capabilities.native_gates`
    /// (see [`GateDecomposer::for_capabilities`]). Errors if a gate is outside
    /// the native set and no decomposition rule applies.
    fn decompose_to_native(
        &self,
        circuit: &Circuit,
        capabilities: &BackendCapabilities,
    ) -> Result<Circuit> {
        GateDecomposer::for_capabilities(&capabilities.native_gates).decompose_circuit(circuit)
    }

    /// Map logical qubits to physical qubits and insert SWAPs
    ///
    /// Uses an identity initial mapping (logical qubit `i` starts on physical
    /// qubit `i`), then walks the circuit inserting a SWAP chain ahead of any
    /// two-qubit gate whose qubits are not adjacent under the current
    /// mapping (via [`Router::find_swap_chain`]). Returns the routed circuit
    /// along with the number of SWAP gates inserted.
    ///
    /// Gates with three or more qubits are mapped directly without routing;
    /// connectivity-aware routing for multi-qubit gates is not implemented.
    fn map_and_route(
        &self,
        circuit: &Circuit,
        capabilities: &BackendCapabilities,
    ) -> Result<(Circuit, usize)> {
        let connectivity = capabilities
            .connectivity
            .as_ref()
            .ok_or_else(|| BackendError::Other("No connectivity graph".to_string()))?;

        if circuit.num_qubits() > connectivity.num_qubits() {
            return Err(BackendError::CapabilityExceeded(format!(
                "Circuit requires {} qubits, connectivity graph has {}",
                circuit.num_qubits(),
                connectivity.num_qubits()
            )));
        }

        let router = Router::new(RoutingStrategy::Identity);
        let mut mapping = router.initial_mapping(circuit.num_qubits(), connectivity)?;

        let mut routed = Circuit::with_capacity(connectivity.num_qubits(), circuit.len());
        let mut swap_count = 0;

        for op in circuit.operations() {
            let logical_qubits = op.qubits();

            if logical_qubits.len() == 2 {
                let swaps = router.find_swap_chain(
                    connectivity,
                    &mapping,
                    logical_qubits[0].index(),
                    logical_qubits[1].index(),
                )?;

                for swap in &swaps {
                    routed.add_gate(
                        Arc::new(simq_gates::Swap),
                        &[QubitId::new(swap.qubit1), QubitId::new(swap.qubit2)],
                    )?;
                    swap.apply(&mut mapping);
                    swap_count += 1;
                }
            }

            let physical_qubits = logical_qubits
                .iter()
                .map(|q| {
                    mapping
                        .get_physical(q.index())
                        .map(QubitId::new)
                        .ok_or_else(|| {
                            BackendError::Other(format!(
                                "No physical qubit mapped for logical qubit {}",
                                q.index()
                            ))
                        })
                })
                .collect::<Result<Vec<_>>>()?;

            routed.add_gate(op.gate().clone(), &physical_qubits)?;
        }

        Ok((routed, swap_count))
    }

    /// Light optimization pass
    ///
    /// - Repeatedly cancels adjacent inverse gate pairs (H-H, X-X, CNOT-CNOT,
    ///   T-Tdg, S-Sdg, ...) until no more cancellations apply, via
    ///   [`crate::gate_decomposition::optimize_inverse_gates`].
    /// - Attempts to merge adjacent same-axis single-qubit rotations via
    ///   [`crate::gate_decomposition::optimize_merge_rotations`], which errors
    ///   loudly (rather than silently skipping) when it finds a mergeable
    ///   pair it cannot yet merge.
    fn optimize_light(&self, circuit: &Circuit) -> Result<Circuit> {
        let mut current = circuit.clone();
        loop {
            let next = crate::gate_decomposition::optimize_inverse_gates(&current)?;
            let converged = next.len() == current.len();
            current = next;
            if converged {
                break;
            }
        }

        crate::gate_decomposition::optimize_merge_rotations(&current)
    }

    /// Medium optimization pass
    ///
    /// Currently equivalent to [`Self::optimize_light`]: commutation-based
    /// reordering and template matching are not yet implemented, so this is
    /// a pass-through beyond the light optimizations rather than the fuller
    /// pipeline described on [`OptimizationLevel::Medium`].
    fn optimize_medium(&self, circuit: &Circuit) -> Result<Circuit> {
        self.optimize_light(circuit)
    }

    /// Heavy optimization pass
    ///
    /// Currently equivalent to [`Self::optimize_medium`]: global circuit
    /// resynthesis and advanced template matching are not yet implemented,
    /// so this is a pass-through beyond the medium optimizations rather than
    /// the fuller pipeline described on [`OptimizationLevel::Heavy`].
    fn optimize_heavy(&self, circuit: &Circuit) -> Result<Circuit> {
        self.optimize_medium(circuit)
    }

    /// Estimate the cost of the transpiled circuit
    pub fn estimate_cost(
        &self,
        circuit: &Circuit,
        capabilities: &BackendCapabilities,
    ) -> TranspilationCost {
        let original_gates = circuit.len();
        let original_depth = circuit.depth();

        let transpiled = self.transpile_with_stats(circuit, capabilities);
        let (transpiled_gates, transpiled_depth, swap_gates) = match transpiled {
            Ok((ref tc, swap_gates)) => (tc.len(), tc.depth(), swap_gates),
            Err(_) => (original_gates, original_depth, 0),
        };

        TranspilationCost {
            original_gates,
            transpiled_gates,
            original_depth,
            transpiled_depth,
            swap_gates,
        }
    }
}

impl Default for Transpiler {
    fn default() -> Self {
        Self::new(OptimizationLevel::Medium)
    }
}

/// Optimization level for transpilation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationLevel {
    /// No optimization, minimal transpilation
    /// - Only validate circuit
    /// - No gate decomposition or optimization
    None,

    /// Light optimization (fast, <100ms for typical circuits)
    /// - Adjacent inverse gate-pair cancellation (H-H, X-X, CNOT-CNOT, ...)
    /// - Same-axis single-qubit rotation merging: attempted, errors loudly
    ///   if a mergeable pair is found that cannot yet be merged
    Light,

    /// Medium optimization (balanced, <1s for typical circuits)
    /// - Light optimizations
    /// - Commutation-based reordering: **not yet implemented** (pass-through)
    /// - Template matching: **not yet implemented** (pass-through)
    #[default]
    Medium,

    /// Heavy optimization (slow, may take several seconds)
    /// - Medium optimizations
    /// - Global resynthesis: **not yet implemented** (pass-through)
    /// - Advanced template matching: **not yet implemented** (pass-through)
    Heavy,
}

/// Cost estimate for transpilation
#[derive(Debug, Clone, Copy)]
pub struct TranspilationCost {
    /// Original number of gates
    pub original_gates: usize,

    /// Number of gates after transpilation
    pub transpiled_gates: usize,

    /// Original circuit depth
    pub original_depth: usize,

    /// Circuit depth after transpilation
    pub transpiled_depth: usize,

    /// Number of SWAP gates inserted
    pub swap_gates: usize,
}

impl TranspilationCost {
    /// Gate overhead factor
    pub fn gate_overhead(&self) -> f64 {
        if self.original_gates == 0 {
            return 0.0;
        }
        (self.transpiled_gates as f64 - self.original_gates as f64) / self.original_gates as f64
    }

    /// Depth overhead factor
    pub fn depth_overhead(&self) -> f64 {
        if self.original_depth == 0 {
            return 0.0;
        }
        (self.transpiled_depth as f64 - self.original_depth as f64) / self.original_depth as f64
    }
}

/// Gate decomposition rules
pub struct DecompositionRules {
    rules: HashMap<String, DecompositionRule>,
}

impl DecompositionRules {
    /// Create empty decomposition rules
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }

    /// Add a decomposition rule
    pub fn add_rule(&mut self, gate_name: String, rule: DecompositionRule) {
        self.rules.insert(gate_name, rule);
    }

    /// Get decomposition rule for a gate
    pub fn get_rule(&self, gate_name: &str) -> Option<&DecompositionRule> {
        self.rules.get(gate_name)
    }

    /// Check if a gate has a decomposition rule
    pub fn has_rule(&self, gate_name: &str) -> bool {
        self.rules.contains_key(gate_name)
    }
}

impl Default for DecompositionRules {
    fn default() -> Self {
        let mut rules = Self::new();

        // Add standard decomposition rules
        // These are common gate decompositions to universal gate sets

        // Hadamard to RZ-SX (IBM native)
        rules.add_rule(
            "H".to_string(),
            DecompositionRule {
                description: "H → RZ(π/2) SX RZ(π/2)".to_string(),
                target_gates: vec!["RZ".to_string(), "SX".to_string()],
                gate_count: 3,
            },
        );

        // T gate decomposition
        rules.add_rule(
            "T".to_string(),
            DecompositionRule {
                description: "T → RZ(π/4)".to_string(),
                target_gates: vec!["RZ".to_string()],
                gate_count: 1,
            },
        );

        // S gate decomposition
        rules.add_rule(
            "S".to_string(),
            DecompositionRule {
                description: "S → RZ(π/2)".to_string(),
                target_gates: vec!["RZ".to_string()],
                gate_count: 1,
            },
        );

        // CZ decomposition to CNOT
        rules.add_rule(
            "CZ".to_string(),
            DecompositionRule {
                description: "CZ → H CNOT H".to_string(),
                target_gates: vec!["H".to_string(), "CNOT".to_string()],
                gate_count: 3,
            },
        );

        // SWAP decomposition to CNOTs
        rules.add_rule(
            "SWAP".to_string(),
            DecompositionRule {
                description: "SWAP → CNOT CNOT CNOT".to_string(),
                target_gates: vec!["CNOT".to_string()],
                gate_count: 3,
            },
        );

        // Toffoli (CCX) decomposition
        rules.add_rule(
            "Toffoli".to_string(),
            DecompositionRule {
                description: "Toffoli → 2CNOTs + single-qubit gates".to_string(),
                target_gates: vec!["CNOT".to_string(), "H".to_string(), "T".to_string()],
                gate_count: 15, // Standard Toffoli decomposition
            },
        );

        rules
    }
}

/// A gate decomposition rule
#[derive(Debug, Clone)]
pub struct DecompositionRule {
    /// Description of the decomposition
    pub description: String,

    /// Target gates used in the decomposition
    pub target_gates: Vec<String>,

    /// Number of gates in the decomposition
    pub gate_count: usize,
}

/// Qubit mapping for transpilation
#[derive(Debug, Clone)]
pub struct QubitMapping {
    /// Logical to physical qubit mapping
    logical_to_physical: Vec<usize>,

    /// Physical to logical qubit mapping
    physical_to_logical: Vec<Option<usize>>,
}

impl QubitMapping {
    /// Create identity mapping
    pub fn identity(num_qubits: usize) -> Self {
        Self {
            logical_to_physical: (0..num_qubits).collect(),
            physical_to_logical: (0..num_qubits).map(Some).collect(),
        }
    }

    /// Create mapping from vector
    pub fn from_vec(mapping: Vec<usize>, num_physical: usize) -> Self {
        let mut physical_to_logical = vec![None; num_physical];
        for (logical, &physical) in mapping.iter().enumerate() {
            physical_to_logical[physical] = Some(logical);
        }

        Self {
            logical_to_physical: mapping,
            physical_to_logical,
        }
    }

    /// Get physical qubit for logical qubit
    pub fn get_physical(&self, logical: usize) -> Option<usize> {
        self.logical_to_physical.get(logical).copied()
    }

    /// Get logical qubit for physical qubit
    pub fn get_logical(&self, physical: usize) -> Option<usize> {
        self.physical_to_logical.get(physical).and_then(|&l| l)
    }

    /// Swap two physical qubits
    pub fn swap(&mut self, phys1: usize, phys2: usize) {
        if let (Some(log1), Some(log2)) =
            (self.physical_to_logical[phys1], self.physical_to_logical[phys2])
        {
            self.logical_to_physical[log1] = phys2;
            self.logical_to_physical[log2] = phys1;
        }
        self.physical_to_logical.swap(phys1, phys2);
    }
}

/// SWAP insertion strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SwapStrategy {
    /// Greedy: Insert SWAPs along shortest path
    #[default]
    Greedy,

    /// SABRE: Stochastic routing algorithm
    Sabre,

    /// Lookahead: Consider future gates when inserting SWAPs
    Lookahead,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpiler_creation() {
        let transpiler = Transpiler::default();
        assert_eq!(transpiler.optimization_level, OptimizationLevel::Medium);
    }

    #[test]
    fn test_optimization_levels() {
        let none = Transpiler::new(OptimizationLevel::None);
        let light = Transpiler::new(OptimizationLevel::Light);
        let medium = Transpiler::new(OptimizationLevel::Medium);
        let heavy = Transpiler::new(OptimizationLevel::Heavy);

        assert_eq!(none.optimization_level, OptimizationLevel::None);
        assert_eq!(light.optimization_level, OptimizationLevel::Light);
        assert_eq!(medium.optimization_level, OptimizationLevel::Medium);
        assert_eq!(heavy.optimization_level, OptimizationLevel::Heavy);
    }

    #[test]
    fn test_decomposition_rules() {
        let rules = DecompositionRules::default();

        // Check standard rules exist
        assert!(rules.has_rule("H"));
        assert!(rules.has_rule("T"));
        assert!(rules.has_rule("S"));
        assert!(rules.has_rule("CZ"));
        assert!(rules.has_rule("SWAP"));
        assert!(rules.has_rule("Toffoli"));

        // Check rule details
        let h_rule = rules.get_rule("H").unwrap();
        assert_eq!(h_rule.gate_count, 3);
        assert!(h_rule.target_gates.contains(&"RZ".to_string()));
        assert!(h_rule.target_gates.contains(&"SX".to_string()));
    }

    #[test]
    fn test_qubit_mapping_identity() {
        let mapping = QubitMapping::identity(5);

        for i in 0..5 {
            assert_eq!(mapping.get_physical(i), Some(i));
            assert_eq!(mapping.get_logical(i), Some(i));
        }
    }

    #[test]
    fn test_qubit_mapping_custom() {
        let mapping = QubitMapping::from_vec(vec![2, 0, 1], 5);

        assert_eq!(mapping.get_physical(0), Some(2));
        assert_eq!(mapping.get_physical(1), Some(0));
        assert_eq!(mapping.get_physical(2), Some(1));

        assert_eq!(mapping.get_logical(2), Some(0));
        assert_eq!(mapping.get_logical(0), Some(1));
        assert_eq!(mapping.get_logical(1), Some(2));
    }

    #[test]
    fn test_qubit_mapping_swap() {
        let mut mapping = QubitMapping::identity(3);

        // Initially: L0→P0, L1→P1, L2→P2
        assert_eq!(mapping.get_physical(0), Some(0));
        assert_eq!(mapping.get_physical(1), Some(1));

        // Swap physical qubits 0 and 1
        mapping.swap(0, 1);

        // After swap: L0→P1, L1→P0, L2→P2
        assert_eq!(mapping.get_physical(0), Some(1));
        assert_eq!(mapping.get_physical(1), Some(0));
        assert_eq!(mapping.get_logical(0), Some(1));
        assert_eq!(mapping.get_logical(1), Some(0));
    }

    #[test]
    fn test_transpilation_cost() {
        let cost = TranspilationCost {
            original_gates: 100,
            transpiled_gates: 150,
            original_depth: 20,
            transpiled_depth: 28,
            swap_gates: 5,
        };

        assert_eq!(cost.gate_overhead(), 0.5); // 50% overhead
        assert_eq!(cost.depth_overhead(), 0.4); // 40% overhead
    }

    #[test]
    fn test_custom_decomposition_rule() {
        let mut transpiler = Transpiler::default();

        transpiler.add_decomposition_rule(
            "CustomGate",
            DecompositionRule {
                description: "CustomGate → X Y Z".to_string(),
                target_gates: vec!["X".to_string(), "Y".to_string(), "Z".to_string()],
                gate_count: 3,
            },
        );

        let rules = &transpiler.decomposition_rules;
        assert!(rules.has_rule("CustomGate"));
        let rule = rules.get_rule("CustomGate").unwrap();
        assert_eq!(rule.gate_count, 3);
    }

    // Tests for previously uncovered lines

    #[test]
    fn test_transpile_too_many_qubits() {
        // Covers lines 87-91: circuit exceeds backend qubit count
        let transpiler = Transpiler::new(OptimizationLevel::None);
        let mut caps = crate::BackendCapabilities::simulator();
        caps.max_qubits = 3;

        let circuit = simq_core::Circuit::new(5);
        let result = transpiler.transpile(&circuit, &caps);
        assert!(result.is_err());
        match result.unwrap_err() {
            crate::BackendError::CapabilityExceeded(msg) => {
                assert!(msg.contains("5 qubits"));
            },
            other => panic!("Expected CapabilityExceeded, got {:?}", other),
        }
    }

    #[test]
    fn test_transpile_with_connectivity() {
        // Covers lines 158-161: map_and_route called when connectivity is set
        use crate::ConnectivityGraph;

        let transpiler = Transpiler::new(OptimizationLevel::None);
        let mut caps = crate::BackendCapabilities::simulator();
        caps.connectivity = Some(ConnectivityGraph::all_to_all(5));

        let circuit = simq_core::Circuit::new(3);
        let result = transpiler.transpile(&circuit, &caps);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transpile_map_route_qubit_mismatch() {
        // map_and_route error when circuit needs more qubits than connectivity graph
        use crate::ConnectivityGraph;

        let transpiler = Transpiler::new(OptimizationLevel::None);
        let mut caps = crate::BackendCapabilities::simulator();
        // Connectivity graph has only 2 qubits, circuit has 3
        caps.max_qubits = 10; // allow the first check to pass
        caps.connectivity = Some(ConnectivityGraph::linear_chain(2));

        let circuit = simq_core::Circuit::new(3);
        let result = transpiler.transpile(&circuit, &caps);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimate_cost_on_valid_circuit() {
        // Covers line 233: estimate_cost Ok branch
        let transpiler = Transpiler::new(OptimizationLevel::Medium);
        let caps = crate::BackendCapabilities::simulator();
        let circuit = simq_core::Circuit::new(2);

        let cost = transpiler.estimate_cost(&circuit, &caps);
        // Circuit is empty so transpile succeeds
        assert_eq!(cost.original_gates, 0);
        assert_eq!(cost.original_depth, 0);
    }

    #[test]
    fn test_transpile_heavy_optimization() {
        // Covers OptimizationLevel::Heavy path
        let transpiler = Transpiler::new(OptimizationLevel::Heavy);
        let caps = crate::BackendCapabilities::simulator();
        let circuit = simq_core::Circuit::new(2);

        let result = transpiler.transpile(&circuit, &caps);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transpile_light_optimization() {
        // Covers OptimizationLevel::Light path
        let transpiler = Transpiler::new(OptimizationLevel::Light);
        let caps = crate::BackendCapabilities::simulator();
        let circuit = simq_core::Circuit::new(2);

        let result = transpiler.transpile(&circuit, &caps);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transpilation_cost_zero_depth_overhead() {
        // Covers depth_overhead with zero original_depth
        let cost = TranspilationCost {
            original_gates: 5,
            transpiled_gates: 7,
            original_depth: 0,
            transpiled_depth: 0,
            swap_gates: 0,
        };
        assert_eq!(cost.depth_overhead(), 0.0);
    }

    #[test]
    fn test_transpilation_cost_zero_gate_overhead() {
        // Covers gate_overhead with zero original_gates
        let cost = TranspilationCost {
            original_gates: 0,
            transpiled_gates: 0,
            original_depth: 0,
            transpiled_depth: 0,
            swap_gates: 0,
        };
        assert_eq!(cost.gate_overhead(), 0.0);
    }

    #[test]
    fn test_with_approximations() {
        let transpiler = Transpiler::new(OptimizationLevel::Medium).with_approximations(true);
        assert!(transpiler.allow_approximations);
    }

    // Tests for issue #49: decomposition, routing, and optimization passes
    // must do real work instead of returning the circuit unchanged.

    #[test]
    fn test_decompose_to_native_ibm_real_decomposition() {
        use simq_gates::Hadamard;

        let transpiler = Transpiler::new(OptimizationLevel::Light);
        let caps =
            crate::BackendCapabilities::ibm_quantum(5, crate::ConnectivityGraph::all_to_all(5));

        let mut circuit = simq_core::Circuit::new(1);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();

        // H is not in IBM's native gate set, so it must be broken down
        // instead of passed through unchanged.
        let decomposed = transpiler.decompose_to_native(&circuit, &caps).unwrap();

        assert_eq!(decomposed.len(), 3);
        let names: Vec<_> = decomposed.operations().map(|op| op.gate().name()).collect();
        assert_eq!(names, vec!["RZ", "SX", "RZ"]);
    }

    #[test]
    fn test_decompose_to_native_errors_on_unsupported_gate() {
        use simq_gates::Toffoli;

        let transpiler = Transpiler::new(OptimizationLevel::Light);
        let caps =
            crate::BackendCapabilities::ibm_quantum(5, crate::ConnectivityGraph::all_to_all(5));

        let mut circuit = simq_core::Circuit::new(3);
        circuit
            .add_gate(Arc::new(Toffoli), &[QubitId::new(0), QubitId::new(1), QubitId::new(2)])
            .unwrap();

        // Toffoli is neither IBM-native nor covered by a registered
        // decomposition rule, so this must fail loudly rather than silently
        // forward an incompatible gate.
        let result = transpiler.decompose_to_native(&circuit, &caps);
        assert!(result.is_err());
    }

    #[test]
    fn test_map_and_route_inserts_swaps_for_distant_qubits() {
        use simq_gates::CNot;

        let transpiler = Transpiler::new(OptimizationLevel::None);
        let mut caps = crate::BackendCapabilities::simulator();
        caps.connectivity = Some(crate::ConnectivityGraph::linear_chain(5));

        let mut circuit = simq_core::Circuit::new(5);
        circuit
            .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(4)])
            .unwrap();

        let (routed, swap_count) = transpiler.map_and_route(&circuit, &caps).unwrap();

        // Qubits 0 and 4 are 4 hops apart on a linear chain, so 3 SWAPs are
        // needed to bring them adjacent before the CNOT can execute.
        assert_eq!(swap_count, 3);
        assert_eq!(routed.len(), 4); // 3 SWAPs + 1 CNOT
        assert_eq!(routed.gate_counts().get("SWAP"), Some(&3));
        assert_eq!(routed.gate_counts().get("CNOT"), Some(&1));
    }

    #[test]
    fn test_map_and_route_no_swaps_when_already_connected() {
        use simq_gates::CNot;

        let transpiler = Transpiler::new(OptimizationLevel::None);
        let mut caps = crate::BackendCapabilities::simulator();
        caps.connectivity = Some(crate::ConnectivityGraph::linear_chain(5));

        let mut circuit = simq_core::Circuit::new(5);
        circuit
            .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();

        let (routed, swap_count) = transpiler.map_and_route(&circuit, &caps).unwrap();

        assert_eq!(swap_count, 0);
        assert_eq!(routed.len(), 1);
    }

    #[test]
    fn test_optimize_light_cancels_adjacent_inverse_gates() {
        use simq_gates::Hadamard;

        let transpiler = Transpiler::default();
        let mut circuit = simq_core::Circuit::new(1);
        let q0 = QubitId::new(0);
        circuit.add_gate(Arc::new(Hadamard), &[q0]).unwrap();
        circuit.add_gate(Arc::new(Hadamard), &[q0]).unwrap();

        let optimized = transpiler.optimize_light(&circuit).unwrap();
        assert_eq!(optimized.len(), 0);
    }

    #[test]
    fn test_optimize_medium_and_heavy_are_light_pass_through() {
        use simq_gates::Hadamard;

        let transpiler = Transpiler::default();
        let mut circuit = simq_core::Circuit::new(1);
        let q0 = QubitId::new(0);
        circuit.add_gate(Arc::new(Hadamard), &[q0]).unwrap();
        circuit.add_gate(Arc::new(Hadamard), &[q0]).unwrap();

        assert_eq!(transpiler.optimize_medium(&circuit).unwrap().len(), 0);
        assert_eq!(transpiler.optimize_heavy(&circuit).unwrap().len(), 0);
    }

    #[test]
    fn test_estimate_cost_reports_swap_gates() {
        use simq_gates::CNot;

        let transpiler = Transpiler::new(OptimizationLevel::None);
        let mut caps = crate::BackendCapabilities::simulator();
        caps.connectivity = Some(crate::ConnectivityGraph::linear_chain(5));

        let mut circuit = simq_core::Circuit::new(5);
        circuit
            .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(4)])
            .unwrap();

        let cost = transpiler.estimate_cost(&circuit, &caps);
        assert_eq!(cost.swap_gates, 3);
    }
}
