//! Comprehensive end-to-end tests for simq-backend crate
//!
//! Covers: BackendCapabilities, ConnectivityGraph, GateSet, BackendResult,
//! ExecutionMetadata, JobStatus, GateDecomposer, Transpiler, Router,
//! SabreRouter, QubitMapping, BackendSelector, LocalSimulatorBackend,
//! error handling, optimization, and integration tests.

use simq_backend::*;
use simq_core::circuit::Circuit;
use simq_core::QubitId;
use simq_gates::standard::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

fn q(i: usize) -> QubitId {
    QubitId::new(i)
}

fn bell_circuit() -> Circuit {
    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();
    c
}

fn ghz_circuit(n: usize) -> Circuit {
    let mut c = Circuit::new(n);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    for i in 0..n - 1 {
        c.add_gate(Arc::new(CNot), &[q(i), q(i + 1)]).unwrap();
    }
    c
}

fn single_x_circuit() -> Circuit {
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    c
}

// ============================================================================
// 1. GateSet
// ============================================================================

#[test]
fn gateset_new_is_empty() {
    let gs = GateSet::new();
    assert!(gs.is_empty());
    assert_eq!(gs.len(), 0);
}

#[test]
fn gateset_insert_and_contains() {
    let mut gs = GateSet::new();
    gs.insert("H".to_string());
    gs.insert("CNOT".to_string());
    assert!(gs.contains(&"H".to_string()));
    assert!(gs.contains(&"CNOT".to_string()));
    assert!(!gs.contains(&"T".to_string()));
    assert_eq!(gs.len(), 2);
}

#[test]
fn gateset_universal() {
    let gs = GateSet::universal();
    assert!(!gs.is_empty());
    assert!(gs.contains(&"H".to_string()));
    assert!(gs.contains(&"CNOT".to_string()));
    assert!(gs.contains(&"X".to_string()));
}

#[test]
fn gateset_gates_returns_all() {
    let mut gs = GateSet::new();
    gs.insert("A".to_string());
    gs.insert("B".to_string());
    let gates = gs.gates();
    assert_eq!(gates.len(), 2);
    assert!(gates.contains(&"A".to_string()));
    assert!(gates.contains(&"B".to_string()));
}

// ============================================================================
// 2. ConnectivityGraph
// ============================================================================

#[test]
fn connectivity_all_to_all() {
    let cg = ConnectivityGraph::all_to_all(4);
    assert_eq!(cg.num_qubits(), 4);
    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                assert!(cg.are_connected(i, j), "qubits {} and {} should be connected", i, j);
            }
        }
    }
}

#[test]
fn connectivity_linear_chain() {
    let cg = ConnectivityGraph::linear_chain(5);
    assert_eq!(cg.num_qubits(), 5);
    assert!(cg.are_connected(0, 1));
    assert!(cg.are_connected(1, 2));
    assert!(cg.are_connected(2, 3));
    assert!(cg.are_connected(3, 4));
    assert!(!cg.are_connected(0, 2));
    assert!(!cg.are_connected(0, 4));
}

#[test]
fn connectivity_grid() {
    let cg = ConnectivityGraph::grid(2, 3);
    assert_eq!(cg.num_qubits(), 6);
    // Row 0: 0-1-2, Row 1: 3-4-5
    assert!(cg.are_connected(0, 1));
    assert!(cg.are_connected(1, 2));
    assert!(cg.are_connected(0, 3)); // column connection
    assert!(cg.are_connected(1, 4));
    assert!(cg.are_connected(2, 5));
    assert!(!cg.are_connected(0, 4)); // diagonal: not connected
}

#[test]
fn connectivity_add_edge() {
    let mut cg = ConnectivityGraph::new(3, false);
    assert!(!cg.are_connected(0, 1));
    cg.add_edge(0, 1);
    assert!(cg.are_connected(0, 1));
    assert!(cg.are_connected(1, 0)); // undirected
}

#[test]
fn connectivity_directed() {
    let mut cg = ConnectivityGraph::new(3, true);
    assert!(cg.is_directed());
    cg.add_edge(0, 1);
    assert!(cg.are_connected(0, 1));
    // Directed graph: reverse might not be connected
    // (depends on implementation - check both directions)
}

#[test]
fn connectivity_neighbors() {
    let cg = ConnectivityGraph::linear_chain(3);
    let n = cg.neighbors(1).unwrap();
    assert!(n.contains(&0));
    assert!(n.contains(&2));
    assert_eq!(n.len(), 2);
}

#[test]
fn connectivity_degree() {
    let cg = ConnectivityGraph::linear_chain(5);
    assert_eq!(cg.degree(0), 1); // endpoint
    assert_eq!(cg.degree(2), 2); // middle
    assert_eq!(cg.degree(4), 1); // endpoint
}

#[test]
fn connectivity_shortest_path() {
    let cg = ConnectivityGraph::linear_chain(5);
    let path = cg.shortest_path(0, 4).unwrap();
    assert_eq!(path.len(), 5); // 0→1→2→3→4
    assert_eq!(path[0], 0);
    assert_eq!(path[4], 4);
}

#[test]
fn connectivity_shortest_path_same() {
    let cg = ConnectivityGraph::linear_chain(3);
    let path = cg.shortest_path(1, 1).unwrap();
    assert_eq!(path.len(), 1);
    assert_eq!(path[0], 1);
}

// ============================================================================
// 3. BackendCapabilities
// ============================================================================

#[test]
fn capabilities_simulator_defaults() {
    let cap = BackendCapabilities::simulator();
    assert!(cap.max_qubits > 0);
    assert!(cap.supports_mid_circuit_measurement);
    assert!(cap.supports_conditional);
    assert!(cap.supports_reset);
    assert!(cap.supports_parametric);
}

#[test]
fn capabilities_default() {
    let cap = BackendCapabilities::default();
    assert_eq!(cap.max_qubits, 64);
}

#[test]
fn capabilities_ibm_quantum() {
    let cg = ConnectivityGraph::linear_chain(5);
    let cap = BackendCapabilities::ibm_quantum(5, cg);
    assert_eq!(cap.max_qubits, 5);
    assert!(cap.connectivity.is_some());
}

#[test]
fn capabilities_supports_gate() {
    let cap = BackendCapabilities::simulator();
    assert!(cap.supports_gate("H".to_string()));
}

#[test]
fn capabilities_is_native_gate() {
    let cg = ConnectivityGraph::linear_chain(5);
    let cap = BackendCapabilities::ibm_quantum(5, cg);
    assert!(cap.is_native_gate("CNOT".to_string()));
}

#[test]
fn capabilities_qubits_connected() {
    let cg = ConnectivityGraph::linear_chain(5);
    let cap = BackendCapabilities::ibm_quantum(5, cg);
    assert!(cap.are_qubits_connected(0, 1));
    assert!(!cap.are_qubits_connected(0, 3));
}

#[test]
fn capabilities_all_to_all_connected() {
    let cap = BackendCapabilities::simulator();
    // Simulator has no connectivity graph → all-to-all
    assert!(cap.are_qubits_connected(0, 10));
}

// ============================================================================
// 4. JobStatus
// ============================================================================

#[test]
fn job_status_default_is_queued() {
    let status = JobStatus::default();
    assert_eq!(status, JobStatus::Queued);
}

#[test]
fn job_status_display() {
    assert!(!format!("{}", JobStatus::Completed).is_empty());
    assert!(!format!("{}", JobStatus::Failed).is_empty());
    assert!(!format!("{}", JobStatus::Running).is_empty());
}

#[test]
fn job_status_all_variants() {
    let _ = JobStatus::Queued;
    let _ = JobStatus::Validating;
    let _ = JobStatus::Running;
    let _ = JobStatus::Completed;
    let _ = JobStatus::Failed;
    let _ = JobStatus::Cancelled;
}

// ============================================================================
// 5. ExecutionMetadata
// ============================================================================

#[test]
fn metadata_default() {
    let meta = ExecutionMetadata::default();
    assert_eq!(meta.status, JobStatus::Queued);
    assert!(meta.execution_time.is_none());
    assert!(meta.backend_name.is_none());
}

#[test]
fn metadata_success() {
    let meta = ExecutionMetadata::success("test-backend".to_string(), Duration::from_millis(100));
    assert!(meta.is_success());
    assert!(!meta.is_failed());
    assert_eq!(meta.backend_name.as_deref(), Some("test-backend"));
    assert!(meta.execution_time.is_some());
}

#[test]
fn metadata_failed() {
    let meta = ExecutionMetadata::failed("something went wrong".to_string());
    assert!(meta.is_failed());
    assert!(!meta.is_success());
    assert!(meta.error_message.is_some());
}

// ============================================================================
// 6. BackendResult
// ============================================================================

#[test]
fn result_new() {
    let mut counts = HashMap::new();
    counts.insert("00".to_string(), 500);
    counts.insert("11".to_string(), 500);
    let result = BackendResult::new(counts, 1000);
    assert_eq!(result.shots, 1000);
}

#[test]
fn result_most_frequent() {
    let mut counts = HashMap::new();
    counts.insert("00".to_string(), 700);
    counts.insert("11".to_string(), 300);
    let result = BackendResult::new(counts, 1000);
    let (bitstring, count) = result.most_frequent().unwrap();
    assert_eq!(bitstring, "00");
    assert_eq!(*count, 700);
}

#[test]
fn result_probabilities() {
    let mut counts = HashMap::new();
    counts.insert("00".to_string(), 500);
    counts.insert("11".to_string(), 500);
    let result = BackendResult::new(counts, 1000);
    let probs = result.probabilities();
    assert!((probs["00"] - 0.5).abs() < 1e-10);
    assert!((probs["11"] - 0.5).abs() < 1e-10);
}

#[test]
fn result_get_count() {
    let mut counts = HashMap::new();
    counts.insert("01".to_string(), 42);
    let result = BackendResult::new(counts, 100);
    assert_eq!(result.get_count("01"), 42);
    assert_eq!(result.get_count("10"), 0);
}

#[test]
fn result_bitstrings() {
    let mut counts = HashMap::new();
    counts.insert("00".to_string(), 50);
    counts.insert("01".to_string(), 30);
    counts.insert("11".to_string(), 20);
    let result = BackendResult::new(counts, 100);
    let bs = result.bitstrings();
    assert_eq!(bs.len(), 3);
}

#[test]
fn result_expectation_value() {
    let mut counts = HashMap::new();
    counts.insert("0".to_string(), 800);
    counts.insert("1".to_string(), 200);
    let result = BackendResult::new(counts, 1000);
    // Z operator: |0⟩ → +1, |1⟩ → -1
    let z_expectation = result.expectation_value(|bs| if bs == "0" { 1.0 } else { -1.0 });
    assert!((z_expectation - 0.6).abs() < 1e-10);
}

#[test]
fn result_merge() {
    let mut counts1 = HashMap::new();
    counts1.insert("00".to_string(), 300);
    counts1.insert("11".to_string(), 200);
    let mut r1 = BackendResult::new(counts1, 500);

    let mut counts2 = HashMap::new();
    counts2.insert("00".to_string(), 200);
    counts2.insert("11".to_string(), 300);
    let r2 = BackendResult::new(counts2, 500);

    r1.merge(&r2);
    assert_eq!(r1.shots, 1000);
    assert_eq!(r1.get_count("00"), 500);
    assert_eq!(r1.get_count("11"), 500);
}

// ============================================================================
// 7. BackendType
// ============================================================================

#[test]
fn backend_type_display() {
    assert!(!format!("{}", BackendType::Simulator).is_empty());
    assert!(!format!("{}", BackendType::Hardware).is_empty());
    assert!(!format!("{}", BackendType::CloudSimulator).is_empty());
}

#[test]
fn backend_type_equality() {
    assert_eq!(BackendType::Simulator, BackendType::Simulator);
    assert_ne!(BackendType::Simulator, BackendType::Hardware);
}

#[test]
fn backend_type_copy() {
    let t = BackendType::Simulator;
    let t2 = t;
    assert_eq!(t, t2);
}

// ============================================================================
// 8. BackendError
// ============================================================================

#[test]
fn error_circuit_incompatible() {
    let e = BackendError::CircuitIncompatible("too many qubits".to_string());
    assert!(format!("{}", e).contains("too many qubits"));
}

#[test]
fn error_capability_exceeded() {
    let e = BackendError::CapabilityExceeded("depth exceeded".to_string());
    assert!(format!("{}", e).contains("depth exceeded"));
}

#[test]
fn error_job_not_found() {
    let e = BackendError::JobNotFound {
        job_id: "abc123".to_string(),
    };
    assert!(format!("{}", e).contains("abc123"));
}

#[test]
fn error_job_timeout() {
    let e = BackendError::JobTimeout {
        timeout_seconds: 30,
    };
    assert!(format!("{}", e).contains("30"));
}

#[test]
fn error_from_serde() {
    let json_err = serde_json::from_str::<i32>("not json").unwrap_err();
    let be: BackendError = json_err.into();
    match be {
        BackendError::SerializationError(_) => {},
        _ => panic!("Expected SerializationError"),
    }
}

// ============================================================================
// 9. QubitMapping
// ============================================================================

#[test]
fn qubit_mapping_identity() {
    let m = QubitMapping::identity(4);
    for i in 0..4 {
        assert_eq!(m.get_physical(i), Some(i));
        assert_eq!(m.get_logical(i), Some(i));
    }
}

#[test]
fn qubit_mapping_from_vec() {
    let m = QubitMapping::from_vec(vec![2, 0, 1], 3);
    assert_eq!(m.get_physical(0), Some(2));
    assert_eq!(m.get_physical(1), Some(0));
    assert_eq!(m.get_physical(2), Some(1));
}

#[test]
fn qubit_mapping_swap() {
    let mut m = QubitMapping::identity(3);
    m.swap(0, 2);
    assert_eq!(m.get_logical(0), Some(2));
    assert_eq!(m.get_logical(2), Some(0));
    assert_eq!(m.get_logical(1), Some(1));
}

#[test]
fn qubit_mapping_out_of_range() {
    let m = QubitMapping::identity(3);
    assert_eq!(m.get_physical(5), None);
}

// ============================================================================
// 10. SwapGate
// ============================================================================

#[test]
fn swap_gate_create() {
    let s = SwapGate::new(1, 3);
    assert_eq!(s.qubit1, 1);
    assert_eq!(s.qubit2, 3);
}

#[test]
fn swap_gate_ordered() {
    let s = SwapGate::new(3, 1);
    assert_eq!(s.ordered(), (1, 3));
}

#[test]
fn swap_gate_apply() {
    let mut m = QubitMapping::identity(4);
    let s = SwapGate::new(0, 3);
    s.apply(&mut m);
    assert_eq!(m.get_logical(0), Some(3));
    assert_eq!(m.get_logical(3), Some(0));
}

#[test]
fn swap_gate_equality() {
    let s1 = SwapGate::new(0, 1);
    let s2 = SwapGate::new(0, 1);
    assert_eq!(s1, s2);
}

// ============================================================================
// 11. Router
// ============================================================================

#[test]
fn router_identity_strategy() {
    let router = Router::new(RoutingStrategy::Identity);
    let cg = ConnectivityGraph::linear_chain(5);
    let mapping = router.initial_mapping(3, &cg).unwrap();
    // Identity: logical i → physical i
    assert_eq!(mapping.get_physical(0), Some(0));
    assert_eq!(mapping.get_physical(1), Some(1));
    assert_eq!(mapping.get_physical(2), Some(2));
}

#[test]
fn router_highest_degree() {
    let router = Router::new(RoutingStrategy::HighestDegree);
    let cg = ConnectivityGraph::grid(3, 3);
    let mapping = router.initial_mapping(3, &cg).unwrap();
    // Should map to highest degree nodes
    for i in 0..3 {
        assert!(mapping.get_physical(i).is_some());
    }
}

#[test]
fn router_subgraph_strategy() {
    let router = Router::new(RoutingStrategy::Subgraph);
    let cg = ConnectivityGraph::grid(3, 3);
    let mapping = router.initial_mapping(4, &cg).unwrap();
    for i in 0..4 {
        assert!(mapping.get_physical(i).is_some());
    }
}

#[test]
fn router_find_swap_chain() {
    let router = Router::new(RoutingStrategy::Identity);
    let cg = ConnectivityGraph::linear_chain(5);
    let mapping = QubitMapping::identity(5);
    // Qubits 0 and 4 are far apart on linear chain
    let swaps = router.find_swap_chain(&cg, &mapping, 0, 4).unwrap();
    // Should return a chain of swaps to bring them adjacent
    assert!(!swaps.is_empty());
}

#[test]
fn router_swap_chain_adjacent() {
    let router = Router::new(RoutingStrategy::Identity);
    let cg = ConnectivityGraph::linear_chain(5);
    let mapping = QubitMapping::identity(5);
    // Already adjacent → should be empty or minimal
    let swaps = router.find_swap_chain(&cg, &mapping, 0, 1).unwrap();
    assert!(swaps.is_empty());
}

// ============================================================================
// 12. SabreRouter
// ============================================================================

#[test]
fn sabre_router_create() {
    let sr = SabreRouter::new(20, 0.99);
    let cg = ConnectivityGraph::linear_chain(5);
    let result = sr.route(3, &cg);
    // Honestly unimplemented: must error instead of claiming zero SWAPs.
    assert!(result.is_err());
}

#[test]
fn sabre_router_default() {
    let sr = SabreRouter::default();
    let cg = ConnectivityGraph::all_to_all(4);
    let result = sr.route(4, &cg);
    assert!(result.is_err());
}

// ============================================================================
// 13. RoutingStats
// ============================================================================

#[test]
fn routing_stats_new() {
    let stats = RoutingStats::new(5, vec![0, 1, 2, 3]);
    assert_eq!(stats.swap_count, 5);
    assert_eq!(stats.cnot_count, 15); // 5 * 3
}

#[test]
fn routing_stats_gate_overhead() {
    let stats = RoutingStats::new(5, vec![0, 1, 2]);
    let overhead = stats.gate_overhead(10);
    // 15 extra CNOTs added to 10 original gates
    assert!(overhead > 1.0);
}

// ============================================================================
// 14. GateDecomposer
// ============================================================================

#[test]
fn decomposer_ibm_native() {
    let decomposer = GateDecomposer::ibm_native();
    // H needs decomposition to IBM native
    assert!(decomposer.needs_decomposition("H"));
    // RZ is native
    assert!(!decomposer.needs_decomposition("RZ"));
}

#[test]
fn decomposer_rigetti_native() {
    let decomposer = GateDecomposer::rigetti_native();
    assert!(decomposer.needs_decomposition("H"));
    assert!(!decomposer.needs_decomposition("RZ"));
    assert!(!decomposer.needs_decomposition("RX"));
}

#[test]
fn decomposer_decompose_circuit_ibm() {
    let decomposer = GateDecomposer::ibm_native();
    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();
    let decomposed = decomposer.decompose_circuit(&c).unwrap();
    // H should be decomposed; CNOT is native
    assert!(decomposed.operations_slice().len() >= 2);
}

#[test]
fn decomposer_decompose_circuit_rigetti() {
    let decomposer = GateDecomposer::rigetti_native();
    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();
    let decomposed = decomposer.decompose_circuit(&c).unwrap();
    assert!(decomposed.operations_slice().len() >= 2);
}

#[test]
fn decomposer_custom_gateset() {
    let gs = GateSet::universal();
    let decomposer = GateDecomposer::new(gs);
    // Universal gate set: nothing needs decomposition
    assert!(!decomposer.needs_decomposition("H"));
}

// ============================================================================
// 15. Gate distribution analysis
// ============================================================================

#[test]
fn analyze_gate_distribution_basic() {
    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(Hadamard), &[q(1)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();
    let dist = analyze_gate_distribution(&c);
    assert_eq!(*dist.get("H").unwrap_or(&0), 2);
    assert_eq!(*dist.get("CNOT").unwrap_or(&0), 1);
}

#[test]
fn analyze_gate_distribution_empty() {
    let c = Circuit::new(1);
    let dist = analyze_gate_distribution(&c);
    assert!(dist.is_empty());
}

// ============================================================================
// 16. Inverse gate optimization
// ============================================================================

#[test]
fn optimize_inverse_gates_hh() {
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    let optimized = optimize_inverse_gates(&c).unwrap();
    assert!(optimized.operations_slice().len() < c.operations_slice().len());
}

#[test]
fn optimize_inverse_gates_xx() {
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    c.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    let optimized = optimize_inverse_gates(&c).unwrap();
    assert!(optimized.operations_slice().len() < c.operations_slice().len());
}

#[test]
fn optimize_inverse_gates_no_inverse() {
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    let optimized = optimize_inverse_gates(&c).unwrap();
    assert_eq!(optimized.operations_slice().len(), c.operations_slice().len());
}

// ============================================================================
// 17. Merge rotations optimization
// ============================================================================

#[test]
fn optimize_merge_rotations_basic() {
    // Adjacent same-axis rotations on the same qubit are a real mergeable
    // pair; since angle extraction isn't implemented, this must error
    // instead of silently returning the circuit unmerged.
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(RotationZ::new(0.5)), &[q(0)]).unwrap();
    c.add_gate(Arc::new(RotationZ::new(0.3)), &[q(0)]).unwrap();
    let result = optimize_merge_rotations(&c);
    assert!(result.is_err());
}

#[test]
fn optimize_merge_rotations_no_mergeable_pair_is_a_true_no_op() {
    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(RotationZ::new(0.5)), &[q(0)]).unwrap();
    c.add_gate(Arc::new(RotationZ::new(0.3)), &[q(1)]).unwrap();
    let result = optimize_merge_rotations(&c).unwrap();
    assert_eq!(result.len(), c.len());
}

// ============================================================================
// 18. DecompositionRules
// ============================================================================

#[test]
fn decomposition_rules_default() {
    let rules = DecompositionRules::default();
    assert!(rules.has_rule("H"));
    assert!(rules.has_rule("SWAP"));
}

#[test]
fn decomposition_rules_custom() {
    let mut rules = DecompositionRules::new();
    assert!(!rules.has_rule("MyGate"));
    rules.add_rule(
        "MyGate".to_string(),
        DecompositionRule {
            description: "Custom decomposition".to_string(),
            target_gates: vec!["H".to_string(), "CNOT".to_string()],
            gate_count: 3,
        },
    );
    assert!(rules.has_rule("MyGate"));
    let rule = rules.get_rule("MyGate").unwrap();
    assert_eq!(rule.gate_count, 3);
}

// ============================================================================
// 19. Transpiler
// ============================================================================

#[test]
fn transpiler_default() {
    let t = Transpiler::default();
    let cap = BackendCapabilities::simulator();
    let c = bell_circuit();
    let result = t.transpile(&c, &cap);
    assert!(result.is_ok());
}

#[test]
fn transpiler_none_level() {
    let t = Transpiler::new(OptimizationLevel::None);
    let cap = BackendCapabilities::simulator();
    let c = bell_circuit();
    let result = t.transpile(&c, &cap).unwrap();
    assert!(result.num_qubits() >= c.num_qubits());
}

#[test]
fn transpiler_light_level() {
    let t = Transpiler::new(OptimizationLevel::Light);
    let cap = BackendCapabilities::simulator();
    let c = bell_circuit();
    let result = t.transpile(&c, &cap);
    assert!(result.is_ok());
}

#[test]
fn transpiler_heavy_level() {
    let t = Transpiler::new(OptimizationLevel::Heavy);
    let cap = BackendCapabilities::simulator();
    let c = bell_circuit();
    let result = t.transpile(&c, &cap);
    assert!(result.is_ok());
}

#[test]
fn transpiler_with_approximations() {
    let t = Transpiler::new(OptimizationLevel::Medium).with_approximations(true);
    let cap = BackendCapabilities::simulator();
    let c = bell_circuit();
    let result = t.transpile(&c, &cap);
    assert!(result.is_ok());
}

#[test]
fn transpiler_estimate_cost() {
    let t = Transpiler::default();
    let cap = BackendCapabilities::simulator();
    let c = bell_circuit();
    let cost = t.estimate_cost(&c, &cap);
    assert!(cost.original_gates > 0);
    assert!(cost.gate_overhead() >= 0.0);
    assert!(cost.depth_overhead() >= 0.0);
}

#[test]
fn transpiler_ibm_backend() {
    let t = Transpiler::new(OptimizationLevel::Medium);
    let cg = ConnectivityGraph::linear_chain(5);
    let cap = BackendCapabilities::ibm_quantum(5, cg);
    let c = bell_circuit();
    let result = t.transpile(&c, &cap);
    assert!(result.is_ok());
}

// ============================================================================
// 20. TranspilationCost
// ============================================================================

#[test]
fn transpilation_cost_overhead() {
    let t = Transpiler::default();
    let cap = BackendCapabilities::simulator();
    let c = ghz_circuit(4);
    let cost = t.estimate_cost(&c, &cap);
    let gate_overhead = cost.gate_overhead();
    let depth_overhead = cost.depth_overhead();
    assert!(gate_overhead >= 0.0);
    assert!(depth_overhead >= 0.0);
}

// ============================================================================
// 21. LocalSimulatorBackend
// ============================================================================

#[cfg(feature = "local-simulator")]
mod local_sim_tests {
    use super::*;

    #[test]
    fn local_sim_new() {
        let backend = LocalSimulatorBackend::new();
        assert_eq!(backend.backend_type(), BackendType::Simulator);
        assert!(backend.is_available());
    }

    #[test]
    fn local_sim_name() {
        let backend = LocalSimulatorBackend::new().with_name("my-sim".to_string());
        assert_eq!(backend.name(), "my-sim");
    }

    #[test]
    fn local_sim_execute_x() {
        let backend = LocalSimulatorBackend::new();
        let c = single_x_circuit();
        let result = backend.execute(&c, 100).unwrap();
        assert_eq!(result.shots, 100);
        // X|0⟩ = |1⟩, so all shots should measure "1"
        assert_eq!(result.get_count("1"), 100);
    }

    #[test]
    fn local_sim_execute_bell() {
        let backend = LocalSimulatorBackend::new();
        let c = bell_circuit();
        let result = backend.execute(&c, 1000).unwrap();
        assert_eq!(result.shots, 1000);
        // Bell state: only "00" and "11"
        let p00 = result.get_count("00");
        let p11 = result.get_count("11");
        assert_eq!(p00 + p11, 1000);
        // Both should be approximately 500
        assert!(p00 > 300 && p00 < 700);
    }

    #[test]
    fn local_sim_execute_ghz() {
        let backend = LocalSimulatorBackend::new();
        let c = ghz_circuit(3);
        let result = backend.execute(&c, 1000).unwrap();
        let p000 = result.get_count("000");
        let p111 = result.get_count("111");
        assert_eq!(p000 + p111, 1000);
    }

    #[test]
    fn local_sim_deterministic_seed() {
        let config = LocalSimulatorConfig {
            seed: Some(42),
            ..Default::default()
        };
        let backend = LocalSimulatorBackend::with_config(config.clone());
        let c = bell_circuit();
        let r1 = backend.execute(&c, 100).unwrap();

        let backend2 = LocalSimulatorBackend::with_config(config);
        let r2 = backend2.execute(&c, 100).unwrap();

        // Same seed → same results
        assert_eq!(r1.get_count("00"), r2.get_count("00"));
        assert_eq!(r1.get_count("11"), r2.get_count("11"));
    }

    #[test]
    fn local_sim_capabilities() {
        let backend = LocalSimulatorBackend::new();
        let cap = backend.capabilities();
        assert!(cap.max_qubits > 0);
        assert!(cap.supports_parametric);
    }

    #[test]
    fn local_sim_estimate_cost() {
        let backend = LocalSimulatorBackend::new();
        let c = bell_circuit();
        let cost = backend.estimate_cost(&c, 1000);
        // Local simulator is free
        assert_eq!(cost, Some(0.0));
    }

    #[test]
    fn local_sim_validate_circuit() {
        let backend = LocalSimulatorBackend::new();
        let c = bell_circuit();
        assert!(backend.validate_circuit(&c).is_ok());
    }

    #[test]
    fn local_sim_validate_too_many_qubits() {
        let config = LocalSimulatorConfig {
            max_qubits: 2,
            ..Default::default()
        };
        let backend = LocalSimulatorBackend::with_config(config);
        let c = ghz_circuit(5);
        // Should fail validation since 5 > 2
        assert!(backend.validate_circuit(&c).is_err());
    }

    #[test]
    fn local_sim_submit_job() {
        let backend = LocalSimulatorBackend::new();
        let c = single_x_circuit();
        let job_id = backend.submit_job(&c, 10).unwrap();
        assert!(!job_id.is_empty());
    }

    #[test]
    fn local_sim_job_status() {
        let backend = LocalSimulatorBackend::new();
        let status = backend.job_status("anything").unwrap();
        assert_eq!(status, JobStatus::Completed);
    }

    #[test]
    fn local_sim_description() {
        let backend = LocalSimulatorBackend::new();
        let desc = backend.description();
        assert!(!desc.is_empty());
    }

    #[test]
    fn local_sim_custom_config() {
        let config = LocalSimulatorConfig {
            seed: Some(123),
            max_qubits: 20,
            sparse_threshold: 0.05,
            parallel: false,
            num_threads: Some(1),
        };
        let backend = LocalSimulatorBackend::with_config(config);
        assert!(backend.is_available());
    }

    #[test]
    fn local_sim_metadata_populated() {
        let backend = LocalSimulatorBackend::new();
        let c = bell_circuit();
        let result = backend.execute(&c, 100).unwrap();
        assert!(result.metadata.is_success());
        assert!(result.metadata.execution_time.is_some());
    }

    #[test]
    fn local_sim_all_pauli_gates() {
        let backend = LocalSimulatorBackend::new();

        // X gate
        let mut c = Circuit::new(1);
        c.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
        let r = backend.execute(&c, 100).unwrap();
        assert_eq!(r.get_count("1"), 100);

        // Y gate
        let mut c = Circuit::new(1);
        c.add_gate(Arc::new(PauliY), &[q(0)]).unwrap();
        let r = backend.execute(&c, 100).unwrap();
        assert_eq!(r.get_count("1"), 100);

        // Z gate (|0⟩ → |0⟩)
        let mut c = Circuit::new(1);
        c.add_gate(Arc::new(PauliZ), &[q(0)]).unwrap();
        let r = backend.execute(&c, 100).unwrap();
        assert_eq!(r.get_count("0"), 100);
    }

    #[test]
    fn local_sim_rotation_gates() {
        let backend = LocalSimulatorBackend::new();
        // RY(π) should flip |0⟩ to |1⟩
        let mut c = Circuit::new(1);
        c.add_gate(Arc::new(RotationY::new(std::f64::consts::PI)), &[q(0)])
            .unwrap();
        let r = backend.execute(&c, 100).unwrap();
        assert_eq!(r.get_count("1"), 100);
    }
}

// ============================================================================
// 22. BackendSelector
// ============================================================================

#[cfg(feature = "local-simulator")]
mod selector_tests {
    use super::*;

    #[test]
    fn selector_register_and_list() {
        let mut selector = BackendSelector::new();
        let backend = Arc::new(LocalSimulatorBackend::new());
        selector.register(backend);
        let available = selector.available_backends();
        assert!(!available.is_empty());
    }

    #[test]
    fn selector_select_by_name() {
        let mut selector = BackendSelector::new();
        let backend = Arc::new(LocalSimulatorBackend::new().with_name("test-sim".to_string()));
        selector.register(backend);
        let found = selector.select_by_name("test-sim");
        assert!(found.is_ok());
        assert_eq!(found.unwrap().name(), "test-sim");
    }

    #[test]
    fn selector_select_by_name_not_found() {
        let selector = BackendSelector::new();
        let result = selector.select_by_name("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn selector_select_with_criteria() {
        let mut selector = BackendSelector::new();
        let backend = Arc::new(LocalSimulatorBackend::new());
        selector.register(backend);
        let c = bell_circuit();
        let criteria = SelectionCriteria::default();
        let result = selector.select(&c, &criteria);
        assert!(result.is_ok());
    }

    #[test]
    fn selector_criteria_from_circuit() {
        let c = ghz_circuit(3);
        let criteria = SelectionCriteria::from_circuit(&c);
        assert!(criteria.max_qubits.is_some());
    }

    #[test]
    fn selector_criteria_builder() {
        let criteria = SelectionCriteria::default()
            .prefer_simulation()
            .max_cost(0.01)
            .require_gates(vec!["H".to_string(), "CNOT".to_string()]);
        assert!(criteria.prefer_simulation);
        assert_eq!(criteria.max_cost_per_shot, Some(0.01));
        assert_eq!(criteria.required_gates.len(), 2);
    }

    #[test]
    fn selector_prefer_features() {
        let criteria = SelectionCriteria::default()
            .prefer_features(vec![BackendFeature::Free, BackendFeature::FastExecution]);
        assert_eq!(criteria.prefer_features.len(), 2);
    }
}

// ============================================================================
// 23. SwapStrategy
// ============================================================================

#[test]
fn swap_strategy_default() {
    let s = SwapStrategy::default();
    assert_eq!(s, SwapStrategy::Greedy);
}

#[test]
fn swap_strategy_variants() {
    let _ = SwapStrategy::Greedy;
    let _ = SwapStrategy::Sabre;
    let _ = SwapStrategy::Lookahead;
}

// ============================================================================
// 24. OptimizationLevel
// ============================================================================

#[test]
fn optimization_level_default() {
    let level = OptimizationLevel::default();
    assert_eq!(level, OptimizationLevel::Medium);
}

#[test]
fn optimization_level_variants() {
    let _ = OptimizationLevel::None;
    let _ = OptimizationLevel::Light;
    let _ = OptimizationLevel::Medium;
    let _ = OptimizationLevel::Heavy;
}

// ============================================================================
// 25. Integration tests
// ============================================================================

#[cfg(feature = "local-simulator")]
mod integration_tests {
    use super::*;

    #[test]
    fn full_pipeline_bell_state() {
        // Create backend
        let backend = LocalSimulatorBackend::new();

        // Create and transpile circuit
        let transpiler = Transpiler::new(OptimizationLevel::Light);
        let c = bell_circuit();
        let transpiled = transpiler.transpile(&c, backend.capabilities()).unwrap();

        // Execute
        let result = backend.execute(&transpiled, 1000).unwrap();

        // Verify Bell state
        let p00 = result.get_count("00");
        let p11 = result.get_count("11");
        assert_eq!(p00 + p11, 1000);
        assert!(p00 > 300);
        assert!(p11 > 300);
    }

    #[test]
    fn full_pipeline_ghz() {
        let backend = LocalSimulatorBackend::new();
        let transpiler = Transpiler::default();
        let c = ghz_circuit(4);
        let transpiled = transpiler.transpile(&c, backend.capabilities()).unwrap();
        let result = backend.execute(&transpiled, 1000).unwrap();

        let p0000 = result.get_count("0000");
        let p1111 = result.get_count("1111");
        assert_eq!(p0000 + p1111, 1000);
    }

    #[test]
    fn decompose_then_execute() {
        let decomposer = GateDecomposer::ibm_native();
        let c = bell_circuit();
        let decomposed = decomposer.decompose_circuit(&c).unwrap();

        let backend = LocalSimulatorBackend::new();
        let result = backend.execute(&decomposed, 1000).unwrap();

        // Should still produce Bell state
        let p00 = result.get_count("00");
        let p11 = result.get_count("11");
        assert_eq!(p00 + p11, 1000);
    }

    #[test]
    fn selector_execute_workflow() {
        let mut selector = BackendSelector::new();
        selector.register(Arc::new(LocalSimulatorBackend::new()));

        let c = bell_circuit();
        let criteria = SelectionCriteria::from_circuit(&c).prefer_simulation();
        let backend = selector.select(&c, &criteria).unwrap();

        let result = backend.execute(&c, 100).unwrap();
        assert_eq!(result.shots, 100);
    }

    #[test]
    fn routing_on_linear_chain() {
        let router = Router::new(RoutingStrategy::Subgraph);
        let cg = ConnectivityGraph::linear_chain(5);
        let mapping = router.initial_mapping(3, &cg).unwrap();

        // Verify mapping is valid
        for i in 0..3 {
            let phys = mapping.get_physical(i).unwrap();
            assert!(phys < 5);
        }
    }

    #[test]
    fn result_analysis_workflow() {
        let backend = LocalSimulatorBackend::new();
        let c = bell_circuit();
        let result = backend.execute(&c, 10000).unwrap();

        // Analyze results
        let probs = result.probabilities();
        let total_prob: f64 = probs.values().sum();
        assert!((total_prob - 1.0).abs() < 1e-10);

        // Expectation value of ZZ
        let zz_exp = result.expectation_value(|bs| {
            let chars: Vec<char> = bs.chars().collect();
            let z0 = if chars[0] == '0' { 1.0 } else { -1.0 };
            let z1 = if chars[1] == '0' { 1.0 } else { -1.0 };
            z0 * z1
        });
        // Bell state: ⟨ZZ⟩ = 1
        assert!((zz_exp - 1.0).abs() < 0.1);
    }

    #[test]
    fn multiple_backends_merge_results() {
        let b1 = LocalSimulatorBackend::with_config(LocalSimulatorConfig {
            seed: Some(1),
            ..Default::default()
        });
        let b2 = LocalSimulatorBackend::with_config(LocalSimulatorConfig {
            seed: Some(2),
            ..Default::default()
        });

        let c = bell_circuit();
        let mut r1 = b1.execute(&c, 500).unwrap();
        let r2 = b2.execute(&c, 500).unwrap();
        r1.merge(&r2);

        assert_eq!(r1.shots, 1000);
        let total = r1.get_count("00") + r1.get_count("11");
        assert_eq!(total, 1000);
    }

    #[test]
    fn connectivity_graph_routing_integration() {
        let cg = ConnectivityGraph::grid(3, 3);

        // Check shortest paths exist between all pairs
        for i in 0..9 {
            for j in 0..9 {
                let path = cg.shortest_path(i, j);
                assert!(path.is_some(), "No path from {} to {}", i, j);
            }
        }

        // Route on this connectivity
        let router = Router::new(RoutingStrategy::HighestDegree);
        let mapping = router.initial_mapping(5, &cg).unwrap();
        for i in 0..5 {
            assert!(mapping.get_physical(i).unwrap() < 9);
        }
    }
}

// ============================================================================
// 26. Stress tests
// ============================================================================

#[test]
fn stress_large_connectivity_graph() {
    let cg = ConnectivityGraph::all_to_all(50);
    assert_eq!(cg.num_qubits(), 50);
    assert!(cg.are_connected(0, 49));
    let path = cg.shortest_path(0, 49).unwrap();
    assert_eq!(path.len(), 2); // Direct connection
}

#[test]
fn stress_grid_routing() {
    let cg = ConnectivityGraph::grid(5, 5);
    let router = Router::new(RoutingStrategy::Subgraph);
    let mapping = router.initial_mapping(10, &cg).unwrap();
    for i in 0..10 {
        assert!(mapping.get_physical(i).unwrap() < 25);
    }
}

#[test]
fn stress_many_swap_operations() {
    let mut m = QubitMapping::identity(100);
    for i in 0..99 {
        m.swap(i, i + 1);
    }
    // After shifting everything by one, qubit 0 should be at physical position 99
    // (each swap shifts by one)
    assert_eq!(m.get_logical(99), Some(0));
}

#[cfg(feature = "local-simulator")]
#[test]
fn stress_many_backend_executions() {
    let backend = LocalSimulatorBackend::new();
    let c = single_x_circuit();
    for _ in 0..100 {
        let result = backend.execute(&c, 10).unwrap();
        assert_eq!(result.get_count("1"), 10);
    }
}

#[test]
fn stress_decompose_large_circuit() {
    let decomposer = GateDecomposer::ibm_native();
    let mut c = Circuit::new(5);
    for i in 0..5 {
        c.add_gate(Arc::new(Hadamard), &[q(i)]).unwrap();
    }
    for i in 0..4 {
        c.add_gate(Arc::new(CNot), &[q(i), q(i + 1)]).unwrap();
    }
    let decomposed = decomposer.decompose_circuit(&c).unwrap();
    // Should have more gates after decomposition (H decomposed, CNOT native)
    assert!(decomposed.operations_slice().len() >= c.operations_slice().len());
}

#[test]
fn stress_gate_distribution_large() {
    let mut c = Circuit::new(10);
    for _ in 0..50 {
        for i in 0..10 {
            c.add_gate(Arc::new(Hadamard), &[q(i)]).unwrap();
        }
    }
    let dist = analyze_gate_distribution(&c);
    assert_eq!(*dist.get("H").unwrap(), 500);
}
