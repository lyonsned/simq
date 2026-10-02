//! Edge case and error path tests for simq-backend

use simq_backend::*;
use simq_core::{Circuit, QubitId};
use simq_gates::{CNot, Hadamard, PauliX, PauliZ, TGate};
use std::collections::HashMap;
use std::sync::Arc;

// ===========================================================================
// BackendResult edge cases
// ===========================================================================

mod result_edge_cases {
    use super::*;

    #[test]
    fn empty_counts() {
        let result = BackendResult::new(HashMap::new(), 100);
        assert!(result.most_frequent().is_none());
        assert!(result.probabilities().is_empty());
        assert!(result.bitstrings().is_empty());
        assert_eq!(result.get_count("00"), 0);
    }

    #[test]
    fn single_outcome() {
        let mut counts = HashMap::new();
        counts.insert("000".to_string(), 1024);
        let result = BackendResult::new(counts, 1024);

        let (bs, count) = result.most_frequent().unwrap();
        assert_eq!(bs, "000");
        assert_eq!(*count, 1024);

        let probs = result.probabilities();
        assert!((probs["000"] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn merge_non_overlapping() {
        let mut counts1 = HashMap::new();
        counts1.insert("00".to_string(), 50);
        let mut r1 = BackendResult::new(counts1, 50);

        let mut counts2 = HashMap::new();
        counts2.insert("11".to_string(), 50);
        let r2 = BackendResult::new(counts2, 50);

        r1.merge(&r2);
        assert_eq!(r1.shots, 100);
        assert_eq!(r1.get_count("00"), 50);
        assert_eq!(r1.get_count("11"), 50);
    }

    #[test]
    fn merge_overlapping() {
        let mut counts1 = HashMap::new();
        counts1.insert("01".to_string(), 30);
        counts1.insert("10".to_string(), 20);
        let mut r1 = BackendResult::new(counts1, 50);

        let mut counts2 = HashMap::new();
        counts2.insert("01".to_string(), 40);
        counts2.insert("11".to_string(), 60);
        let r2 = BackendResult::new(counts2, 100);

        r1.merge(&r2);
        assert_eq!(r1.shots, 150);
        assert_eq!(r1.get_count("01"), 70);
        assert_eq!(r1.get_count("10"), 20);
        assert_eq!(r1.get_count("11"), 60);
    }

    #[test]
    fn expectation_value_z_observable() {
        let mut counts = HashMap::new();
        counts.insert("0".to_string(), 700);
        counts.insert("1".to_string(), 300);
        let result = BackendResult::new(counts, 1000);

        let exp = result.expectation_value(|bs| match bs {
            "0" => 1.0,
            "1" => -1.0,
            _ => 0.0,
        });
        assert!((exp - 0.4).abs() < 1e-10);
    }

    #[test]
    fn expectation_value_uniform() {
        let mut counts = HashMap::new();
        counts.insert("0".to_string(), 500);
        counts.insert("1".to_string(), 500);
        let result = BackendResult::new(counts, 1000);

        let exp = result.expectation_value(|bs| match bs {
            "0" => 1.0,
            "1" => -1.0,
            _ => 0.0,
        });
        assert!(exp.abs() < 1e-10);
    }

    #[test]
    fn probabilities_sum_to_one() {
        let mut counts = HashMap::new();
        counts.insert("00".to_string(), 25);
        counts.insert("01".to_string(), 25);
        counts.insert("10".to_string(), 25);
        counts.insert("11".to_string(), 25);
        let result = BackendResult::new(counts, 100);

        let probs = result.probabilities();
        let sum: f64 = probs.values().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}

// ===========================================================================
// BackendError edge cases
// ===========================================================================

mod error_edge_cases {
    use super::*;

    #[test]
    fn all_error_variants_display() {
        let errors: Vec<BackendError> = vec![
            BackendError::CircuitIncompatible("test".into()),
            BackendError::CapabilityExceeded("test".into()),
            BackendError::CommunicationError("test".into()),
            BackendError::AuthenticationFailed("test".into()),
            BackendError::JobSubmissionFailed("test".into()),
            BackendError::JobExecutionFailed("test".into()),
            BackendError::JobNotFound {
                job_id: "abc".into(),
            },
            BackendError::JobTimeout {
                timeout_seconds: 60,
            },
            BackendError::TranspilationFailed("test".into()),
            BackendError::InvalidConfiguration("test".into()),
            BackendError::BackendUnavailable("test".into()),
            BackendError::RateLimitExceeded("test".into()),
            BackendError::InsufficientQuota("test".into()),
            BackendError::SerializationError("test".into()),
            BackendError::DeserializationError("test".into()),
            BackendError::NetworkError("test".into()),
            BackendError::Other("test".into()),
        ];

        for e in &errors {
            let msg = format!("{}", e);
            assert!(!msg.is_empty());
        }
    }

    #[test]
    fn from_serde_json_error() {
        let json_err: serde_json::Error =
            serde_json::from_str::<serde_json::Value>("not json{{{").unwrap_err();
        let backend_err: BackendError = json_err.into();
        match backend_err {
            BackendError::SerializationError(msg) => {
                assert!(!msg.is_empty());
            },
            _ => panic!("expected SerializationError"),
        }
    }

    #[test]
    fn from_quantum_error() {
        let core_err = simq_core::QuantumError::InvalidQubit(99, 2);
        let backend_err: BackendError = core_err.into();
        match backend_err {
            BackendError::Other(msg) => {
                assert!(msg.contains("Core error"));
            },
            _ => panic!("expected Other"),
        }
    }

    #[test]
    fn error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(BackendError::Other("test".into()));
        assert!(!err.to_string().is_empty());
    }
}

// ===========================================================================
// ExecutionMetadata and JobStatus edge cases
// ===========================================================================

mod metadata_edge_cases {
    use super::*;
    use std::time::Duration;

    #[test]
    fn job_status_display_all_variants() {
        let variants = [
            (JobStatus::Queued, "Queued"),
            (JobStatus::Validating, "Validating"),
            (JobStatus::Running, "Running"),
            (JobStatus::Completed, "Completed"),
            (JobStatus::Failed, "Failed"),
            (JobStatus::Cancelled, "Cancelled"),
        ];

        for (status, expected) in &variants {
            assert_eq!(format!("{}", status), *expected);
        }
    }

    #[test]
    fn default_job_status_is_queued() {
        let status = JobStatus::default();
        assert_eq!(status, JobStatus::Queued);
    }

    #[test]
    fn metadata_success() {
        let meta = ExecutionMetadata::success("sim".to_string(), Duration::from_millis(100));
        assert!(meta.is_success());
        assert!(!meta.is_failed());
        assert_eq!(meta.status, JobStatus::Completed);
        assert_eq!(meta.backend_name.as_deref(), Some("sim"));
    }

    #[test]
    fn metadata_failed() {
        let meta = ExecutionMetadata::failed("boom".to_string());
        assert!(meta.is_failed());
        assert!(!meta.is_success());
        assert_eq!(meta.error_message.as_deref(), Some("boom"));
    }

    #[test]
    fn metadata_cancelled_is_failed() {
        let meta = ExecutionMetadata {
            status: JobStatus::Cancelled,
            ..Default::default()
        };
        assert!(meta.is_failed());
    }
}

// ===========================================================================
// ConnectivityGraph edge cases
// ===========================================================================

mod connectivity_edge_cases {
    use super::*;

    #[test]
    fn single_qubit_graph() {
        let graph = ConnectivityGraph::new(1, false);
        assert_eq!(graph.num_qubits(), 1);
        assert_eq!(graph.degree(0), 0);
        assert!(graph.neighbors(0).is_none());
    }

    #[test]
    fn directed_graph() {
        let mut graph = ConnectivityGraph::new(3, true);
        assert!(graph.is_directed());
        graph.add_edge(0, 1);
        assert!(graph.are_connected(0, 1));
        assert!(!graph.are_connected(1, 0));
    }

    #[test]
    fn undirected_graph_bidirectional() {
        let mut graph = ConnectivityGraph::new(3, false);
        graph.add_edge(0, 1);
        assert!(graph.are_connected(0, 1));
        assert!(graph.are_connected(1, 0));
    }

    #[test]
    fn disconnected_qubits_no_path() {
        let mut graph = ConnectivityGraph::new(4, false);
        graph.add_edge(0, 1);
        graph.add_edge(2, 3);
        // 0-1 and 2-3 are disconnected components
        assert!(graph.shortest_path(0, 2).is_none());
    }

    #[test]
    fn shortest_path_to_self() {
        let graph = ConnectivityGraph::linear_chain(3);
        let path = graph.shortest_path(1, 1).unwrap();
        assert_eq!(path, vec![1]);
    }

    #[test]
    fn grid_topology_path() {
        let graph = ConnectivityGraph::grid(2, 3);
        // 0-1-2
        // | | |
        // 3-4-5
        assert_eq!(graph.num_qubits(), 6);
        assert!(graph.are_connected(0, 1));
        assert!(graph.are_connected(0, 3));
        assert!(!graph.are_connected(0, 4));

        let path = graph.shortest_path(0, 5).unwrap();
        assert!(path.len() <= 4);
    }

    #[test]
    fn all_to_all_degree() {
        let graph = ConnectivityGraph::all_to_all(4);
        for q in 0..4 {
            assert_eq!(graph.degree(q), 3);
        }
    }

    #[test]
    fn linear_chain_endpoints_degree() {
        let graph = ConnectivityGraph::linear_chain(5);
        assert_eq!(graph.degree(0), 1);
        assert_eq!(graph.degree(2), 2);
        assert_eq!(graph.degree(4), 1);
    }

    #[test]
    fn neighbors_of_isolated_qubit() {
        let graph = ConnectivityGraph::new(5, false);
        assert!(graph.neighbors(3).is_none());
        assert_eq!(graph.degree(3), 0);
    }
}

// ===========================================================================
// GateSet and BackendCapabilities edge cases
// ===========================================================================

mod capabilities_edge_cases {
    use super::*;

    #[test]
    fn empty_gate_set() {
        let gs = GateSet::new();
        assert!(gs.is_empty());
        assert_eq!(gs.len(), 0);
        assert!(!gs.contains(&"H".to_string()));
    }

    #[test]
    fn universal_gate_set_contains_standard() {
        let gs = GateSet::universal();
        assert!(gs.contains(&"H".to_string()));
        assert!(gs.contains(&"CNOT".to_string()));
        assert!(gs.contains(&"Toffoli".to_string()));
        assert!(!gs.contains(&"MyCustomGate".to_string()));
    }

    #[test]
    fn default_capabilities_all_to_all() {
        let caps = BackendCapabilities::default();
        assert!(caps.are_qubits_connected(0, 63));
        assert!(caps.connectivity.is_none());
    }

    #[test]
    fn ibm_capabilities() {
        let conn = ConnectivityGraph::linear_chain(5);
        let caps = BackendCapabilities::ibm_quantum(5, conn);
        assert_eq!(caps.max_qubits, 5);
        assert!(caps.is_native_gate("CNOT".to_string()));
        assert!(caps.is_native_gate("RZ".to_string()));
        assert!(!caps.is_native_gate("H".to_string()));
        assert!(caps.are_qubits_connected(0, 1));
        assert!(!caps.are_qubits_connected(0, 3));
    }

    #[test]
    fn simulator_capabilities() {
        let caps = BackendCapabilities::simulator();
        assert!(caps.supports_mid_circuit_measurement);
        assert!(caps.supports_conditional);
        assert!(caps.supports_reset);
        assert!(caps.supports_parametric);
    }
}

// ===========================================================================
// Router edge cases
// ===========================================================================

mod router_edge_cases {
    use super::*;

    #[test]
    fn too_many_qubits_for_topology() {
        let router = Router::new(RoutingStrategy::Identity);
        let connectivity = ConnectivityGraph::linear_chain(3);

        match router.initial_mapping(5, &connectivity) {
            Err(BackendError::CapabilityExceeded(msg)) => {
                assert!(msg.contains("5"));
                assert!(msg.contains("3"));
            },
            Err(other) => panic!("expected CapabilityExceeded, got {:?}", other),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn swap_chain_already_connected() {
        let router = Router::new(RoutingStrategy::Identity);
        let connectivity = ConnectivityGraph::linear_chain(5);
        let mapping = QubitMapping::identity(5);

        let swaps = router
            .find_swap_chain(&connectivity, &mapping, 0, 1)
            .unwrap();
        assert!(swaps.is_empty());
    }

    #[test]
    fn swap_chain_distant_qubits() {
        let router = Router::new(RoutingStrategy::Identity);
        let connectivity = ConnectivityGraph::linear_chain(5);
        let mapping = QubitMapping::identity(5);

        let swaps = router
            .find_swap_chain(&connectivity, &mapping, 0, 4)
            .unwrap();
        assert_eq!(swaps.len(), 3);
    }

    #[test]
    fn highest_degree_strategy() {
        let router = Router::new(RoutingStrategy::HighestDegree);
        let connectivity = ConnectivityGraph::all_to_all(5);
        let mapping = router.initial_mapping(3, &connectivity).unwrap();
        for i in 0..3 {
            assert!(mapping.get_physical(i).is_some());
        }
    }

    #[test]
    fn subgraph_strategy_grid() {
        let router = Router::new(RoutingStrategy::Subgraph);
        let connectivity = ConnectivityGraph::grid(3, 3);
        let mapping = router.initial_mapping(4, &connectivity).unwrap();
        for i in 0..4 {
            assert!(mapping.get_physical(i).is_some());
        }
    }

    #[test]
    fn swap_gate_ordered() {
        let swap = SwapGate::new(5, 2);
        assert_eq!(swap.ordered(), (2, 5));

        let swap2 = SwapGate::new(1, 3);
        assert_eq!(swap2.ordered(), (1, 3));
    }

    #[test]
    fn swap_gate_apply() {
        let mut mapping = QubitMapping::identity(3);
        let swap = SwapGate::new(0, 2);
        swap.apply(&mut mapping);
        assert_eq!(mapping.get_physical(0), Some(2));
        assert_eq!(mapping.get_physical(2), Some(0));
        assert_eq!(mapping.get_physical(1), Some(1));
    }

    #[test]
    fn routing_stats() {
        let stats = RoutingStats::new(10, vec![0, 1, 2]);
        assert_eq!(stats.swap_count, 10);
        assert_eq!(stats.cnot_count, 30);
        assert_eq!(stats.depth_increase, 10);
        assert!((stats.gate_overhead(100) - 0.3).abs() < 1e-10);
    }

    #[test]
    fn routing_stats_zero_original_gates() {
        let stats = RoutingStats::new(5, vec![]);
        assert!((stats.gate_overhead(0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn sabre_router_default() {
        let router = SabreRouter::default();
        let result = router.route(3, &ConnectivityGraph::linear_chain(3));
        // Must error rather than return misleading empty plan.
        assert!(result.is_err());
    }
}

// ===========================================================================
// Transpiler edge cases
// ===========================================================================

mod transpiler_edge_cases {
    use super::*;

    #[test]
    fn transpiler_default_is_medium() {
        let t = Transpiler::default();
        let caps = BackendCapabilities::default();
        let circuit = Circuit::new(2);
        // should not error on empty circuit
        let result = t.transpile(&circuit, &caps);
        assert!(result.is_ok());
    }

    #[test]
    fn transpile_too_many_qubits() {
        let t = Transpiler::new(OptimizationLevel::None);
        let circuit = Circuit::new(10);
        let caps = BackendCapabilities {
            max_qubits: 5,
            ..BackendCapabilities::default()
        };

        let err = t.transpile(&circuit, &caps).unwrap_err();
        match err {
            BackendError::CapabilityExceeded(msg) => {
                assert!(msg.contains("10"));
                assert!(msg.contains("5"));
            },
            _ => panic!("expected CapabilityExceeded"),
        }
    }

    #[test]
    fn all_optimization_levels() {
        let circuit = Circuit::new(2);
        let caps = BackendCapabilities::default();

        for level in [
            OptimizationLevel::None,
            OptimizationLevel::Light,
            OptimizationLevel::Medium,
            OptimizationLevel::Heavy,
        ] {
            let t = Transpiler::new(level);
            assert!(t.transpile(&circuit, &caps).is_ok());
        }
    }

    #[test]
    fn transpilation_cost_zeros() {
        let cost = TranspilationCost {
            original_gates: 0,
            transpiled_gates: 0,
            original_depth: 0,
            transpiled_depth: 0,
            swap_gates: 0,
        };
        assert!((cost.gate_overhead() - 0.0).abs() < 1e-10);
        assert!((cost.depth_overhead() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn transpilation_cost_calculation() {
        let cost = TranspilationCost {
            original_gates: 100,
            transpiled_gates: 200,
            original_depth: 10,
            transpiled_depth: 20,
            swap_gates: 5,
        };
        assert!((cost.gate_overhead() - 1.0).abs() < 1e-10);
        assert!((cost.depth_overhead() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn estimate_cost_empty_circuit() {
        let t = Transpiler::default();
        let circuit = Circuit::new(2);
        let caps = BackendCapabilities::default();

        let cost = t.estimate_cost(&circuit, &caps);
        assert_eq!(cost.original_gates, 0);
        assert_eq!(cost.original_depth, 0);
    }

    #[test]
    fn with_approximations() {
        let t = Transpiler::new(OptimizationLevel::Medium).with_approximations(true);
        let circuit = Circuit::new(2);
        let caps = BackendCapabilities::default();
        assert!(t.transpile(&circuit, &caps).is_ok());
    }
}

// ===========================================================================
// DecompositionRules edge cases
// ===========================================================================

mod decomposition_edge_cases {
    use super::*;

    #[test]
    fn default_rules_exist() {
        let rules = DecompositionRules::default();
        assert!(rules.has_rule("H"));
        assert!(rules.has_rule("T"));
        assert!(rules.has_rule("S"));
        assert!(rules.has_rule("CZ"));
        assert!(rules.has_rule("SWAP"));
        assert!(rules.has_rule("Toffoli"));
        assert!(!rules.has_rule("MyCustomGate"));
    }

    #[test]
    fn custom_rule_addition() {
        let mut t = Transpiler::default();
        t.add_decomposition_rule(
            "Foo",
            DecompositionRule {
                description: "Foo → X Y".into(),
                target_gates: vec!["X".into(), "Y".into()],
                gate_count: 2,
            },
        );
        // no panic = success
    }

    #[test]
    fn empty_decomposition_rules() {
        let rules = DecompositionRules::new();
        assert!(!rules.has_rule("H"));
        assert!(rules.get_rule("H").is_none());
    }

    #[test]
    fn rule_details() {
        let rules = DecompositionRules::default();
        let swap = rules.get_rule("SWAP").unwrap();
        assert_eq!(swap.gate_count, 3);
        assert!(swap.target_gates.contains(&"CNOT".to_string()));
    }
}

// ===========================================================================
// QubitMapping edge cases
// ===========================================================================

mod mapping_edge_cases {
    use super::*;

    #[test]
    fn identity_mapping() {
        let mapping = QubitMapping::identity(5);
        for i in 0..5 {
            assert_eq!(mapping.get_physical(i), Some(i));
            assert_eq!(mapping.get_logical(i), Some(i));
        }
    }

    #[test]
    fn custom_mapping() {
        let mapping = QubitMapping::from_vec(vec![3, 1, 0], 5);
        assert_eq!(mapping.get_physical(0), Some(3));
        assert_eq!(mapping.get_physical(1), Some(1));
        assert_eq!(mapping.get_physical(2), Some(0));

        assert_eq!(mapping.get_logical(3), Some(0));
        assert_eq!(mapping.get_logical(1), Some(1));
        assert_eq!(mapping.get_logical(0), Some(2));
        assert_eq!(mapping.get_logical(4), None);
    }

    #[test]
    fn mapping_swap() {
        let mut mapping = QubitMapping::identity(4);
        mapping.swap(0, 3);
        assert_eq!(mapping.get_physical(0), Some(3));
        assert_eq!(mapping.get_physical(3), Some(0));
        assert_eq!(mapping.get_logical(0), Some(3));
        assert_eq!(mapping.get_logical(3), Some(0));

        // Middle qubits unchanged
        assert_eq!(mapping.get_physical(1), Some(1));
        assert_eq!(mapping.get_physical(2), Some(2));
    }

    #[test]
    fn mapping_double_swap_restores() {
        let mut mapping = QubitMapping::identity(3);
        mapping.swap(0, 2);
        mapping.swap(0, 2);
        for i in 0..3 {
            assert_eq!(mapping.get_physical(i), Some(i));
        }
    }

    #[test]
    fn out_of_range_logical() {
        let mapping = QubitMapping::identity(3);
        assert_eq!(mapping.get_physical(10), None);
    }
}

// ===========================================================================
// GateDecomposer edge cases
// ===========================================================================

mod gate_decomposer_edge_cases {
    use super::*;

    #[test]
    fn ibm_native_gate_no_decomposition() {
        let decomposer = GateDecomposer::ibm_native();
        assert!(!decomposer.needs_decomposition("RZ"));
        assert!(!decomposer.needs_decomposition("SX"));
        assert!(!decomposer.needs_decomposition("X"));
        assert!(!decomposer.needs_decomposition("CNOT"));
    }

    #[test]
    fn rigetti_native_gate_no_decomposition() {
        let decomposer = GateDecomposer::rigetti_native();
        assert!(!decomposer.needs_decomposition("RZ"));
        assert!(!decomposer.needs_decomposition("RX"));
        assert!(!decomposer.needs_decomposition("CZ"));
    }

    #[test]
    fn decompose_native_gate_returns_self() {
        let decomposer = GateDecomposer::ibm_native();
        let q0 = QubitId::new(0);
        let op = simq_core::GateOp::new(Arc::new(PauliX), &[q0]).unwrap();
        let result = decomposer.decompose_gate(&op).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].gate().name(), "X");
    }

    #[test]
    fn decompose_h_to_ibm() {
        let decomposer = GateDecomposer::ibm_native();
        let q0 = QubitId::new(0);
        let op = simq_core::GateOp::new(Arc::new(Hadamard), &[q0]).unwrap();
        let result = decomposer.decompose_gate(&op).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].gate().name(), "RZ");
        assert_eq!(result[1].gate().name(), "SX");
        assert_eq!(result[2].gate().name(), "RZ");
    }

    #[test]
    fn decompose_unknown_gate_errors() {
        let decomposer = GateDecomposer::ibm_native();
        let q0 = QubitId::new(0);
        // Fredkin needs decomposition but has no rule in IBM set
        let op = simq_core::GateOp::new(
            Arc::new(simq_gates::Fredkin),
            &[q0, QubitId::new(1), QubitId::new(2)],
        )
        .unwrap();
        let err = decomposer.decompose_gate(&op).unwrap_err();
        match err {
            BackendError::TranspilationFailed(msg) => {
                assert!(msg.contains("CSWAP"));
            },
            _ => panic!("expected TranspilationFailed"),
        }
    }

    #[test]
    fn decompose_circuit_all_gates() {
        let decomposer = GateDecomposer::ibm_native();
        let mut circuit = Circuit::new(2);
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);

        circuit.add_gate(Arc::new(Hadamard), &[q0]).unwrap();
        circuit.add_gate(Arc::new(TGate), &[q1]).unwrap();
        circuit.add_gate(Arc::new(CNot), &[q0, q1]).unwrap();

        let decomposed = decomposer.decompose_circuit(&circuit).unwrap();
        // H → 3, T → 1, CNOT → 1 (native) = 5
        assert_eq!(decomposed.len(), 5);

        for op in decomposed.operations() {
            assert!(
                !decomposer.needs_decomposition(op.gate().name()),
                "{} should not need decomposition",
                op.gate().name()
            );
        }
    }

    #[test]
    fn optimize_inverse_gates_all_cancel() {
        let mut circuit = Circuit::new(1);
        let q0 = QubitId::new(0);

        circuit.add_gate(Arc::new(Hadamard), &[q0]).unwrap();
        circuit.add_gate(Arc::new(Hadamard), &[q0]).unwrap();

        let optimized = optimize_inverse_gates(&circuit).unwrap();
        assert_eq!(optimized.len(), 0);
    }

    #[test]
    fn optimize_inverse_gates_partial() {
        let mut circuit = Circuit::new(1);
        let q0 = QubitId::new(0);

        circuit.add_gate(Arc::new(PauliX), &[q0]).unwrap();
        circuit.add_gate(Arc::new(PauliX), &[q0]).unwrap();
        circuit.add_gate(Arc::new(PauliZ), &[q0]).unwrap();

        let optimized = optimize_inverse_gates(&circuit).unwrap();
        assert_eq!(optimized.len(), 1);
        assert_eq!(optimized.operations().next().unwrap().gate().name(), "Z");
    }

    #[test]
    fn optimize_inverse_different_qubits_no_cancel() {
        let mut circuit = Circuit::new(2);
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);

        circuit.add_gate(Arc::new(Hadamard), &[q0]).unwrap();
        circuit.add_gate(Arc::new(Hadamard), &[q1]).unwrap();

        let optimized = optimize_inverse_gates(&circuit).unwrap();
        assert_eq!(optimized.len(), 2);
    }

    #[test]
    fn analyze_gate_distribution_counts() {
        let mut circuit = Circuit::new(2);
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);

        circuit.add_gate(Arc::new(Hadamard), &[q0]).unwrap();
        circuit.add_gate(Arc::new(Hadamard), &[q1]).unwrap();
        circuit.add_gate(Arc::new(CNot), &[q0, q1]).unwrap();

        let counts = analyze_gate_distribution(&circuit);
        assert_eq!(counts.get("H"), Some(&2));
        assert_eq!(counts.get("CNOT"), Some(&1));
    }

    #[test]
    fn merge_rotations_is_no_op() {
        let circuit = Circuit::new(1);
        let result = optimize_merge_rotations(&circuit).unwrap();
        assert_eq!(result.len(), 0);
    }
}

// ===========================================================================
// SwapStrategy edge cases
// ===========================================================================

mod swap_strategy_edge_cases {
    use super::*;

    #[test]
    fn default_swap_strategy_is_greedy() {
        let strat = SwapStrategy::default();
        assert_eq!(strat, SwapStrategy::Greedy);
    }

    #[test]
    fn all_swap_strategies() {
        let _g = SwapStrategy::Greedy;
        let _s = SwapStrategy::Sabre;
        let _l = SwapStrategy::Lookahead;
    }

    #[test]
    fn default_routing_strategy_is_subgraph() {
        let strat = RoutingStrategy::default();
        assert_eq!(strat, RoutingStrategy::Subgraph);
    }
}
