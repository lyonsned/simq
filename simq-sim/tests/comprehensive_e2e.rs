//! Comprehensive end-to-end tests for simq-sim crate
//!
//! Covers: Simulator, SimulatorConfig, SimulationResult, MeasurementCounts,
//! ExecutionStatistics, ExecutionEngine, gradient computation, VQE/QAOA helpers,
//! autodiff, caching, checkpointing, and stress tests.

use num_complex::Complex64;
use simq_core::circuit::Circuit;
use simq_core::QubitId;
use simq_gates::standard::*;
use simq_sim::*;
use std::sync::Arc;

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
// 1. SimulatorConfig
// ============================================================================

#[test]
fn config_default() {
    let config = SimulatorConfig::default();
    assert_eq!(config.shots, 1024);
    assert!(config.optimize_circuit);
    assert_eq!(config.optimization_level, 2);
    assert_eq!(config.sparse_threshold, 0.1);
    assert!(!config.collect_statistics);
    assert!(config.seed.is_none());
}

#[test]
fn config_fast() {
    let config = SimulatorConfig::fast();
    assert_eq!(config.optimization_level, 3);
    assert!(!config.collect_statistics);
}

#[test]
fn config_accurate() {
    let config = SimulatorConfig::accurate();
    assert!(!config.optimize_circuit);
    assert_eq!(config.shots, 10000);
    assert!(config.collect_statistics);
}

#[test]
fn config_debug() {
    let config = SimulatorConfig::debug();
    assert!(!config.optimize_circuit);
    assert!(config.collect_statistics);
    assert_eq!(config.seed, Some(42));
}

#[test]
fn config_builder() {
    let config = SimulatorConfig::new()
        .with_shots(2048)
        .with_optimization_level(3)
        .with_seed(123)
        .with_sparse_threshold(0.2)
        .with_statistics(true)
        .with_memory_limit(1024 * 1024);
    assert_eq!(config.shots, 2048);
    assert_eq!(config.optimization_level, 3);
    assert_eq!(config.seed, Some(123));
    assert_eq!(config.sparse_threshold, 0.2);
    assert!(config.collect_statistics);
    assert_eq!(config.memory_limit, 1024 * 1024);
}

#[test]
fn config_validate_ok() {
    let config = SimulatorConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn config_validate_bad_threshold() {
    let config = SimulatorConfig {
        sparse_threshold: 1.5,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn config_validate_zero_shots() {
    let config = SimulatorConfig {
        shots: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

// ============================================================================
// 2. Simulator creation and basic runs
// ============================================================================

#[test]
fn simulator_default_creation() {
    let sim = Simulator::default();
    assert_eq!(sim.config().shots, 1024);
}

#[test]
fn simulator_run_bell_state() {
    let sim = Simulator::new(SimulatorConfig::default());
    let circuit = bell_circuit();
    let result = sim.run(&circuit);
    assert!(result.is_ok(), "Bell state simulation should succeed: {:?}", result.err());
    let result = result.unwrap();
    assert_eq!(result.num_qubits(), 2);
}

#[test]
fn simulator_run_x_gate() {
    let sim = Simulator::new(SimulatorConfig::default());
    let circuit = single_x_circuit();
    let result = sim.run(&circuit).unwrap();
    assert_eq!(result.num_qubits(), 1);
}

#[test]
fn simulator_empty_circuit_errors() {
    let sim = Simulator::default();
    let circuit = Circuit::new(2);
    let result = sim.run(&circuit);
    assert!(result.is_err(), "Empty circuit should error");
}

#[test]
fn simulator_with_optimization() {
    let sim = Simulator::new(SimulatorConfig::default().with_optimization(true));
    let mut circuit = Circuit::new(1);
    circuit.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    circuit.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    circuit.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    let result = sim.run(&circuit);
    assert!(result.is_ok());
}

#[test]
fn simulator_without_optimization() {
    let sim = Simulator::new(SimulatorConfig::default().with_optimization(false));
    let circuit = bell_circuit();
    let result = sim.run(&circuit);
    assert!(result.is_ok());
}

#[test]
fn simulator_optimization_levels() {
    for level in 0..=3u8 {
        let sim = Simulator::new(SimulatorConfig::default().with_optimization_level(level));
        let circuit = bell_circuit();
        let result = sim.run(&circuit);
        assert!(result.is_ok(), "Level {} should succeed", level);
    }
}

// ============================================================================
// 3. ExecutionStatistics
// ============================================================================

#[test]
fn statistics_enabled() {
    let sim = Simulator::new(SimulatorConfig::default().with_statistics(true));
    let circuit = bell_circuit();
    let result = sim.run(&circuit).unwrap();
    assert!(result.statistics.is_some());
    let stats = result.statistics.unwrap();
    assert!(stats.gates_executed > 0);
    assert!(stats.peak_memory_bytes > 0);
}

#[test]
fn statistics_disabled() {
    let sim = Simulator::new(SimulatorConfig::default().with_statistics(false));
    let circuit = bell_circuit();
    let result = sim.run(&circuit).unwrap();
    assert!(result.statistics.is_none());
}

#[test]
fn statistics_optimization_ratio() {
    let sim = Simulator::new(SimulatorConfig::default().with_statistics(true));
    let mut circuit = Circuit::new(1);
    circuit.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    circuit.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    circuit.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    let result = sim.run(&circuit).unwrap();
    let stats = result.statistics.unwrap();
    assert!(stats.gates_executed >= 1);
}

#[test]
fn statistics_display() {
    let sim = Simulator::new(SimulatorConfig::default().with_statistics(true));
    let circuit = bell_circuit();
    let result = sim.run(&circuit).unwrap();
    let stats = result.statistics.unwrap();
    let display = format!("{}", stats);
    assert!(!display.is_empty());
    assert!(display.contains("Execution Statistics"));
}

// ============================================================================
// 4. SimulationResult
// ============================================================================

#[test]
fn result_num_qubits() {
    let sim = Simulator::default();
    let circuit = ghz_circuit(3);
    let result = sim.run(&circuit).unwrap();
    assert_eq!(result.num_qubits(), 3);
}

#[test]
fn result_state_representation() {
    let sim = Simulator::default();
    let circuit = bell_circuit();
    let result = sim.run(&circuit).unwrap();
    assert!(result.is_sparse() || result.is_dense());
}

// ============================================================================
// 5. MeasurementCounts
// ============================================================================

#[test]
fn measurement_counts_basic() {
    let mut counts = MeasurementCounts::new(100);
    counts.add("00".to_string(), 50);
    counts.add("11".to_string(), 50);
    assert_eq!(counts.total_shots(), 100);
    assert_eq!(counts.num_outcomes(), 2);
    assert_eq!(counts.get("00"), 50);
    assert_eq!(counts.get("10"), 0);
}

#[test]
fn measurement_counts_probability() {
    let mut counts = MeasurementCounts::new(1000);
    counts.add("00".to_string(), 500);
    counts.add("11".to_string(), 500);
    assert!((counts.probability("00") - 0.5).abs() < 1e-10);
    assert!((counts.probability("11") - 0.5).abs() < 1e-10);
    assert_eq!(counts.probability("01"), 0.0);
}

#[test]
fn measurement_counts_most_common() {
    let mut counts = MeasurementCounts::new(100);
    counts.add("00".to_string(), 60);
    counts.add("01".to_string(), 30);
    counts.add("11".to_string(), 10);
    let (bs, count) = counts.most_common().unwrap();
    assert_eq!(bs, "00");
    assert_eq!(count, 60);
}

#[test]
fn measurement_counts_sorted() {
    let mut counts = MeasurementCounts::new(100);
    counts.add("00".to_string(), 10);
    counts.add("01".to_string(), 60);
    counts.add("11".to_string(), 30);
    let sorted = counts.sorted();
    assert_eq!(sorted[0].0, "01");
    assert_eq!(sorted[1].0, "11");
    assert_eq!(sorted[2].0, "00");
}

#[test]
fn measurement_counts_from_map() {
    use std::collections::HashMap;
    let mut map = HashMap::new();
    map.insert("00".to_string(), 50);
    map.insert("11".to_string(), 50);
    let counts = MeasurementCounts::from_counts(map);
    assert_eq!(counts.total_shots(), 100);
    assert_eq!(counts.num_outcomes(), 2);
}

#[test]
fn measurement_counts_to_probabilities() {
    let mut counts = MeasurementCounts::new(100);
    counts.add("0".to_string(), 75);
    counts.add("1".to_string(), 25);
    let probs = counts.to_probabilities();
    assert!((probs["0"] - 0.75).abs() < 1e-10);
    assert!((probs["1"] - 0.25).abs() < 1e-10);
}

#[test]
fn measurement_counts_display() {
    let mut counts = MeasurementCounts::new(100);
    counts.add("00".to_string(), 50);
    counts.add("11".to_string(), 50);
    let display = format!("{}", counts);
    assert!(display.contains("Measurement Counts"));
    assert!(display.contains("100 shots"));
}

// ============================================================================
// 6. ExecutionEngine directly
// ============================================================================

#[test]
fn execution_engine_basic() {
    use simq_sim::execution_engine::{ExecutionConfig, ExecutionEngine};
    let config = ExecutionConfig::default();
    let mut engine = ExecutionEngine::new(config);
    let circuit = bell_circuit();
    let mut state = simq_state::AdaptiveState::new(2).unwrap();
    let result = engine.execute(&circuit, &mut state);
    assert!(result.is_ok());
}

#[test]
fn execution_engine_single_gate() {
    use simq_sim::execution_engine::{ExecutionConfig, ExecutionEngine};
    let config = ExecutionConfig::default();
    let mut engine = ExecutionEngine::new(config);
    let circuit = single_x_circuit();
    let mut state = simq_state::AdaptiveState::new(1).unwrap();
    let result = engine.execute(&circuit, &mut state);
    assert!(result.is_ok());
}

// ============================================================================
// 7. VQE/QAOA helpers
// ============================================================================

#[test]
fn qaoa_circuit_basic() {
    let cost_h = vec![(0, 1.0), (1, -0.5)];
    let mixer = vec![0, 1];
    let params = vec![0.5, 0.3];
    let circuit = simq_sim::qaoa_circuit(2, &cost_h, &mixer, 1, &params);
    assert_eq!(circuit.num_qubits(), 2);
    assert!(!circuit.is_empty());
}

#[test]
fn qaoa_circuit_multi_layer() {
    let cost_h = vec![(0, 1.0), (1, 1.0), (2, 1.0)];
    let mixer = vec![0, 1, 2];
    let params = vec![0.1, 0.2, 0.3, 0.4];
    let circuit = simq_sim::qaoa_circuit(3, &cost_h, &mixer, 2, &params);
    assert_eq!(circuit.num_qubits(), 3);
    assert!(circuit.len() > 6);
}

#[test]
fn qaoa_circuit_simulatable() {
    let sim = Simulator::new(SimulatorConfig::default().with_optimization(false));
    let cost_h = vec![(0, 1.0)];
    let mixer = vec![0, 1];
    let params = vec![0.5, 0.3];
    let circuit = simq_sim::qaoa_circuit(2, &cost_h, &mixer, 1, &params);
    let result = sim.run(&circuit);
    assert!(result.is_ok());
}

#[test]
fn vqe_ansatz_basic() {
    let params = vec![0.1, 0.2, 0.3];
    let circuit = simq_sim::vqe_hardware_efficient_ansatz(3, &params);
    assert_eq!(circuit.num_qubits(), 3);
    assert!(!circuit.is_empty());
}

#[test]
fn vqe_ansatz_simulatable() {
    let sim = Simulator::new(SimulatorConfig::default().with_optimization(false));
    let params = vec![0.5, 0.3];
    let circuit = simq_sim::vqe_hardware_efficient_ansatz(2, &params);
    let result = sim.run(&circuit);
    assert!(result.is_ok());
}

// ============================================================================
// 8. Autodiff
// ============================================================================

#[test]
fn differentiable_parameter() {
    let p = DifferentiableParameter::new(1.5);
    assert_eq!(p.value, 1.5);
    assert_eq!(p.get_grad(), 0.0);
    p.set_grad(2.0);
    assert_eq!(p.get_grad(), 2.0);
}

#[test]
fn compute_gradients_ad_basic() {
    let params = vec![
        DifferentiableParameter::new(1.0),
        DifferentiableParameter::new(2.0),
    ];
    let grads = compute_gradients_ad(&params, |p| p[0] * p[0] + p[1] * p[1]);
    assert_eq!(grads.len(), 2);
    assert!((grads[0] - 2.0).abs() < 1e-4, "d/dx(x^2) at x=1 should be ~2: got {}", grads[0]);
    assert!((grads[1] - 4.0).abs() < 1e-4, "d/dy(y^2) at y=2 should be ~4: got {}", grads[1]);
}

// ============================================================================
// 9. Gradient module
// ============================================================================

#[test]
fn gradient_config_default() {
    let config = simq_sim::gradient::GradientConfig::default();
    assert_eq!(config.method, simq_sim::gradient::GradientMethod::ParameterShift);
    assert!(config.parallel);
}

#[test]
fn gradient_result_norm() {
    let result = simq_sim::gradient::GradientResult {
        gradients: vec![3.0, 4.0],
        num_evaluations: 4,
        computation_time: std::time::Duration::from_millis(1),
        method_used: simq_sim::gradient::GradientMethod::FiniteDifference,
    };
    assert!((result.norm() - 5.0).abs() < 1e-10);
    assert_eq!(result.len(), 2);
    assert!(!result.is_empty());
}

#[test]
fn gradient_dual_number() {
    use simq_sim::gradient::Dual;
    let a = Dual::new(2.0, 1.0);
    let b = Dual::new(3.0, 0.0);
    let c = a * b;
    assert!((c.value() - 6.0).abs() < 1e-10);
    assert!((c.derivative() - 3.0).abs() < 1e-10);
}

// ============================================================================
// 10. Multi-qubit simulation
// ============================================================================

#[test]
fn simulate_ghz_3() {
    let sim = Simulator::default();
    let circuit = ghz_circuit(3);
    let result = sim.run(&circuit);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().num_qubits(), 3);
}

#[test]
fn simulate_ghz_5() {
    let sim = Simulator::default();
    let circuit = ghz_circuit(5);
    let result = sim.run(&circuit);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().num_qubits(), 5);
}

#[test]
fn simulate_rotation_gates() {
    let sim = Simulator::default();
    let mut circuit = Circuit::new(2);
    circuit
        .add_gate(Arc::new(RotationX::new(1.0)), &[q(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(RotationY::new(0.5)), &[q(1)])
        .unwrap();
    circuit
        .add_gate(Arc::new(RotationZ::new(0.3)), &[q(0)])
        .unwrap();
    circuit.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();
    let result = sim.run(&circuit);
    assert!(result.is_ok());
}

#[test]
fn simulate_all_pauli_gates() {
    let sim = Simulator::default();
    let mut circuit = Circuit::new(3);
    circuit.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    circuit.add_gate(Arc::new(PauliY), &[q(1)]).unwrap();
    circuit.add_gate(Arc::new(PauliZ), &[q(2)]).unwrap();
    let result = sim.run(&circuit);
    assert!(result.is_ok());
}

#[test]
fn simulate_s_t_gates() {
    let sim = Simulator::default();
    let mut circuit = Circuit::new(2);
    circuit.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    circuit.add_gate(Arc::new(SGate), &[q(0)]).unwrap();
    circuit.add_gate(Arc::new(TGate), &[q(1)]).unwrap();
    let result = sim.run(&circuit);
    assert!(result.is_ok());
}

// ============================================================================
// 11. Memory limit
// ============================================================================

#[test]
fn memory_limit_too_small() {
    let sim = Simulator::new(SimulatorConfig::default().with_memory_limit(16));
    let circuit = bell_circuit();
    let result = sim.run(&circuit);
    assert!(result.is_err(), "Should fail with tiny memory limit");
}

#[test]
fn memory_limit_sufficient() {
    let sim = Simulator::new(SimulatorConfig::default().with_memory_limit(1024 * 1024));
    let circuit = bell_circuit();
    let result = sim.run(&circuit);
    assert!(result.is_ok());
}

// ============================================================================
// 12. Deterministic seed
// ============================================================================

#[test]
fn deterministic_seed() {
    let config = SimulatorConfig::default().with_seed(42);
    let sim = Simulator::new(config);
    let circuit = bell_circuit();
    let r1 = sim.run(&circuit).unwrap();
    let r2 = sim.run(&circuit).unwrap();
    assert_eq!(r1.num_qubits(), r2.num_qubits());
}

// ============================================================================
// 13. Optimization comparison
// ============================================================================

#[test]
fn optimization_reduces_gates() {
    let sim_opt = Simulator::new(
        SimulatorConfig::default()
            .with_optimization(true)
            .with_statistics(true),
    );
    let sim_no = Simulator::new(
        SimulatorConfig::default()
            .with_optimization(false)
            .with_statistics(true),
    );

    let mut circuit = Circuit::new(1);
    circuit.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    circuit.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();

    let r_opt = sim_opt.run(&circuit).unwrap();
    let r_no = sim_no.run(&circuit).unwrap();

    let s_opt = r_opt.statistics.unwrap();
    let s_no = r_no.statistics.unwrap();

    assert!(
        s_opt.optimized_gates < s_no.optimized_gates,
        "Optimization should reduce gates: opt={}, no_opt={}",
        s_opt.optimized_gates,
        s_no.optimized_gates
    );
}

// ============================================================================
// 14. Convergence and classical optimizers
// ============================================================================

#[test]
fn convergence_monitor() {
    let mut config = simq_sim::gradient::MonitorConfig::default();
    config.patience = 5;
    config.window_size = 3;
    let mut monitor = simq_sim::gradient::ConvergenceMonitor::new(config);
    for i in 0..10 {
        let energy = 1.0 / (i as f64 + 1.0);
        let gradient = vec![0.1];
        let params = vec![i as f64 * 0.1];
        monitor.record(i, energy, &gradient, &params);
    }
}

#[test]
fn best_tracker() {
    let mut tracker = simq_sim::gradient::BestTracker::new();
    tracker.update(1.0, &[0.1, 0.2]);
    tracker.update(0.5, &[0.3, 0.4]);
    tracker.update(0.8, &[0.5, 0.6]);
    assert!((tracker.best_energy() - 0.5).abs() < 1e-10);
    assert_eq!(tracker.best_parameters(), &[0.3, 0.4]);
}

// ============================================================================
// 15. Stress tests
// ============================================================================

#[test]
fn stress_many_simulations() {
    let sim = Simulator::default();
    for _ in 0..20 {
        let circuit = bell_circuit();
        let result = sim.run(&circuit);
        assert!(result.is_ok());
    }
}

#[test]
fn stress_large_circuit() {
    let sim = Simulator::new(SimulatorConfig::default().with_statistics(true));
    let mut circuit = Circuit::new(8);
    for i in 0..8 {
        circuit.add_gate(Arc::new(Hadamard), &[q(i)]).unwrap();
    }
    for i in 0..7 {
        circuit.add_gate(Arc::new(CNot), &[q(i), q(i + 1)]).unwrap();
    }
    let result = sim.run(&circuit);
    assert!(result.is_ok());
    let r = result.unwrap();
    assert_eq!(r.num_qubits(), 8);
}

#[test]
fn stress_deep_circuit() {
    let sim = Simulator::default();
    let mut circuit = Circuit::new(2);
    for _ in 0..50 {
        circuit.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
        circuit.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();
    }
    let result = sim.run(&circuit);
    assert!(result.is_ok());
}

#[test]
fn stress_all_configs() {
    let configs = vec![
        SimulatorConfig::default(),
        SimulatorConfig::fast(),
        SimulatorConfig::accurate(),
        SimulatorConfig::debug(),
    ];
    for config in configs {
        let sim = Simulator::new(config);
        let circuit = bell_circuit();
        let result = sim.run(&circuit);
        assert!(result.is_ok());
    }
}

// ============================================================================
// 16. Integration tests
// ============================================================================

#[test]
fn full_vqe_workflow() {
    let params = vec![0.1, 0.2];
    let circuit = simq_sim::vqe_hardware_efficient_ansatz(2, &params);
    let sim = Simulator::new(SimulatorConfig::default().with_statistics(true));
    let result = sim.run(&circuit).unwrap();
    assert_eq!(result.num_qubits(), 2);
    assert!(result.statistics.is_some());
}

#[test]
fn full_qaoa_workflow() {
    let cost_h = vec![(0, 1.0), (1, -1.0)];
    let mixer = vec![0, 1];
    let params = vec![0.5, 0.3];
    let circuit = simq_sim::qaoa_circuit(2, &cost_h, &mixer, 1, &params);
    let sim = Simulator::new(SimulatorConfig::default().with_statistics(true));
    let result = sim.run(&circuit).unwrap();
    assert_eq!(result.num_qubits(), 2);
    assert!(result.statistics.is_some());
}

#[test]
fn simulate_then_analyze() {
    let sim = Simulator::new(SimulatorConfig::default().with_statistics(true));
    let circuit = ghz_circuit(4);
    let result = sim.run(&circuit).unwrap();
    let stats = result.statistics.unwrap();
    assert!(stats.gates_executed > 0);
    assert!(stats.peak_memory_bytes > 0);
    let _ = stats.optimization_ratio();
    let _ = stats.peak_memory_mb();
    let _ = stats.compilation_overhead_percent();
}

// ============================================================================
// 8. Multi-qubit gate fusion (issue #76 follow-up): correctness at/above the
// qubit-count gate, and correctness of the persistent fusion-structure cache
// across repeated Simulator::run() calls with different gate parameters.
// ============================================================================

/// VQE-shaped circuit (H + RY-CNOT-RZ neighborhoods), matching
/// `simq-sim/examples/scaling_probe.rs`'s pattern, parametrized by a single
/// angle so repeated calls can vary it. `num_qubits` must be >=
/// `SimulatorConfig::parallel_threshold` (default 18) for multi-qubit block
/// fusion to engage at all (see `FusionConfig::parallel_threshold_qubits`
/// in simq-compiler) — the whole point of gating fusion width on qubit
/// count is that below that, this code path is provably unreached.
fn vqe_shaped_circuit(num_qubits: usize, theta: f64) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);
    for i in 0..num_qubits {
        circuit.add_gate(Arc::new(Hadamard), &[q(i)]).unwrap();
    }
    for i in 0..num_qubits {
        circuit
            .add_gate(Arc::new(RotationY::new(theta + i as f64 * 0.1)), &[q(i)])
            .unwrap();
    }
    for i in 0..num_qubits - 1 {
        circuit.add_gate(Arc::new(CNot), &[q(i), q(i + 1)]).unwrap();
    }
    for i in 0..num_qubits {
        circuit
            .add_gate(Arc::new(RotationZ::new(theta * 0.5 + i as f64 * 0.07)), &[q(i)])
            .unwrap();
    }
    circuit
}

fn assert_states_close(a: &[Complex64], b: &[Complex64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).norm() < tol, "state mismatch: {:?} vs {:?}", x, y);
    }
}

/// At 20 qubits (above the default 18-qubit fusion-width gate), a fused
/// (optimized) run must agree with an unoptimized run — this is the
/// integration-level version of `simq-compiler`'s own multi-ordering
/// correctness test, now exercised through the real dense kernels
/// (`apply_three_qubit_dense` and friends), not just `circuit_matrix`.
#[test]
fn multi_qubit_fusion_matches_unoptimized_above_threshold() {
    let circuit = vqe_shaped_circuit(20, 0.83);

    let optimized_sim = Simulator::new(SimulatorConfig::default().with_optimization_level(2));
    let unoptimized_sim = Simulator::new(SimulatorConfig::default().with_optimization(false));

    let optimized = optimized_sim.run(&circuit).unwrap();
    let unoptimized = unoptimized_sim.run(&circuit).unwrap();

    assert_states_close(&optimized.state.to_dense_vec(), &unoptimized.state.to_dense_vec(), 1e-9);
}

/// Below the fusion-width gate (16 qubits, matching BENCHMARKS.md's largest
/// published row), optimized and unoptimized runs must still agree — this
/// doesn't prove the <16q *code path* is untouched (that's
/// `simq-compiler`'s own `test_below_threshold_dispatches_to_legacy_path_exactly`),
/// but it does confirm the qubit-count gate doesn't introduce an
/// end-to-end correctness regression for the currently-published sizes.
#[test]
fn fusion_matches_unoptimized_below_threshold() {
    let circuit = vqe_shaped_circuit(16, 0.83);

    let optimized_sim = Simulator::new(SimulatorConfig::default().with_optimization_level(2));
    let unoptimized_sim = Simulator::new(SimulatorConfig::default().with_optimization(false));

    let optimized = optimized_sim.run(&circuit).unwrap();
    let unoptimized = unoptimized_sim.run(&circuit).unwrap();

    assert_states_close(&optimized.state.to_dense_vec(), &unoptimized.state.to_dense_vec(), 1e-9);
}

/// The critical safety test for the persistent fusion-structure cache
/// (`Simulator::fusion_cache`): running the *same-shaped* circuit twice
/// through one `Simulator` with *different* rotation angles must produce
/// results matching each angle's own unoptimized baseline — not a stale
/// result reused from the first call. This is exactly the scenario a naive
/// "cache the compiled circuit" approach (like
/// `simq_compiler::cached_compiler::CachedCompiler`, which caches baked-in
/// matrices) would get wrong; `Simulator`'s cache only ever stores fusion
/// *block structure*, so it must stay correct here.
#[test]
fn fusion_cache_stays_correct_across_different_angles() {
    let sim = Simulator::new(SimulatorConfig::default().with_optimization_level(2));
    let unopt_sim = Simulator::new(SimulatorConfig::default().with_optimization(false));

    for theta in [0.2, 1.7, -0.9] {
        let circuit = vqe_shaped_circuit(20, theta);
        let cached_run = sim.run(&circuit).unwrap();
        let baseline = unopt_sim.run(&circuit).unwrap();
        assert_states_close(&cached_run.state.to_dense_vec(), &baseline.state.to_dense_vec(), 1e-9);
    }
}

/// End-to-end proof that the cache wiring actually engages through
/// `Simulator::run` (not just that the underlying `FusionStructureCache`
/// type works in isolation, which `simq-compiler`'s own tests already
/// cover): three same-shaped, differently-parametrized circuits run
/// through one `Simulator` should produce more cache hits than misses.
#[test]
fn fusion_cache_hits_across_repeated_same_shape_runs() {
    let sim = Simulator::new(SimulatorConfig::default().with_optimization_level(2));

    for theta in [0.11, 0.42, -1.3, 2.5, 0.9] {
        let circuit = vqe_shaped_circuit(20, theta);
        sim.run(&circuit).unwrap();
    }

    // First run is necessarily a miss (nothing cached yet); the remaining
    // four calls compile the exact same circuit *shape* and must hit.
    assert!(
        sim.fusion_cache_hits() >= 4,
        "expected at least 4 fusion-structure cache hits across 5 same-shaped runs, got {}",
        sim.fusion_cache_hits()
    );
}

// ============================================================================
// Parallel-threshold equivalence (issue #76)
// ============================================================================

/// Forcing the rayon kernels on (tiny threshold) vs off (huge threshold)
/// must not change the simulated state: parallel chunking only affects
/// summation order, so probabilities must agree to floating-point noise.
#[test]
fn parallel_and_serial_kernels_agree() {
    let n = 8;
    let mut c = Circuit::new(n);
    for layer in 0..3 {
        for qq in 0..n {
            c.add_gate(
                Arc::new(RotationY::new(0.3 * (layer as f64 + 1.0) + 0.05 * qq as f64)),
                &[q(qq)],
            )
            .unwrap();
        }
        for qq in 0..n - 1 {
            c.add_gate(Arc::new(CNot), &[q(qq), q(qq + 1)]).unwrap();
        }
    }

    let run = |par_qubits: usize| {
        let mut cfg = SimulatorConfig::default();
        cfg.parallel_threshold = par_qubits;
        cfg.optimize_circuit = false;
        Simulator::new(cfg).run(&c).unwrap().state.to_dense_vec()
    };

    let serial = run(30); // 2^30 amplitudes: kernels stay single-threaded
    let parallel = run(1); // 2 amplitudes: every eligible kernel uses rayon

    assert_eq!(serial.len(), parallel.len());
    let max_diff = serial
        .iter()
        .zip(parallel.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    assert!(max_diff < 1e-12, "parallel vs serial max amplitude diff {max_diff:e}");
}
