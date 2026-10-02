//! Qubit routing and SWAP insertion algorithms
//!
//! This module provides algorithms for mapping logical qubits to physical qubits
//! and inserting SWAP gates to handle limited connectivity constraints.

use crate::{BackendError, ConnectivityGraph, QubitMapping, Result};
use std::collections::{HashSet, VecDeque};

/// Router for handling qubit connectivity constraints
///
/// The router maps logical qubits to physical qubits and inserts SWAP gates
/// when two-qubit gates need to operate on non-adjacent qubits.
pub struct Router {
    strategy: RoutingStrategy,
}

impl Router {
    /// Create a new router with the specified strategy
    pub fn new(strategy: RoutingStrategy) -> Self {
        Self { strategy }
    }

    /// Find initial qubit mapping
    ///
    /// Chooses an initial mapping of logical qubits to physical qubits.
    /// Different strategies can be used for optimization.
    pub fn initial_mapping(
        &self,
        num_logical_qubits: usize,
        connectivity: &ConnectivityGraph,
    ) -> Result<QubitMapping> {
        if num_logical_qubits > connectivity.num_qubits() {
            return Err(BackendError::CapabilityExceeded(format!(
                "Circuit requires {} qubits, connectivity graph has {}",
                num_logical_qubits,
                connectivity.num_qubits()
            )));
        }

        match self.strategy {
            RoutingStrategy::Identity => {
                // Simple identity mapping: L0→P0, L1→P1, ...
                Ok(QubitMapping::identity(num_logical_qubits))
            },
            RoutingStrategy::HighestDegree => {
                // Map to physical qubits with highest degree (most connections)
                self.highest_degree_mapping(num_logical_qubits, connectivity)
            },
            RoutingStrategy::Subgraph => {
                // Find densely connected subgraph
                self.subgraph_mapping(num_logical_qubits, connectivity)
            },
        }
    }

    /// Find SWAP chain to connect two qubits
    ///
    /// Given a qubit mapping and two logical qubits that need to interact,
    /// find the shortest sequence of SWAPs to bring them adjacent.
    pub fn find_swap_chain(
        &self,
        connectivity: &ConnectivityGraph,
        mapping: &QubitMapping,
        logical_q1: usize,
        logical_q2: usize,
    ) -> Result<Vec<SwapGate>> {
        let phys_q1 = mapping
            .get_physical(logical_q1)
            .ok_or_else(|| BackendError::Other("Invalid logical qubit".to_string()))?;
        let phys_q2 = mapping
            .get_physical(logical_q2)
            .ok_or_else(|| BackendError::Other("Invalid logical qubit".to_string()))?;

        // Check if already connected
        if connectivity.are_connected(phys_q1, phys_q2) {
            return Ok(vec![]); // No SWAPs needed
        }

        // Find shortest path
        let path = connectivity
            .shortest_path(phys_q1, phys_q2)
            .ok_or_else(|| {
                BackendError::Other(format!(
                    "No path between physical qubits {} and {}",
                    phys_q1, phys_q2
                ))
            })?;

        // Convert path to SWAP chain
        let mut swaps = Vec::new();
        for i in 0..path.len() - 2 {
            swaps.push(SwapGate {
                qubit1: path[i],
                qubit2: path[i + 1],
            });
        }

        Ok(swaps)
    }

    /// Highest degree mapping heuristic
    fn highest_degree_mapping(
        &self,
        num_logical: usize,
        connectivity: &ConnectivityGraph,
    ) -> Result<QubitMapping> {
        // Sort physical qubits by degree (descending)
        let mut qubit_degrees: Vec<_> = (0..connectivity.num_qubits())
            .map(|q| (q, connectivity.degree(q)))
            .collect();
        qubit_degrees.sort_by_key(|b| std::cmp::Reverse(b.1));

        // Map logical qubits to highest degree physical qubits
        let mapping: Vec<_> = qubit_degrees
            .iter()
            .take(num_logical)
            .map(|(q, _)| *q)
            .collect();

        Ok(QubitMapping::from_vec(mapping, connectivity.num_qubits()))
    }

    /// Subgraph mapping heuristic
    fn subgraph_mapping(
        &self,
        num_logical: usize,
        connectivity: &ConnectivityGraph,
    ) -> Result<QubitMapping> {
        // Find densely connected subgraph using greedy approach
        // Start with highest degree qubit and expand

        let mut selected = HashSet::new();
        let mut candidates = VecDeque::new();

        // Find qubit with highest degree
        let start_qubit = (0..connectivity.num_qubits())
            .max_by_key(|&q| connectivity.degree(q))
            .unwrap_or(0);

        selected.insert(start_qubit);
        if let Some(neighbors) = connectivity.neighbors(start_qubit) {
            for &n in neighbors {
                candidates.push_back(n);
            }
        }

        // Expand greedily
        while selected.len() < num_logical && !candidates.is_empty() {
            // Pick candidate with most connections to selected set
            let best = candidates
                .iter()
                .max_by_key(|&&q| {
                    connectivity
                        .neighbors(q)
                        .map(|neighbors| neighbors.intersection(&selected).count())
                        .unwrap_or(0)
                })
                .copied();

            if let Some(best) = best {
                selected.insert(best);
                candidates.retain(|&q| q != best);

                // Add new neighbors
                if let Some(neighbors) = connectivity.neighbors(best) {
                    for &n in neighbors {
                        if !selected.contains(&n) && !candidates.contains(&n) {
                            candidates.push_back(n);
                        }
                    }
                }
            } else {
                break;
            }
        }

        // If we didn't find enough qubits, fill with remaining
        let mapping: Vec<_> = if selected.len() < num_logical {
            let mut m: Vec<_> = selected.into_iter().collect();
            for q in 0..connectivity.num_qubits() {
                if !m.contains(&q) && m.len() < num_logical {
                    m.push(q);
                }
            }
            m
        } else {
            selected.into_iter().take(num_logical).collect()
        };

        Ok(QubitMapping::from_vec(mapping, connectivity.num_qubits()))
    }
}

/// Routing strategy for initial qubit mapping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoutingStrategy {
    /// Identity mapping (L0→P0, L1→P1, ...)
    Identity,

    /// Map to physical qubits with highest connectivity
    HighestDegree,

    /// Find densely connected subgraph
    #[default]
    Subgraph,
}

/// Represents a SWAP gate insertion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwapGate {
    /// First physical qubit
    pub qubit1: usize,

    /// Second physical qubit
    pub qubit2: usize,
}

impl SwapGate {
    /// Create a new SWAP gate
    pub fn new(qubit1: usize, qubit2: usize) -> Self {
        Self { qubit1, qubit2 }
    }

    /// Apply this SWAP to a qubit mapping
    pub fn apply(&self, mapping: &mut QubitMapping) {
        mapping.swap(self.qubit1, self.qubit2);
    }

    /// Get the ordered qubits (smaller first)
    pub fn ordered(&self) -> (usize, usize) {
        if self.qubit1 < self.qubit2 {
            (self.qubit1, self.qubit2)
        } else {
            (self.qubit2, self.qubit1)
        }
    }
}

/// SABRE routing algorithm
///
/// SWAP-based Bidirectional heuristic search for qubit Routing.
/// Reference: https://arxiv.org/abs/1809.02573
///
/// Note: the full SABRE algorithm requires the circuit's two-qubit gate
/// list to evaluate candidate SWAPs. The current `route()` signature does
/// not receive a circuit, so it cannot produce a valid SWAP plan and
/// returns an error rather than a misleading empty plan. Use
/// [`Router::find_swap_chain`] (greedy shortest-path insertion, as used by
/// the transpiler's `map_and_route`) for routing today.
pub struct SabreRouter {
    /// Lookahead window size
    lookahead: usize,

    /// Decay factor for heuristic
    decay: f64,
}

impl SabreRouter {
    /// Create a new SABRE router
    pub fn new(lookahead: usize, decay: f64) -> Self {
        Self { lookahead, decay }
    }

    /// Route a circuit using SABRE algorithm
    ///
    /// Returns an error: full SABRE routing is not implemented yet, and the
    /// method does not receive the circuit gates it would need to compute a
    /// plan. Returning `Ok(vec![])` would falsely claim every circuit is
    /// routable with zero SWAPs, so callers get an explicit error instead.
    pub fn route(
        &self,
        _num_qubits: usize,
        _connectivity: &ConnectivityGraph,
    ) -> Result<Vec<SwapGate>> {
        Err(BackendError::Other(
            "SabreRouter::route is not implemented: circuit-aware SABRE routing \
             requires the gate list; use Router::find_swap_chain for greedy \
             shortest-path SWAP insertion"
                .to_string(),
        ))
    }

    /// Calculate SABRE heuristic score for a SWAP
    ///
    /// Scores a candidate SWAP by the total shortest-path distance between
    /// each upcoming two-qubit pair after the SWAP is applied, weighted by
    /// `decay` per lookahead step. Lower is better.
    ///
    /// Currently exercised only by tests: `route()` errors until the
    /// circuit-aware SABRE pass lands, at which point this will score
    /// candidate SWAPs.
    #[allow(dead_code)]
    fn heuristic_score(
        &self,
        swap: &SwapGate,
        mapping: &QubitMapping,
        connectivity: &ConnectivityGraph,
        upcoming_pairs: &[(usize, usize)],
    ) -> f64 {
        let mut trial = mapping.clone();
        swap.apply(&mut trial);

        let mut score = 0.0;
        let mut weight = 1.0;
        for (i, &(l1, l2)) in upcoming_pairs.iter().enumerate() {
            if i >= self.lookahead {
                break;
            }
            let (Some(p1), Some(p2)) = (trial.get_physical(l1), trial.get_physical(l2)) else {
                continue;
            };
            let dist = connectivity
                .shortest_path(p1, p2)
                .map(|p| p.len().saturating_sub(1))
                .unwrap_or(usize::MAX / 2) as f64;
            score += weight * dist;
            weight *= self.decay;
        }
        score
    }
}

impl Default for SabreRouter {
    fn default() -> Self {
        Self::new(20, 0.99)
    }
}

/// Routing statistics
#[derive(Debug, Clone)]
pub struct RoutingStats {
    /// Number of SWAP gates inserted
    pub swap_count: usize,

    /// Total CNOT count (each SWAP = 3 CNOTs)
    pub cnot_count: usize,

    /// Number of layers added by SWAPs
    pub depth_increase: usize,

    /// Initial mapping used
    pub initial_mapping: Vec<usize>,
}

impl RoutingStats {
    /// Create routing statistics
    pub fn new(swap_count: usize, initial_mapping: Vec<usize>) -> Self {
        Self {
            swap_count,
            cnot_count: swap_count * 3,
            depth_increase: swap_count, // Conservative estimate
            initial_mapping,
        }
    }

    /// Calculate gate overhead factor
    pub fn gate_overhead(&self, original_gates: usize) -> f64 {
        if original_gates == 0 {
            return 0.0;
        }
        self.cnot_count as f64 / original_gates as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let router = Router::new(RoutingStrategy::Identity);
        assert_eq!(router.strategy, RoutingStrategy::Identity);

        let router = Router::new(RoutingStrategy::HighestDegree);
        assert_eq!(router.strategy, RoutingStrategy::HighestDegree);
    }

    #[test]
    fn test_identity_mapping() {
        let router = Router::new(RoutingStrategy::Identity);
        let connectivity = ConnectivityGraph::all_to_all(5);

        let mapping = router.initial_mapping(3, &connectivity).unwrap();

        assert_eq!(mapping.get_physical(0), Some(0));
        assert_eq!(mapping.get_physical(1), Some(1));
        assert_eq!(mapping.get_physical(2), Some(2));
    }

    #[test]
    fn test_highest_degree_mapping() {
        let router = Router::new(RoutingStrategy::HighestDegree);

        // Create connectivity with varying degrees
        // 0-1-2 (linear chain)
        let mut connectivity = ConnectivityGraph::new(5, false);
        connectivity.add_edge(0, 1);
        connectivity.add_edge(1, 2);
        connectivity.add_edge(2, 3);
        connectivity.add_edge(3, 4);
        // Add extra connection to qubit 2
        connectivity.add_edge(2, 4);

        let mapping = router.initial_mapping(3, &connectivity).unwrap();

        // Qubit 2 has highest degree (3 connections: 1, 3, 4)
        // Should be included in mapping
        let physical_qubits: Vec<_> = (0..3).filter_map(|l| mapping.get_physical(l)).collect();

        assert!(physical_qubits.contains(&2)); // Highest degree qubit
        assert_eq!(physical_qubits.len(), 3);
    }

    #[test]
    fn test_swap_chain_connected() {
        let router = Router::new(RoutingStrategy::Identity);
        let connectivity = ConnectivityGraph::linear_chain(5);
        let mapping = QubitMapping::identity(5);

        // Qubits 0 and 1 are connected
        let swaps = router
            .find_swap_chain(&connectivity, &mapping, 0, 1)
            .unwrap();

        assert_eq!(swaps.len(), 0); // No SWAPs needed
    }

    #[test]
    fn test_swap_chain_not_connected() {
        let router = Router::new(RoutingStrategy::Identity);
        let connectivity = ConnectivityGraph::linear_chain(5);
        let mapping = QubitMapping::identity(5);

        // Qubits 0 and 4 are not directly connected
        // Path: 0-1-2-3-4
        let swaps = router
            .find_swap_chain(&connectivity, &mapping, 0, 4)
            .unwrap();

        // Need 3 SWAPs: (0,1), (1,2), (2,3) to bring 0 and 4 adjacent
        assert_eq!(swaps.len(), 3);
    }

    #[test]
    fn test_swap_gate_ordered() {
        let swap1 = SwapGate::new(5, 2);
        assert_eq!(swap1.ordered(), (2, 5));

        let swap2 = SwapGate::new(1, 3);
        assert_eq!(swap2.ordered(), (1, 3));
    }

    #[test]
    fn test_swap_gate_apply() {
        let mut mapping = QubitMapping::identity(3);
        let swap = SwapGate::new(0, 1);

        swap.apply(&mut mapping);

        // After SWAP, L0→P1 and L1→P0
        assert_eq!(mapping.get_physical(0), Some(1));
        assert_eq!(mapping.get_physical(1), Some(0));
    }

    #[test]
    fn test_routing_stats() {
        let stats = RoutingStats::new(5, vec![0, 1, 2]);

        assert_eq!(stats.swap_count, 5);
        assert_eq!(stats.cnot_count, 15); // 5 SWAPs × 3 CNOTs
        assert_eq!(stats.gate_overhead(100), 0.15); // 15/100 = 15%
    }

    #[test]
    fn test_sabre_router_creation() {
        let router = SabreRouter::new(20, 0.99);
        assert_eq!(router.lookahead, 20);
        assert_eq!(router.decay, 0.99);

        let default = SabreRouter::default();
        assert_eq!(default.lookahead, 20);
    }

    #[test]
    fn test_subgraph_mapping() {
        let router = Router::new(RoutingStrategy::Subgraph);
        let connectivity = ConnectivityGraph::grid(3, 3);

        let mapping = router.initial_mapping(4, &connectivity).unwrap();

        // Should map to a connected subgraph
        let physical_qubits: Vec<_> = (0..4).filter_map(|l| mapping.get_physical(l)).collect();

        assert_eq!(physical_qubits.len(), 4);

        // Verify connectivity between mapped qubits
        let mut connected_count = 0;
        for i in 0..physical_qubits.len() {
            for j in i + 1..physical_qubits.len() {
                if connectivity.are_connected(physical_qubits[i], physical_qubits[j]) {
                    connected_count += 1;
                }
            }
        }

        // At least some qubits should be connected
        assert!(connected_count > 0);
    }

    // Tests for previously uncovered lines

    #[test]
    fn test_find_swap_chain_no_path() {
        // Covers lines 83-85: no path between physical qubits
        // Create a disconnected graph where qubits 0 and 1 have no path between them
        let mut connectivity = ConnectivityGraph::new(4, false);
        // Only connect 0-1 and 2-3, leaving 1 and 2 disconnected
        connectivity.add_edge(0, 1);
        connectivity.add_edge(2, 3);

        let router = Router::new(RoutingStrategy::Identity);
        let mapping = crate::QubitMapping::identity(4);

        // Try to find swap chain between q0 and q2 (no path: 0-1 and 2-3 are separate components)
        let result = router.find_swap_chain(&connectivity, &mapping, 0, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_initial_mapping_too_many_qubits() {
        // Covers line 33-37: more logical qubits than physical
        let router = Router::new(RoutingStrategy::Identity);
        let connectivity = ConnectivityGraph::linear_chain(3);

        // Try to map 5 logical qubits to a 3-qubit graph
        let result = router.initial_mapping(5, &connectivity);
        assert!(result.is_err());
        match result.err().unwrap() {
            crate::BackendError::CapabilityExceeded(msg) => {
                assert!(msg.contains("5"));
            },
            other => panic!("Expected CapabilityExceeded, got {:?}", other),
        }
    }

    #[test]
    fn test_subgraph_mapping_fill_remaining() {
        // Covers lines 179-182, 185: subgraph_mapping fallback when not enough connected qubits
        // Use a graph with a star topology - center + leaves
        // With sparse connectivity, we might need to fill remaining qubits
        let mut connectivity = ConnectivityGraph::new(6, false);
        connectivity.add_edge(0, 1); // Only one edge
                                     // Now map 5 qubits to this 6-node graph with only 1 edge
                                     // The subgraph algorithm will find {0, 1} connected, then must fill from {2,3,4,5}
        let router = Router::new(RoutingStrategy::Subgraph);
        let mapping = router.initial_mapping(5, &connectivity).unwrap();

        let physical_qubits: Vec<_> = (0..5).filter_map(|l| mapping.get_physical(l)).collect();
        assert_eq!(physical_qubits.len(), 5);
        // All physical qubits should be unique
        let mut unique = physical_qubits.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 5);
    }

    #[test]
    fn test_routing_stats_gate_overhead_zero_gates() {
        // Covers line 280: gate_overhead with 0 original gates
        let stats = RoutingStats::new(3, vec![0, 1, 2]);
        assert_eq!(stats.gate_overhead(0), 0.0);
    }

    #[test]
    fn test_sabre_router_route() {
        // SabreRouter::route is honestly unimplemented: without the circuit
        // gate list it cannot produce a plan, so it must error rather than
        // return a misleading empty (zero-SWAP) success.
        let router = SabreRouter::default();
        let connectivity = ConnectivityGraph::linear_chain(4);
        let result = router.route(4, &connectivity);
        assert!(result.is_err());
    }

    #[test]
    fn test_sabre_heuristic_score_prefers_closer_swap() {
        // Pair (1,3) on 0-1-2-3 chain: identity distance is 2.
        // SWAP(1,2) brings them adjacent (P2,P3 -> distance 1);
        // SWAP(0,1) pushes L1 to P0 (P0,P3 -> distance 3).
        let router = SabreRouter::new(20, 0.99);
        let connectivity = ConnectivityGraph::linear_chain(4);
        let mapping = QubitMapping::identity(4);
        let upcoming = vec![(1usize, 3usize)];
        let closer = SwapGate::new(1, 2);
        let farther = SwapGate::new(0, 1);
        let closer_score = router.heuristic_score(&closer, &mapping, &connectivity, &upcoming);
        let farther_score = router.heuristic_score(&farther, &mapping, &connectivity, &upcoming);
        assert!(closer_score.is_finite());
        assert!(farther_score.is_finite());
        assert!(
            closer_score < farther_score,
            "closer SWAP ({closer_score}) should beat farther SWAP ({farther_score})"
        );
        assert_eq!(router.heuristic_score(&closer, &mapping, &connectivity, &[]), 0.0);
    }
}
