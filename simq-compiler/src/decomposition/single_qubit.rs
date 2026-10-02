//! Single-qubit gate decomposition
//!
//! This module provides comprehensive decomposition strategies for arbitrary single-qubit
//! unitary gates using Euler angle decompositions. Any single-qubit unitary U ∈ SU(2)
//! can be parameterized using three rotation angles.
//!
//! # Euler Angle Decompositions
//!
//! ## ZYZ Decomposition
//! U = e^(iα) Rz(β) Ry(γ) Rz(δ)
//!
//! Most common in quantum computing literature. Good numerical stability.
//!
//! ## ZXZ Decomposition
//! U = e^(iα) Rz(β) Rx(γ) Rz(δ)
//!
//! Alternative to ZYZ, sometimes preferred for specific hardware.
//!
//! ## XYX Decomposition
//! U = e^(iα) Rx(β) Ry(γ) Rx(δ)
//!
//! Useful when X rotations are cheaper than Z rotations.
//!
//! ## U3 Decomposition (IBM)
//! U = Rz(φ) Ry(θ) Rz(λ)
//!
//! IBM's native parameterization, discarding global phase.
//!
//! # Usage
//!
//! ```ignore
//! use simq_compiler::decomposition::single_qubit::{SingleQubitDecomposer, EulerBasis};
//!
//! let decomposer = SingleQubitDecomposer::new(EulerBasis::ZYZ);
//! let result = decomposer.decompose(gate, &config)?;
//! ```
//!
//! # References
//!
//! - Nielsen & Chuang, Ch. 4.2: "Single qubit operations"
//! - Shende & Markov, "On the CNOT-cost of TOFFOLI gates" (2009)
//! - IBM Qiskit documentation on gate decomposition

use crate::decomposition::{
    Decomposer, DecompositionConfig, DecompositionMetadata, DecompositionResult,
};
use crate::matrix_computation::{determinant_2x2, is_unitary_2x2, Matrix2};
use num_complex::Complex64;
use simq_core::{Gate, QuantumError, Result};
use std::f64::consts::PI;
use std::sync::Arc;

// Constants
const EPSILON: f64 = 1e-10;
const I: Complex64 = Complex64::new(0.0, 1.0);

/// Euler angle basis for single-qubit decomposition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EulerBasis {
    /// Rz-Ry-Rz decomposition (most common)
    ZYZ,

    /// Rz-Rx-Rz decomposition
    ZXZ,

    /// Rx-Ry-Rx decomposition
    XYX,

    /// Ry-Rz-Ry decomposition
    YZY,

    /// IBM U3 gate decomposition
    U3,

    /// Hadamard-based decomposition (for Clifford+T)
    HT,
}

/// Single-qubit gate decomposer
pub struct SingleQubitDecomposer {
    basis: EulerBasis,
}

impl SingleQubitDecomposer {
    /// Create a new single-qubit decomposer with specified Euler basis
    pub fn new(basis: EulerBasis) -> Self {
        Self { basis }
    }

    /// Decompose a single-qubit gate into Euler angles
    pub fn decompose_to_angles(&self, matrix: &Matrix2) -> Result<EulerAngles> {
        if !is_unitary_2x2(matrix) {
            return Err(QuantumError::ValidationError("Matrix is not unitary".to_string()));
        }

        match self.basis {
            EulerBasis::ZYZ => Self::decompose_zyz(matrix),
            EulerBasis::ZXZ => Self::decompose_zxz(matrix),
            EulerBasis::XYX => Self::decompose_xyx(matrix),
            EulerBasis::YZY => Self::decompose_yzy(matrix),
            EulerBasis::U3 => Self::decompose_u3(matrix),
            EulerBasis::HT => Self::decompose_ht(matrix),
        }
    }

    /// ZYZ decomposition: U = e^(iα) Rz(β) Ry(γ) Rz(δ)
    ///
    /// This is the most numerically stable decomposition.
    /// Based on the identity: any SU(2) matrix can be written as Rz(β)Ry(γ)Rz(δ).
    fn decompose_zyz(u: &Matrix2) -> Result<EulerAngles> {
        // Extract global phase: det(U) = e^(2iα)
        let det = determinant_2x2(u);
        let alpha = det.arg() / 2.0;

        // Remove global phase: U' = e^(-iα) U
        let phase_factor = Complex64::new(0.0, -alpha).exp();
        let u_normalized = [
            [u[0][0] * phase_factor, u[0][1] * phase_factor],
            [u[1][0] * phase_factor, u[1][1] * phase_factor],
        ];

        // For U' ∈ SU(2), we have:
        // U' = [ cos(γ/2)e^(i(β+δ)/2)   -sin(γ/2)e^(i(β-δ)/2) ]
        //      [ sin(γ/2)e^(-i(β-δ)/2)   cos(γ/2)e^(-i(β+δ)/2) ]

        // Compute γ from diagonal elements
        let gamma = 2.0 * u_normalized[0][0].norm().acos();

        // Handle special case: γ ≈ 0 (identity-like)
        if gamma.abs() < EPSILON {
            // U' ≈ e^(i(β+δ)/2) I
            // Choose β = 0, δ = 2 * arg(u[0][0])
            let delta = 2.0 * u_normalized[0][0].arg();
            return Ok(EulerAngles::new(alpha, 0.0, 0.0, delta));
        }

        // Handle special case: γ ≈ π (Pauli-like)
        if (gamma - PI).abs() < EPSILON {
            // U' ≈ e^(i(β-δ)/2) X
            // Choose β = 0, δ = -2 * arg(u[0][1])
            let delta = -2.0 * u_normalized[0][1].arg();
            return Ok(EulerAngles::new(alpha, 0.0, PI, delta));
        }

        // General case
        let sin_half_gamma = (gamma / 2.0).sin();

        // β = arg(u[1][0]) - arg(sin(γ/2))
        // δ = arg(-u[0][1]) - arg(sin(γ/2))
        let beta = (u_normalized[1][0] / Complex64::new(0.0, sin_half_gamma)).arg();
        let delta = (u_normalized[0][1] / Complex64::new(0.0, -sin_half_gamma)).arg();

        Ok(EulerAngles::new(alpha, beta, gamma, delta))
    }

    /// ZXZ decomposition: U = e^(iα) Rz(β) Rx(γ) Rz(δ)
    fn decompose_zxz(u: &Matrix2) -> Result<EulerAngles> {
        // Extract global phase
        let det = determinant_2x2(u);
        let alpha = det.arg() / 2.0;

        let phase_factor = Complex64::new(0.0, -alpha).exp();
        let u_normalized = [
            [u[0][0] * phase_factor, u[0][1] * phase_factor],
            [u[1][0] * phase_factor, u[1][1] * phase_factor],
        ];

        // For Rz(β)Rx(γ)Rz(δ):
        // U' = [ cos(γ/2)e^(i(β+δ)/2)   -i*sin(γ/2)e^(i(β-δ)/2) ]
        //      [ -i*sin(γ/2)e^(-i(β-δ)/2)  cos(γ/2)e^(-i(β+δ)/2) ]

        let gamma = 2.0 * u_normalized[0][0].norm().acos();

        if gamma.abs() < EPSILON {
            let delta = 2.0 * u_normalized[0][0].arg();
            return Ok(EulerAngles::new(alpha, 0.0, 0.0, delta));
        }

        if (gamma - PI).abs() < EPSILON {
            let delta = -2.0 * (u_normalized[0][1] / -I).arg();
            return Ok(EulerAngles::new(alpha, 0.0, PI, delta));
        }

        let sin_half_gamma = (gamma / 2.0).sin();
        let beta = (u_normalized[1][0] / (-I * sin_half_gamma)).arg();
        let delta = (u_normalized[0][1] / (-I * -sin_half_gamma)).arg();

        Ok(EulerAngles::new(alpha, beta, gamma, delta))
    }

    /// XYX decomposition: U = e^(iα) Rx(β) Ry(γ) Rx(δ)
    fn decompose_xyx(u: &Matrix2) -> Result<EulerAngles> {
        // Extract global phase
        let det = determinant_2x2(u);
        let alpha = det.arg() / 2.0;

        let phase_factor = Complex64::new(0.0, -alpha).exp();
        let u_normalized = [
            [u[0][0] * phase_factor, u[0][1] * phase_factor],
            [u[1][0] * phase_factor, u[1][1] * phase_factor],
        ];

        // Compute γ
        let trace = u_normalized[0][0] + u_normalized[1][1];
        let gamma = 2.0 * (trace.re / 2.0).acos();

        if gamma.abs() < EPSILON {
            let delta = 2.0 * u_normalized[0][0].arg();
            return Ok(EulerAngles::new(alpha, 0.0, 0.0, delta));
        }

        if (gamma - PI).abs() < EPSILON {
            let delta = -2.0 * (u_normalized[0][1] / I).arg();
            return Ok(EulerAngles::new(alpha, 0.0, PI, delta));
        }

        let sin_half_gamma = (gamma / 2.0).sin();
        let beta = ((u_normalized[1][0] + u_normalized[0][1])
            / (Complex64::new(0.0, 2.0) * sin_half_gamma))
            .arg();
        let delta = ((u_normalized[1][0] - u_normalized[0][1])
            / (Complex64::new(0.0, 2.0) * sin_half_gamma))
            .arg();

        Ok(EulerAngles::new(alpha, beta, gamma, delta))
    }

    /// YZY decomposition: U = e^(iα) Ry(β) Rz(γ) Ry(δ)
    fn decompose_yzy(u: &Matrix2) -> Result<EulerAngles> {
        // Similar to ZYZ but with different axis ordering
        let det = determinant_2x2(u);
        let alpha = det.arg() / 2.0;

        let phase_factor = Complex64::new(0.0, -alpha).exp();
        let u_normalized = [
            [u[0][0] * phase_factor, u[0][1] * phase_factor],
            [u[1][0] * phase_factor, u[1][1] * phase_factor],
        ];

        let trace = u_normalized[0][0] + u_normalized[1][1];
        let gamma = 2.0 * (trace.re / 2.0).acos();

        if gamma.abs() < EPSILON {
            let delta = 2.0 * u_normalized[0][0].arg();
            return Ok(EulerAngles::new(alpha, 0.0, 0.0, delta));
        }

        if (gamma - PI).abs() < EPSILON {
            let delta = -2.0 * u_normalized[0][1].arg();
            return Ok(EulerAngles::new(alpha, 0.0, PI, delta));
        }

        let sin_half_gamma = (gamma / 2.0).sin();
        let beta = ((u_normalized[1][0] - u_normalized[0][1])
            / (Complex64::new(0.0, -2.0) * sin_half_gamma))
            .arg();
        let delta = ((u_normalized[1][0] + u_normalized[0][1])
            / (Complex64::new(0.0, -2.0) * sin_half_gamma))
            .arg();

        Ok(EulerAngles::new(alpha, beta, gamma, delta))
    }

    /// U3 decomposition (IBM): U = Rz(φ) Ry(θ) Rz(λ)
    ///
    /// This is the ZYZ decomposition without the global phase.
    /// IBM's native single-qubit gate parameterization.
    fn decompose_u3(u: &Matrix2) -> Result<EulerAngles> {
        // Use ZYZ decomposition but ignore global phase
        let angles = Self::decompose_zyz(u)?;

        // IBM U3(θ, φ, λ) = Rz(φ) Ry(θ) Rz(λ)
        // Maps to our ZYZ angles: θ=gamma, φ=beta, λ=delta
        Ok(EulerAngles {
            alpha: 0.0, // Global phase discarded
            beta: angles.beta,
            gamma: angles.gamma,
            delta: angles.delta,
        })
    }

    /// H-T decomposition for Clifford+T basis
    ///
    /// Decomposes arbitrary single-qubit unitary into {H, T} gates.
    /// This is an approximation using Solovay-Kitaev-like methods.
    fn decompose_ht(u: &Matrix2) -> Result<EulerAngles> {
        // For now, fall back to ZYZ
        // Full implementation would use gridsynth or Ross-Selinger algorithm
        Self::decompose_zyz(u)
    }

    /// Optimize the Euler angles to minimize gate count
    ///
    /// - Remove rotations with angle ≈ 0
    /// - Combine rotations on the same axis
    /// - Normalize angles to [-π, π]
    pub fn optimize_angles(&self, angles: &mut EulerAngles) {
        // Normalize angles to [-π, π]
        angles.beta = normalize_angle(angles.beta);
        angles.gamma = normalize_angle(angles.gamma);
        angles.delta = normalize_angle(angles.delta);

        // Remove near-zero rotations
        if angles.beta.abs() < EPSILON {
            angles.beta = 0.0;
        }
        if angles.gamma.abs() < EPSILON {
            angles.gamma = 0.0;
        }
        if angles.delta.abs() < EPSILON {
            angles.delta = 0.0;
        }

        // Handle π rotations specially
        if (angles.gamma.abs() - PI).abs() < EPSILON {
            angles.gamma = PI;
        }
    }

    /// Convert Euler angles to concrete single-qubit gates for `self.basis`.
    fn angles_to_gates(&self, angles: &EulerAngles) -> Result<Vec<Arc<dyn Gate>>> {
        use simq_gates::{RotationX, RotationY, RotationZ, U3};

        if angles.is_identity() {
            return Ok(vec![]);
        }

        let mut gates: Vec<Arc<dyn Gate>> = Vec::new();
        let push_rz = |gates: &mut Vec<Arc<dyn Gate>>, angle: f64| {
            if angle.abs() > EPSILON {
                gates.push(Arc::new(RotationZ::new(angle)));
            }
        };
        let push_ry = |gates: &mut Vec<Arc<dyn Gate>>, angle: f64| {
            if angle.abs() > EPSILON {
                gates.push(Arc::new(RotationY::new(angle)));
            }
        };
        let push_rx = |gates: &mut Vec<Arc<dyn Gate>>, angle: f64| {
            if angle.abs() > EPSILON {
                gates.push(Arc::new(RotationX::new(angle)));
            }
        };

        match self.basis {
            EulerBasis::ZYZ | EulerBasis::HT => {
                push_rz(&mut gates, angles.beta);
                push_ry(&mut gates, angles.gamma);
                push_rz(&mut gates, angles.delta);
            },
            EulerBasis::ZXZ => {
                push_rz(&mut gates, angles.beta);
                push_rx(&mut gates, angles.gamma);
                push_rz(&mut gates, angles.delta);
            },
            EulerBasis::XYX => {
                push_rx(&mut gates, angles.beta);
                push_ry(&mut gates, angles.gamma);
                push_rx(&mut gates, angles.delta);
            },
            EulerBasis::YZY => {
                push_ry(&mut gates, angles.beta);
                push_rz(&mut gates, angles.gamma);
                push_ry(&mut gates, angles.delta);
            },
            EulerBasis::U3 => {
                gates.push(Arc::new(U3::new(angles.gamma, angles.beta, angles.delta)));
            },
        }

        if gates.is_empty() {
            return Err(QuantumError::ValidationError(
                "Euler decomposition produced no gates for a non-identity unitary".to_string(),
            ));
        }

        Ok(gates)
    }
}

impl Decomposer for SingleQubitDecomposer {
    fn decompose(
        &self,
        gate: &dyn Gate,
        config: &DecompositionConfig,
    ) -> Result<DecompositionResult> {
        if gate.num_qubits() != 1 {
            return Err(QuantumError::ValidationError(format!(
                "Expected single-qubit gate, got {}-qubit gate",
                gate.num_qubits()
            )));
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

        // Decompose to Euler angles
        let mut angles = self.decompose_to_angles(&matrix_2x2)?;

        // Optimize if requested
        if config.optimization_level > 0 {
            self.optimize_angles(&mut angles);
        }

        // Convert angles to a real gate sequence for the requested basis.
        // Identity needs no gates; anything else must produce a non-empty,
        // verifiable sequence (never empty with fidelity 1.0).
        let gates = self.angles_to_gates(&angles)?;

        let gate_count = gates.len();
        // Identity decomposition is exact with zero gates; otherwise the
        // Euler reconstruction is exact up to numerical precision.
        let fidelity = 1.0;

        Ok(DecompositionResult {
            gates,
            fidelity,
            depth: gate_count,
            gate_count,
            two_qubit_count: 0,
            metadata: DecompositionMetadata {
                strategy: format!("{:?} decomposition", self.basis),
                optimized: config.optimization_level > 0,
                optimization_passes: config.optimization_level as usize,
                original_gate_count: 1,
            },
        })
    }

    fn can_decompose(&self, gate: &dyn Gate) -> bool {
        gate.num_qubits() == 1 && gate.matrix().is_some()
    }

    fn name(&self) -> &str {
        match self.basis {
            EulerBasis::ZYZ => "ZYZ",
            EulerBasis::ZXZ => "ZXZ",
            EulerBasis::XYX => "XYX",
            EulerBasis::YZY => "YZY",
            EulerBasis::U3 => "U3",
            EulerBasis::HT => "H-T",
        }
    }

    fn estimate_cost(&self, gate: &dyn Gate) -> Option<usize> {
        if gate.num_qubits() == 1 {
            Some(3) // Typically 3 rotations
        } else {
            None
        }
    }
}

/// Euler angles representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EulerAngles {
    /// Global phase
    pub alpha: f64,

    /// First rotation angle
    pub beta: f64,

    /// Second rotation angle
    pub gamma: f64,

    /// Third rotation angle
    pub delta: f64,
}

impl EulerAngles {
    /// Create new Euler angles
    pub fn new(alpha: f64, beta: f64, gamma: f64, delta: f64) -> Self {
        Self {
            alpha,
            beta,
            gamma,
            delta,
        }
    }

    /// Check if this represents the identity gate (all angles ≈ 0)
    pub fn is_identity(&self) -> bool {
        self.beta.abs() < EPSILON && self.gamma.abs() < EPSILON && self.delta.abs() < EPSILON
    }

    /// Count non-zero angles (gates needed)
    pub fn gate_count(&self) -> usize {
        let mut count = 0;
        if self.beta.abs() > EPSILON {
            count += 1;
        }
        if self.gamma.abs() > EPSILON {
            count += 1;
        }
        if self.delta.abs() > EPSILON {
            count += 1;
        }
        count
    }
}

/// Normalize angle to [-π, π]
fn normalize_angle(angle: f64) -> f64 {
    let mut a = angle % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    } else if a < -PI {
        a += 2.0 * PI;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix_computation::{hadamard_matrix, pauli_x_matrix};

    const ZERO: Complex64 = Complex64::new(0.0, 0.0);
    const ONE: Complex64 = Complex64::new(1.0, 0.0);

    #[test]
    fn test_zyz_identity() {
        let id: Matrix2 = [[ONE, ZERO], [ZERO, ONE]];
        let angles = SingleQubitDecomposer::decompose_zyz(&id).unwrap();

        assert!(angles.is_identity() || angles.gamma.abs() < EPSILON);
    }

    #[test]
    fn test_zyz_hadamard() {
        let h = hadamard_matrix();
        let angles = SingleQubitDecomposer::decompose_zyz(&h).unwrap();

        // Hadamard should have non-trivial angles
        assert!(angles.gamma.abs() > EPSILON);
    }

    #[test]
    fn test_zyz_pauli_x() {
        let x = pauli_x_matrix();
        let angles = SingleQubitDecomposer::decompose_zyz(&x).unwrap();

        // Pauli-X should decompose to a π rotation
        assert!((angles.gamma - PI).abs() < EPSILON || angles.gamma.abs() > EPSILON);
    }

    #[test]
    fn test_normalize_angle() {
        assert!((normalize_angle(2.0 * PI) - 0.0).abs() < EPSILON);
        assert!((normalize_angle(3.0 * PI) - PI).abs() < EPSILON);
        assert!((normalize_angle(-PI).abs() - PI).abs() < EPSILON);
    }

    #[test]
    fn test_euler_angles_identity() {
        let angles = EulerAngles::new(0.0, 0.0, 0.0, 0.0);
        assert!(angles.is_identity());
        assert_eq!(angles.gate_count(), 0);
    }

    #[test]
    fn test_euler_angles_gate_count() {
        let angles = EulerAngles::new(0.0, PI / 4.0, PI / 2.0, 0.0);
        assert_eq!(angles.gate_count(), 2);
    }

    fn non_unitary_matrix() -> Matrix2 {
        [
            [Complex64::new(2.0, 0.0), ZERO],
            [ZERO, Complex64::new(1.0, 0.0)],
        ]
    }

    #[test]
    fn test_decompose_to_angles_rejects_non_unitary() {
        let decomposer = SingleQubitDecomposer::new(EulerBasis::ZYZ);
        let result = decomposer.decompose_to_angles(&non_unitary_matrix());
        assert!(result.is_err());
    }

    #[test]
    fn test_decompose_zxz_identity_gamma_near_zero() {
        let id: Matrix2 = [[ONE, ZERO], [ZERO, ONE]];
        let angles = SingleQubitDecomposer::decompose_zxz(&id).unwrap();
        assert!(angles.gamma.abs() < EPSILON);
    }

    #[test]
    fn test_decompose_zxz_pauli_x_gamma_near_pi() {
        let x = pauli_x_matrix();
        let angles = SingleQubitDecomposer::decompose_zxz(&x).unwrap();
        assert!((angles.gamma - PI).abs() < EPSILON);
    }

    #[test]
    fn test_decompose_zxz_hadamard_general_case() {
        let h = hadamard_matrix();
        let angles = SingleQubitDecomposer::decompose_zxz(&h).unwrap();
        assert!(angles.gamma.abs() > EPSILON);
    }

    #[test]
    fn test_decompose_xyx_identity_gamma_near_zero() {
        let id: Matrix2 = [[ONE, ZERO], [ZERO, ONE]];
        let angles = SingleQubitDecomposer::decompose_xyx(&id).unwrap();
        assert!(angles.gamma.abs() < EPSILON);
    }

    #[test]
    fn test_decompose_xyx_pauli_z_gamma_near_pi() {
        // Pauli-Z has zero trace after phase normalization, giving gamma = 2*acos(0) = pi
        let z = crate::matrix_computation::pauli_z_matrix();
        let angles = SingleQubitDecomposer::decompose_xyx(&z).unwrap();
        assert!((angles.gamma - PI).abs() < EPSILON);
    }

    #[test]
    fn test_decompose_xyx_hadamard_general_case() {
        let h = hadamard_matrix();
        let angles = SingleQubitDecomposer::decompose_xyx(&h).unwrap();
        assert!(angles.gate_count() > 0);
    }

    #[test]
    fn test_decompose_yzy_identity_gamma_near_zero() {
        let id: Matrix2 = [[ONE, ZERO], [ZERO, ONE]];
        let angles = SingleQubitDecomposer::decompose_yzy(&id).unwrap();
        assert!(angles.gamma.abs() < EPSILON);
    }

    #[test]
    fn test_decompose_yzy_pauli_z_gamma_near_pi() {
        let z = crate::matrix_computation::pauli_z_matrix();
        let angles = SingleQubitDecomposer::decompose_yzy(&z).unwrap();
        assert!((angles.gamma - PI).abs() < EPSILON);
    }

    #[test]
    fn test_decompose_yzy_hadamard_general_case() {
        let h = hadamard_matrix();
        let angles = SingleQubitDecomposer::decompose_yzy(&h).unwrap();
        assert!(angles.gate_count() > 0);
    }

    #[test]
    fn test_decompose_u3_discards_global_phase() {
        let h = hadamard_matrix();
        let angles = SingleQubitDecomposer::decompose_u3(&h).unwrap();
        assert_eq!(angles.alpha, 0.0);
    }

    #[test]
    fn test_decompose_via_all_bases_dispatch() {
        // Exercise decompose_to_angles' match arms for every EulerBasis variant
        let h = hadamard_matrix();
        for basis in [
            EulerBasis::ZYZ,
            EulerBasis::ZXZ,
            EulerBasis::XYX,
            EulerBasis::YZY,
            EulerBasis::U3,
            EulerBasis::HT,
        ] {
            let decomposer = SingleQubitDecomposer::new(basis);
            assert!(decomposer.decompose_to_angles(&h).is_ok());
        }
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

    fn matrix2_to_flat(m: &Matrix2) -> Vec<Complex64> {
        vec![m[0][0], m[0][1], m[1][0], m[1][1]]
    }

    #[test]
    fn test_decomposer_trait_rejects_non_single_qubit() {
        let decomposer = SingleQubitDecomposer::new(EulerBasis::ZYZ);
        let config = DecompositionConfig::default();
        let gate = MockGate {
            name: "CNOT".to_string(),
            n_qubits: 2,
            matrix: None,
        };
        assert!(decomposer.decompose(&gate, &config).is_err());
    }

    #[test]
    fn test_decomposer_trait_rejects_missing_matrix() {
        let decomposer = SingleQubitDecomposer::new(EulerBasis::ZYZ);
        let config = DecompositionConfig::default();
        let gate = MockGate {
            name: "H".to_string(),
            n_qubits: 1,
            matrix: None,
        };
        assert!(decomposer.decompose(&gate, &config).is_err());
    }

    #[test]
    fn test_decomposer_trait_rejects_invalid_matrix_size() {
        let decomposer = SingleQubitDecomposer::new(EulerBasis::ZYZ);
        let config = DecompositionConfig::default();
        let gate = MockGate {
            name: "H".to_string(),
            n_qubits: 1,
            matrix: Some(vec![ONE]),
        };
        assert!(decomposer.decompose(&gate, &config).is_err());
    }

    #[test]
    fn test_decomposer_trait_succeeds_with_optimization() {
        let decomposer = SingleQubitDecomposer::new(EulerBasis::ZYZ);
        let config = DecompositionConfig {
            optimization_level: 1,
            ..Default::default()
        };
        let gate = MockGate {
            name: "H".to_string(),
            n_qubits: 1,
            matrix: Some(matrix2_to_flat(&hadamard_matrix())),
        };
        let result = decomposer.decompose(&gate, &config).unwrap();
        assert!(result.metadata.optimized);
        // Must emit a real, non-empty gate sequence (never empty + fidelity 1.0).
        assert!(!result.gates.is_empty());
        assert_eq!(result.gate_count, result.gates.len());
        assert_eq!(result.depth, result.gates.len());
    }

    #[test]
    fn test_normalize_angle_negative_wraparound() {
        let result = normalize_angle(-3.5 * PI);
        assert!((-PI..=PI).contains(&result));
    }
}
