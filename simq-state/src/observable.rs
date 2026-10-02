//! Pauli observable implementation for expectation value computation
//!
//! This module provides efficient computation of expectation values ⟨ψ|O|ψ⟩
//! for Pauli observables without state collapse. This is critical for
//! variational quantum algorithms like VQE and QAOA.
//!
//! # Pauli Operators
//!
//! The four single-qubit Pauli operators:
//! - I: Identity [[1,0],[0,1]]
//! - X: Bit flip [[0,1],[1,0]]
//! - Y: Phase flip [[0,-i],[i,0]]
//! - Z: Phase flip [[1,0],[0,-1]]
//!
//! # Pauli Strings
//!
//! A Pauli string is a tensor product of single-qubit Paulis, e.g., "IXYZ"
//! represents I⊗X⊗Y⊗Z acting on 4 qubits.

use crate::dense_state::DenseState;
use crate::error::{Result, StateError};
use num_complex::Complex64;
use std::fmt;

/// Single-qubit Pauli operator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pauli {
    /// Identity operator
    I,
    /// Pauli X (bit flip)
    X,
    /// Pauli Y (phase flip with i)
    Y,
    /// Pauli Z (phase flip)
    Z,
}

impl Pauli {
    /// Parse a Pauli operator from a character
    pub fn from_char(c: char) -> Result<Self> {
        match c.to_ascii_uppercase() {
            'I' => Ok(Pauli::I),
            'X' => Ok(Pauli::X),
            'Y' => Ok(Pauli::Y),
            'Z' => Ok(Pauli::Z),
            _ => Err(StateError::InvalidDimension { dimension: 0 }),
        }
    }

    /// Convert to character representation
    pub fn to_char(self) -> char {
        match self {
            Pauli::I => 'I',
            Pauli::X => 'X',
            Pauli::Y => 'Y',
            Pauli::Z => 'Z',
        }
    }

    /// Check if this Pauli is diagonal (I or Z)
    pub fn is_diagonal(self) -> bool {
        matches!(self, Pauli::I | Pauli::Z)
    }

    /// Get the eigenvalue for a computational basis state
    /// Returns (+1, -1) for diagonal operators
    pub fn eigenvalue(self, basis_state: bool) -> f64 {
        match self {
            Pauli::I => 1.0,
            Pauli::Z => {
                if basis_state {
                    -1.0
                } else {
                    1.0
                }
            },
            _ => panic!("eigenvalue only valid for diagonal Paulis"),
        }
    }
}

impl fmt::Display for Pauli {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

/// A tensor product of Pauli operators (Pauli string)
///
/// Represents observables like X⊗X⊗I⊗Z (written as "XXIZ").
///
/// # Qubit ordering
///
/// The string reads **left-to-right starting at qubit 0**: character `i`
/// acts on qubit `i`. So `"XZ"` means X on qubit 0 and Z on qubit 1:
///
/// ```
/// use simq_state::{DenseState, PauliString};
/// use num_complex::Complex64;
///
/// // |+0⟩ = (|00⟩ + |10⟩)/√2 — qubit 0 in |+⟩, qubit 1 in |0⟩.
/// let s = std::f64::consts::FRAC_1_SQRT_2;
/// let state = DenseState::from_amplitudes(2, &[
///     Complex64::new(s, 0.0),
///     Complex64::new(s, 0.0),
///     Complex64::new(0.0, 0.0),
///     Complex64::new(0.0, 0.0),
/// ])
/// .unwrap();
/// // X flips |+⟩ to itself, Z leaves |0⟩ alone: ⟨XZ⟩ = +1.
/// let xz = PauliString::from_str("XZ").unwrap();
/// assert!((xz.expectation_value(&state).unwrap() - 1.0).abs() < 1e-12);
/// ```
///
/// # Coming from Qiskit?
///
/// Qiskit reads Pauli strings **right-to-left**: its qubit 0 is the
/// *rightmost* character. The same operator is written mirrored:
///
/// | Operator | SimQ | Qiskit `SparsePauliOp` |
/// |---|---|---|
/// | X on qubit 0, Z on qubit 1 | `"XZ"` | `"ZX"` |
/// | Z on qubit 0, Z on qubit 1 | `"ZZ"` | `"ZZ"` (symmetric — same either way) |
///
/// Porting a Hamiltonian character-by-character without mirroring silently
/// builds the reversed operator on asymmetric strings. Nothing errors, so
/// double-check any string that is not a palindrome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PauliString {
    /// Pauli operators for each qubit
    paulis: Vec<Pauli>,

    /// Overall coefficient
    coeff: i32, // Either +1 or -1 for simplicity

    /// Phase factor as multiple of π/2 (0, 1, 2, 3 for 1, i, -1, -i)
    phase: u8,
}

impl PauliString {
    /// Create a new Pauli string from a string representation
    ///
    /// # Example
    /// ```
    /// use simq_state::PauliString;
    ///
    /// let pauli = PauliString::from_str("XXYZ").unwrap();
    /// assert_eq!(pauli.num_qubits(), 4);
    /// ```
    /// Create a new Pauli string from a string representation
    ///
    /// # Example
    /// ```
    /// use simq_state::PauliString;
    /// use std::str::FromStr;
    ///
    /// let pauli = PauliString::from_str("XXYZ").unwrap();
    /// assert_eq!(pauli.num_qubits(), 4);
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self> {
        s.parse()
    }
}

impl std::str::FromStr for PauliString {
    type Err = StateError;

    fn from_str(s: &str) -> Result<Self> {
        let paulis: Result<Vec<_>> = s.chars().map(Pauli::from_char).collect();
        Ok(Self {
            paulis: paulis?,
            coeff: 1,
            phase: 0,
        })
    }
}

impl PauliString {
    /// Create a Pauli string from a vector of Paulis
    pub fn from_paulis(paulis: Vec<Pauli>) -> Self {
        Self {
            paulis,
            coeff: 1,
            phase: 0,
        }
    }

    /// Create an all-Z Pauli string for a given number of qubits
    pub fn all_z(num_qubits: usize) -> Self {
        Self {
            paulis: vec![Pauli::Z; num_qubits],
            coeff: 1,
            phase: 0,
        }
    }

    /// Create an all-I (identity) Pauli string
    pub fn identity(num_qubits: usize) -> Self {
        Self {
            paulis: vec![Pauli::I; num_qubits],
            coeff: 1,
            phase: 0,
        }
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.paulis.len()
    }

    /// Get the Pauli operator at a specific qubit
    pub fn get(&self, qubit: usize) -> Option<Pauli> {
        self.paulis.get(qubit).copied()
    }

    /// Set coefficient
    pub fn with_coeff(mut self, coeff: i32) -> Self {
        self.coeff = coeff;
        self
    }

    /// Get the overall sign coefficient (+1 or -1)
    pub fn coeff(&self) -> i32 {
        self.coeff
    }

    /// Check if this Pauli string is diagonal (all I or Z)
    pub fn is_diagonal(&self) -> bool {
        self.paulis.iter().all(|p| p.is_diagonal())
    }

    /// Compute expectation value ⟨ψ|P|ψ⟩ for this Pauli string
    ///
    /// # Arguments
    /// * `state` - The quantum state
    ///
    /// # Returns
    /// The expectation value (real number)
    pub fn expectation_value(&self, state: &DenseState) -> Result<f64> {
        if self.num_qubits() != state.num_qubits() {
            return Err(StateError::DimensionMismatch {
                expected: self.num_qubits(),
                actual: state.num_qubits(),
            });
        }

        if self.is_diagonal() {
            // Fast path for diagonal operators
            self.expectation_value_diagonal(state)
        } else {
            // General case: apply Pauli string and compute ⟨ψ|P|ψ⟩
            self.expectation_value_general(state)
        }
    }

    /// Bit masks of the qubits carrying X, Y, and Z operators.
    ///
    /// Qubit q corresponds to bit q (little-endian), matching the state
    /// indexing convention.
    fn masks(&self) -> (usize, usize, usize) {
        let (mut x_mask, mut y_mask, mut z_mask) = (0usize, 0usize, 0usize);
        for (qubit, &pauli) in self.paulis.iter().enumerate() {
            match pauli {
                Pauli::I => {},
                Pauli::X => x_mask |= 1 << qubit,
                Pauli::Y => y_mask |= 1 << qubit,
                Pauli::Z => z_mask |= 1 << qubit,
            }
        }
        (x_mask, y_mask, z_mask)
    }

    /// Compute expectation value for diagonal Pauli string (fast path)
    ///
    /// The eigenvalue of a Z-string on basis state |i⟩ is
    /// (-1)^popcount(i & z_mask) — one popcount per amplitude instead of a
    /// loop over all qubits.
    fn expectation_value_diagonal(&self, state: &DenseState) -> Result<f64> {
        let (_, _, z_mask) = self.masks();

        let mut expectation = 0.0;
        for (basis_state, amplitude) in state.amplitudes().iter().enumerate() {
            let probability = amplitude.norm_sqr();
            if (basis_state & z_mask).count_ones() & 1 == 1 {
                expectation -= probability;
            } else {
                expectation += probability;
            }
        }

        Ok(expectation * self.coeff as f64)
    }

    /// Compute expectation value for general Pauli string
    ///
    /// Evaluates ⟨ψ|P|ψ⟩ = Σ_i conj(ψ[i ^ flip]) · ψ[i] · phase(i) in a single
    /// allocation-free pass: `flip` is the X|Y mask, and
    /// phase(i) = i^{|Y|} · (-1)^popcount(i & (Y|Z)). (The previous
    /// implementation allocated a full 2^n scratch vector and walked the
    /// whole Pauli vector per amplitude — it dominated VQE energy
    /// evaluations at 16 qubits.)
    fn expectation_value_general(&self, state: &DenseState) -> Result<f64> {
        let amplitudes = state.amplitudes();
        let (x_mask, y_mask, z_mask) = self.masks();
        let flip = x_mask | y_mask;
        let sign_mask = y_mask | z_mask;

        let mut acc = Complex64::new(0.0, 0.0);
        for (i, &a) in amplitudes.iter().enumerate() {
            let partner = amplitudes[i ^ flip].conj() * a;
            if (i & sign_mask).count_ones() & 1 == 1 {
                acc -= partner;
            } else {
                acc += partner;
            }
        }

        // Global i^{|Y|} factor
        let y_phase = match y_mask.count_ones() % 4 {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            _ => Complex64::new(0.0, -1.0),
        };

        Ok((acc * y_phase).re * self.coeff as f64)
    }

    /// Apply Pauli string to a computational basis state
    ///
    /// Returns (new_state, phase_factor)
    /// Reference implementation kept for the unit tests; the hot paths above
    /// use the mask/popcount form instead.
    #[cfg_attr(not(test), allow(dead_code))]
    fn apply_to_basis_state(&self, basis_state: usize) -> (usize, Complex64) {
        let mut new_state = basis_state;
        let mut phase = Complex64::new(1.0, 0.0);

        for (qubit, &pauli) in self.paulis.iter().enumerate() {
            let bit = (basis_state >> qubit) & 1;

            match pauli {
                Pauli::I => {
                    // Identity: no change
                },
                Pauli::X => {
                    // Flip bit
                    new_state ^= 1 << qubit;
                },
                Pauli::Y => {
                    // Flip bit with phase
                    new_state ^= 1 << qubit;
                    // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
                    phase *= if bit == 0 {
                        Complex64::new(0.0, 1.0) // i
                    } else {
                        Complex64::new(0.0, -1.0) // -i
                    };
                },
                Pauli::Z => {
                    // Phase flip: |0⟩ → |0⟩, |1⟩ → -|1⟩
                    if bit == 1 {
                        phase *= Complex64::new(-1.0, 0.0);
                    }
                },
            }
        }

        (new_state, phase)
    }
}

impl fmt::Display for PauliString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.coeff == -1 {
            write!(f, "-")?;
        }
        for pauli in &self.paulis {
            write!(f, "{}", pauli)?;
        }
        Ok(())
    }
}

/// A weighted sum of Pauli strings (Pauli observable)
///
/// Represents observables like 0.5*X⊗X + 0.3*Z⊗Z.
///
/// Each term follows [`PauliString`]'s qubit ordering: string character `i`
/// acts on qubit `i` (left-to-right from qubit 0 — the mirror image of
/// Qiskit's right-to-left convention).
#[derive(Debug, Clone)]
pub struct PauliObservable {
    /// Terms in the observable (Pauli string, coefficient)
    terms: Vec<(PauliString, f64)>,
}

impl PauliObservable {
    /// Create a new empty observable
    pub fn new() -> Self {
        Self { terms: Vec::new() }
    }

    /// Create an observable from a single Pauli string
    pub fn from_pauli_string(pauli: PauliString, coeff: f64) -> Self {
        Self {
            terms: vec![(pauli, coeff)],
        }
    }

    /// Add a term to the observable
    pub fn add_term(&mut self, pauli: PauliString, coeff: f64) {
        self.terms.push((pauli, coeff));
    }

    /// Create a Z observable for a single qubit
    ///
    /// Measures spin in Z direction for qubit at position `qubit`
    pub fn single_z(num_qubits: usize, qubit: usize) -> Self {
        let mut paulis = vec![Pauli::I; num_qubits];
        paulis[qubit] = Pauli::Z;

        Self::from_pauli_string(PauliString::from_paulis(paulis), 1.0)
    }

    /// Get the number of terms
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Get the (Pauli string, coefficient) terms making up this observable.
    ///
    /// Exposed for callers (e.g. `simq_sim::pauli_propagation`) that need to
    /// walk the term list directly instead of going through
    /// [`Self::expectation_value`]'s statevector-only path.
    pub fn terms(&self) -> &[(PauliString, f64)] {
        &self.terms
    }

    /// Compute expectation value ⟨ψ|O|ψ⟩
    ///
    /// # Arguments
    /// * `state` - The quantum state
    ///
    /// # Returns
    /// The expectation value (real number)
    pub fn expectation_value(&self, state: &DenseState) -> Result<f64> {
        let mut total = 0.0;

        for (pauli, coeff) in &self.terms {
            let term_expectation = pauli.expectation_value(state)?;
            total += coeff * term_expectation;
        }

        Ok(total)
    }

    /// Check if all terms are diagonal
    pub fn is_diagonal(&self) -> bool {
        self.terms.iter().all(|(p, _)| p.is_diagonal())
    }
}

impl Default for PauliObservable {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PauliObservable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, (pauli, coeff)) in self.terms.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            write!(f, "{:.4}·{}", coeff, pauli)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pauli_from_char() {
        assert_eq!(Pauli::from_char('I').unwrap(), Pauli::I);
        assert_eq!(Pauli::from_char('X').unwrap(), Pauli::X);
        assert_eq!(Pauli::from_char('Y').unwrap(), Pauli::Y);
        assert_eq!(Pauli::from_char('Z').unwrap(), Pauli::Z);
        assert_eq!(Pauli::from_char('x').unwrap(), Pauli::X); // Case insensitive
    }

    #[test]
    fn test_pauli_is_diagonal() {
        assert!(Pauli::I.is_diagonal());
        assert!(!Pauli::X.is_diagonal());
        assert!(!Pauli::Y.is_diagonal());
        assert!(Pauli::Z.is_diagonal());
    }

    #[test]
    fn test_pauli_string_from_str() {
        let pauli = PauliString::from_str("IXYZ").unwrap();
        assert_eq!(pauli.num_qubits(), 4);
        assert_eq!(pauli.get(0), Some(Pauli::I));
        assert_eq!(pauli.get(1), Some(Pauli::X));
        assert_eq!(pauli.get(2), Some(Pauli::Y));
        assert_eq!(pauli.get(3), Some(Pauli::Z));
    }

    #[test]
    fn test_pauli_string_is_diagonal() {
        assert!(PauliString::from_str("IIZZ").unwrap().is_diagonal());
        assert!(!PauliString::from_str("IXYZ").unwrap().is_diagonal());
        assert!(!PauliString::from_str("XIIZ").unwrap().is_diagonal());
    }

    #[test]
    fn test_expectation_value_z_basis_state() {
        // |0⟩ state
        let state = DenseState::new(1).unwrap();

        // Z observable: expect +1 for |0⟩
        let z_obs = PauliString::from_str("Z").unwrap();
        let expectation = z_obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 1.0, epsilon = 1e-10);

        // X observable: expect 0 for |0⟩
        let x_obs = PauliString::from_str("X").unwrap();
        let expectation = x_obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expectation_value_x_basis_state() {
        // |+⟩ = (|0⟩ + |1⟩)/√2 state
        let amplitudes = vec![
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
        ];
        let state = DenseState::from_amplitudes(1, &amplitudes).unwrap();

        // X observable: expect +1 for |+⟩
        let x_obs = PauliString::from_str("X").unwrap();
        let expectation = x_obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 1.0, epsilon = 1e-10);

        // Z observable: expect 0 for |+⟩
        let z_obs = PauliString::from_str("Z").unwrap();
        let expectation = z_obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expectation_value_bell_state() {
        // Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        let amplitudes = vec![
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
        ];
        let state = DenseState::from_amplitudes(2, &amplitudes).unwrap();

        // ZZ observable: expect +1 (both qubits have same Z eigenvalue)
        let zz_obs = PauliString::from_str("ZZ").unwrap();
        let expectation = zz_obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 1.0, epsilon = 1e-10);

        // XX observable: expect +1 for Bell state
        let xx_obs = PauliString::from_str("XX").unwrap();
        let expectation = xx_obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 1.0, epsilon = 1e-10);

        // ZI observable: expect 0 (equal superposition of Z eigenvalues)
        let zi_obs = PauliString::from_str("ZI").unwrap();
        let expectation = zi_obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pauli_observable_single_term() {
        let state = DenseState::new(1).unwrap();

        let mut obs = PauliObservable::new();
        obs.add_term(PauliString::from_str("Z").unwrap(), 1.0);

        let expectation = obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pauli_observable_multiple_terms() {
        // |+⟩ state
        let amplitudes = vec![
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
        ];
        let state = DenseState::from_amplitudes(1, &amplitudes).unwrap();

        // Observable: 0.5*X + 0.3*Z
        let mut obs = PauliObservable::new();
        obs.add_term(PauliString::from_str("X").unwrap(), 0.5);
        obs.add_term(PauliString::from_str("Z").unwrap(), 0.3);

        // Expect: 0.5*1.0 + 0.3*0.0 = 0.5
        let expectation = obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_pauli_y_operator() {
        // |0⟩ state
        let state = DenseState::new(1).unwrap();

        // Y|0⟩ = i|1⟩, so ⟨0|Y|0⟩ = 0
        let y_obs = PauliString::from_str("Y").unwrap();
        let expectation = y_obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 0.0, epsilon = 1e-10);

        // |+i⟩ = (|0⟩ + i|1⟩)/√2 state (eigenstate of Y with +1)
        let amplitudes = vec![
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
            Complex64::new(0.0, 1.0 / 2_f64.sqrt()),
        ];
        let plus_i_state = DenseState::from_amplitudes(1, &amplitudes).unwrap();

        let expectation = y_obs.expectation_value(&plus_i_state).unwrap();
        assert_relative_eq!(expectation, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_to_basis_state() {
        let pauli_x = PauliString::from_str("X").unwrap();

        // X|0⟩ = |1⟩
        let (new_state, phase) = pauli_x.apply_to_basis_state(0);
        assert_eq!(new_state, 1);
        assert_relative_eq!(phase.re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(phase.im, 0.0, epsilon = 1e-10);

        // X|1⟩ = |0⟩
        let (new_state, phase) = pauli_x.apply_to_basis_state(1);
        assert_eq!(new_state, 0);
        assert_relative_eq!(phase.re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_single_z_observable() {
        let state = DenseState::new(3).unwrap();

        // Measure Z on qubit 1 (should be +1 for |000⟩)
        let obs = PauliObservable::single_z(3, 1);
        let expectation = obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pauli_from_char_invalid() {
        // An unrecognized character must return an error rather than panic.
        let err = Pauli::from_char('Q').unwrap_err();
        assert_eq!(err, StateError::InvalidDimension { dimension: 0 });
    }

    #[test]
    fn test_pauli_eigenvalue_identity_and_display() {
        // Pauli::I eigenvalue is always +1 regardless of basis state.
        assert_relative_eq!(Pauli::I.eigenvalue(false), 1.0, epsilon = 1e-10);
        assert_relative_eq!(Pauli::I.eigenvalue(true), 1.0, epsilon = 1e-10);

        // Display impl for Pauli should match `to_char`.
        assert_eq!(Pauli::I.to_string(), "I");
        assert_eq!(Pauli::X.to_string(), "X");
        assert_eq!(Pauli::Y.to_string(), "Y");
        assert_eq!(Pauli::Z.to_string(), "Z");
    }

    #[test]
    #[should_panic(expected = "eigenvalue only valid for diagonal Paulis")]
    fn test_pauli_eigenvalue_panics_for_non_diagonal() {
        // X and Y are not diagonal, so `eigenvalue` must panic for them.
        let _ = Pauli::X.eigenvalue(false);
    }

    #[test]
    fn test_expectation_value_dimension_mismatch() {
        // Pauli string qubit count must match the state's qubit count.
        let state = DenseState::new(2).unwrap();
        let obs = PauliString::from_str("ZZZ").unwrap();

        let err = obs.expectation_value(&state).unwrap_err();
        assert_eq!(
            err,
            StateError::DimensionMismatch {
                expected: 3,
                actual: 2,
            }
        );
    }

    #[test]
    fn test_pauli_string_display_negative_coeff() {
        // Display must prefix a '-' when coeff == -1, and concatenate the
        // per-qubit Pauli characters.
        let pauli = PauliString::from_paulis(vec![Pauli::X, Pauli::Y, Pauli::Z]).with_coeff(-1);
        assert_eq!(pauli.to_string(), "-XYZ");

        let positive = PauliString::from_paulis(vec![Pauli::I, Pauli::Z]);
        assert_eq!(positive.to_string(), "IZ");
    }

    #[test]
    fn test_pauli_observable_terms_accessor() {
        let mut obs = PauliObservable::new();
        obs.add_term(PauliString::from_str("X").unwrap(), 0.5);
        obs.add_term(PauliString::from_str("Z").unwrap(), 0.25);

        let terms = obs.terms();
        assert_eq!(terms.len(), 2);
        assert_eq!(terms[0].0, PauliString::from_str("X").unwrap());
        assert_eq!(terms[0].1, 0.5);
        assert_eq!(terms[1].1, 0.25);
    }

    #[test]
    fn test_pauli_observable_default_and_display() {
        // Default::default() should produce an empty observable.
        let obs = PauliObservable::default();
        assert_eq!(obs.num_terms(), 0);
        assert_eq!(obs.to_string(), "");

        // Display with multiple terms joins them with " + " and formats the
        // coefficient with 4 decimal places.
        let mut obs = PauliObservable::new();
        obs.add_term(PauliString::from_str("X").unwrap(), 0.5);
        obs.add_term(PauliString::from_str("Z").unwrap(), 0.25);
        assert_eq!(obs.to_string(), "0.5000·X + 0.2500·Z");
    }

    #[test]
    fn test_apply_to_basis_state_z_in_general_path() {
        // A mixed string like "ZX" is not diagonal, so expectation_value
        // routes through expectation_value_general, which in turn exercises
        // the Pauli::Z branch of apply_to_basis_state (phase flip on bit==1).
        // Paulis are indexed left-to-right as qubit0, qubit1, ...: "ZX" means
        // qubit0=Z, qubit1=X.
        let zx = PauliString::from_str("ZX").unwrap();
        assert!(!zx.is_diagonal());

        // basis_state = 0b01 -> qubit0 (Z) bit=1 => phase -1, qubit1 (X) bit=0 => flips qubit1
        let (new_state, phase) = zx.apply_to_basis_state(0b01);
        assert_eq!(new_state, 0b11); // X flips qubit 1 bit
        assert_relative_eq!(phase.re, -1.0, epsilon = 1e-10);
        assert_relative_eq!(phase.im, 0.0, epsilon = 1e-10);

        // basis_state = 0b10 -> qubit0 (Z) bit=0 => no phase, qubit1 (X) bit=1 => flips qubit1
        let (new_state, phase) = zx.apply_to_basis_state(0b10);
        assert_eq!(new_state, 0b00); // X flips qubit 1 bit back to 0
        assert_relative_eq!(phase.re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(phase.im, 0.0, epsilon = 1e-10);

        // Full expectation value computation on a basis state for a mixed
        // string, ensuring expectation_value_general runs end-to-end.
        let state = DenseState::new(2).unwrap();
        let expectation = zx.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_to_basis_state_identity_and_y() {
        // "IY" exercises the Pauli::I branch (no-op) and both arms of the
        // Pauli::Y branch (phase +i when bit==0, phase -i when bit==1) of
        // `apply_to_basis_state`. Paulis are indexed left-to-right as
        // qubit0, qubit1, ...: "IY" means qubit0=I, qubit1=Y.
        let iy = PauliString::from_str("IY").unwrap();

        // basis_state = 0b00 -> qubit0 (I) unchanged, qubit1 (Y) bit=0 => flip + i phase
        let (new_state, phase) = iy.apply_to_basis_state(0b00);
        assert_eq!(new_state, 0b10);
        assert_relative_eq!(phase.re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(phase.im, 1.0, epsilon = 1e-10);

        // basis_state = 0b10 -> qubit0 (I) unchanged, qubit1 (Y) bit=1 => flip - i phase
        let (new_state, phase) = iy.apply_to_basis_state(0b10);
        assert_eq!(new_state, 0b00);
        assert_relative_eq!(phase.re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(phase.im, -1.0, epsilon = 1e-10);

        // basis_state = 0b01 -> qubit0 (I) unchanged (bit stays set), qubit1
        // (Y) bit=0 => flip + i phase.
        let (new_state, phase) = iy.apply_to_basis_state(0b01);
        assert_eq!(new_state, 0b11);
        assert_relative_eq!(phase.re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(phase.im, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expectation_value_general_three_y_phase() {
        // With exactly 3 Y operators, y_mask.count_ones() % 4 == 3, which
        // exercises the final `_` arm of the global i^|Y| phase match in
        // `expectation_value_general` (phase = -i). |000⟩ has amplitude 1 on
        // basis state 0, and YYY maps |000⟩ -> i^3 |111⟩ = -i|111⟩, which is
        // orthogonal to |000⟩, so the expectation value is 0 but the
        // -i branch is still exercised on the way there.
        let yyy = PauliString::from_str("YYY").unwrap();
        assert!(!yyy.is_diagonal());

        let state = DenseState::new(3).unwrap();
        let expectation = yyy.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 0.0, epsilon = 1e-10);

        // Directly exercise apply_to_basis_state for the 3-Y case too: all
        // three bits start at 0, so each Y contributes a +i phase and flips
        // its qubit, giving overall phase i^3 = -i and new_state = 0b111.
        let (new_state, phase) = yyy.apply_to_basis_state(0b000);
        assert_eq!(new_state, 0b111);
        assert_relative_eq!(phase.re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(phase.im, -1.0, epsilon = 1e-10);

        // Build a state where ⟨YYY⟩ is nonzero: ψ = (|000⟩ + i|111⟩)/sqrt(2).
        // Since YYY|000⟩ = -i|111⟩ and YYY|111⟩ = i|000⟩ (derived from the
        // per-qubit phase i on a 0-bit and -i on a 1-bit, cubed), YYYψ =
        // (-1/sqrt2)|000⟩ + (-i/sqrt2)|111⟩, giving ⟨ψ|YYY|ψ⟩ = -1.
        let amplitudes = vec![
            Complex64::new(1.0 / std::f64::consts::SQRT_2, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 1.0 / std::f64::consts::SQRT_2),
        ];
        let ghz = DenseState::from_amplitudes(3, &amplitudes).unwrap();
        let expectation = yyy.expectation_value(&ghz).unwrap();
        assert_relative_eq!(expectation, -1.0, epsilon = 1e-10);
    }
}
