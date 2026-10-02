//! Simulator configuration

/// Configuration for the quantum simulator
#[derive(Debug, Clone)]
pub struct SimulatorConfig {
    /// Enable GPU backend (wgpu)
    ///
    /// **Not implemented**: setting this to `true` fails
    /// [`validate`](Self::validate) (and thus `Simulator::new`) instead of
    /// silently executing on the CPU.
    ///
    /// Default: false
    pub use_gpu: bool,
    /// Density threshold for switching from sparse to dense representation
    ///
    /// When the state density (fraction of non-zero amplitudes) exceeds this
    /// threshold, the simulator automatically switches to dense representation.
    ///
    /// Default: 0.1 (10%)
    pub sparse_threshold: f64,

    /// Minimum number of qubits to enable parallel execution
    ///
    /// Circuits with fewer qubits use single-threaded execution to avoid
    /// synchronization overhead. Gate kernels are memory-bound, so rayon
    /// fork/join only pays for itself once the state vector is hundreds of
    /// KiB; below that, single-threaded cache-blocked kernels are faster.
    /// The kernel-level `MIN_PAR_BLOCK` guard (128 KiB tasks) keeps
    /// parallelism coarse regardless of this setting (issue #76).
    ///
    /// Default: 15 (2^15 amplitudes = 512 KiB)
    pub parallel_threshold: usize,

    /// Number of measurement shots for sampling
    ///
    /// When performing final measurements, this determines how many samples
    /// to draw from the probability distribution.
    ///
    /// Default: 1024
    pub shots: usize,

    /// Enable circuit compilation and optimization
    ///
    /// When true, circuits are optimized before execution using the compiler.
    ///
    /// Default: true
    pub optimize_circuit: bool,

    /// Optimization level (0-3)
    ///
    /// - O0: No optimization
    /// - O1: Basic optimizations (dead code elimination)
    /// - O2: Standard optimizations (fusion, commutation)
    /// - O3: Aggressive optimizations (all passes)
    ///
    /// Default: 2 (O2)
    pub optimization_level: u8,

    /// Enable execution statistics collection
    ///
    /// When true, collects detailed timing and resource usage statistics.
    ///
    /// Default: false
    pub collect_statistics: bool,

    /// Random number generator seed for reproducibility
    ///
    /// If None, uses a random seed. Set to Some(seed) for deterministic results.
    ///
    /// Default: None (random)
    pub seed: Option<u64>,

    /// Memory limit in bytes
    ///
    /// Maximum memory to use for state vectors. If exceeded, returns an error.
    /// Set to 0 for no limit.
    ///
    /// Default: 0 (unlimited)
    pub memory_limit: usize,

    /// Maximum number of distinct circuit *shapes* whose multi-qubit fusion
    /// block structure the simulator caches across repeated [`super::Simulator::run`]
    /// calls (0 disables this cache entirely). This is what lets a VQE/QAOA
    /// optimizer's outer loop — which calls `run` on the same-shaped
    /// circuit hundreds of times with only rotation angles differing —
    /// skip re-deriving fusion's block assignment on every call; the actual
    /// fused matrices are still recomputed fresh from each call's angles,
    /// so this never risks a stale-parameter bug (see
    /// `simq_compiler::fusion_cache`'s module docs).
    ///
    /// Default: 32 (a VQE/QAOA loop typically evaluates one circuit shape;
    /// this comfortably covers a handful without unbounded growth).
    pub fusion_cache_size: usize,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            sparse_threshold: 0.1,
            parallel_threshold: 15,
            shots: 1024,
            optimize_circuit: true,
            optimization_level: 2,
            collect_statistics: false,
            seed: None,
            memory_limit: 0,
            use_gpu: false,
            fusion_cache_size: 32,
        }
    }
}

impl SimulatorConfig {
    /// Create a new configuration with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration optimized for speed
    ///
    /// - Aggressive optimization (O3)
    /// - No statistics collection
    pub fn fast() -> Self {
        Self {
            optimization_level: 3,
            collect_statistics: false,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for accuracy
    ///
    /// - No optimization (to preserve exact circuit)
    /// - More measurement shots
    /// - Statistics collection enabled
    pub fn accurate() -> Self {
        Self {
            optimize_circuit: false,
            optimization_level: 0,
            shots: 10000,
            collect_statistics: true,
            ..Default::default()
        }
    }

    /// Create a configuration for debugging
    ///
    /// - No optimization
    /// - Statistics collection
    /// - Deterministic seed
    pub fn debug() -> Self {
        Self {
            optimize_circuit: false,
            optimization_level: 0,
            collect_statistics: true,
            seed: Some(42),
            ..Default::default()
        }
    }

    /// Set the sparse threshold
    pub fn with_sparse_threshold(mut self, threshold: f64) -> Self {
        self.sparse_threshold = threshold;
        self
    }

    /// Set the number of measurement shots
    pub fn with_shots(mut self, shots: usize) -> Self {
        self.shots = shots;
        self
    }

    /// Set the optimization level
    pub fn with_optimization_level(mut self, level: u8) -> Self {
        self.optimization_level = level.min(3);
        self
    }

    /// Enable or disable circuit optimization
    pub fn with_optimization(mut self, enabled: bool) -> Self {
        self.optimize_circuit = enabled;
        self
    }

    /// Set the random seed for deterministic execution
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enable statistics collection
    pub fn with_statistics(mut self, enabled: bool) -> Self {
        self.collect_statistics = enabled;
        self
    }

    /// Set memory limit in bytes
    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = limit;
        self
    }

    /// Set the fusion structure cache size (0 disables it)
    pub fn with_fusion_cache_size(mut self, size: usize) -> Self {
        self.fusion_cache_size = size;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.use_gpu {
            return Err(
                "use_gpu = true, but GPU acceleration is not implemented: SimQ would silently \
                 execute on the CPU. Remove the flag until a GPU backend exists."
                    .to_string(),
            );
        }

        if self.sparse_threshold < 0.0 || self.sparse_threshold > 1.0 {
            return Err(format!(
                "sparse_threshold must be in [0,1], got {}",
                self.sparse_threshold
            ));
        }

        if self.shots == 0 {
            return Err("shots must be > 0".to_string());
        }

        if self.optimization_level > 3 {
            return Err(format!("optimization_level must be 0-3, got {}", self.optimization_level));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SimulatorConfig::default();
        assert_eq!(config.sparse_threshold, 0.1);
        assert_eq!(config.parallel_threshold, 15);
        assert_eq!(config.shots, 1024);
        assert!(config.optimize_circuit);
        assert_eq!(config.optimization_level, 2);
    }

    #[test]
    fn test_fast_config() {
        let config = SimulatorConfig::fast();
        assert_eq!(config.optimization_level, 3);
        assert!(!config.collect_statistics);
        assert_eq!(config.parallel_threshold, 15);
    }

    #[test]
    fn test_accurate_config() {
        let config = SimulatorConfig::accurate();
        assert!(!config.optimize_circuit);
        assert_eq!(config.shots, 10000);
        assert!(config.collect_statistics);
    }

    #[test]
    fn test_builder_pattern() {
        let config = SimulatorConfig::new()
            .with_shots(2048)
            .with_optimization_level(3)
            .with_seed(42);

        assert_eq!(config.shots, 2048);
        assert_eq!(config.optimization_level, 3);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_validate() {
        let config = SimulatorConfig::default();
        assert!(config.validate().is_ok());

        let invalid = SimulatorConfig {
            sparse_threshold: 1.5,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        let invalid = SimulatorConfig {
            shots: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_unimplemented_gpu() {
        let invalid = SimulatorConfig {
            use_gpu: true,
            ..Default::default()
        };
        let err = invalid.validate().unwrap_err();
        assert!(err.contains("not implemented"), "got: {}", err);
    }

    #[test]
    fn test_validate_optimization_level_too_high() {
        // with_optimization_level clamps to 3, so construct directly to
        // exercise the validate() branch for optimization_level > 3.
        let invalid = SimulatorConfig {
            optimization_level: 4,
            ..Default::default()
        };
        let err = invalid.validate().unwrap_err();
        assert!(err.contains("optimization_level must be 0-3"));
        assert!(err.contains('4'));
    }
}
