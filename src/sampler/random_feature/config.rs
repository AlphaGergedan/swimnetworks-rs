use crate::random_feature::{WeightSampler, BiasSampler, RandomFeatureSampler};

/// Configuration for [`RandomFeatureSampler`]
#[derive(Clone, Copy)]
pub struct RandomFeatureSamplerConfig {
    /// Weight sampling method used in random feature sampling
    pub weight_sampler: WeightSampler,
    /// Bias sampling method used in random feature sampling
    pub bias_sampler: BiasSampler,
}

impl RandomFeatureSamplerConfig {
    /// Creates a new [`RandomFeatureSampler`] given configuration. But directly using this
    /// function is discouraged. See [`crate::SamplerConfig::new`] instead.
    /// [`crate::SamplerConfig`] wraps a specific sampler config.
    pub fn new(self) -> RandomFeatureSampler {
        RandomFeatureSampler::new(self)
    }
}
