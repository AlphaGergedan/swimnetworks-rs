//! The module [`crate::random_feature`] implements the random feature sampling method, which is
//! normal distribution for the weights and uniform distribution for the biases here.
//!
//! # Documentation
//!
//! * See [`RandomFeatureSampler`]
//!
//! # Example
//!
//! ```ignore
//! // sample the dense layer using random features
//! let sampler_config = RandomFeatureSamplerConfig {
//!     weight_sampler: random_feature::WeightSampler::Normal,
//!     bias_sampler: random_feature::BiasSampler::Uniform,
//! };
//! let sampler_config = SamplerConfig::RandomFeature(sampler_config);
//! let sampler = sampler_config.new();
//! sampler.sample(&mut model);
//! ```

mod config;
mod sampler;

pub use config::RandomFeatureSamplerConfig;
pub use sampler::RandomFeatureSampler;

/// Weight sampling distribution for the random feature sampling.
#[derive(Clone, Copy)]
pub enum WeightSampler {
    Normal,
}

/// Bias sampling distribution for the random feature sampling.
#[derive(Clone, Copy)]
pub enum BiasSampler {
    Uniform,
}
