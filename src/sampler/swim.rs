//! The module [`crate::swim`] implements the sampling method that utilizes the [SWIM][associated_paper] algorithm.
//!
//! # Documentation
//!
//! * See [`SWIMSampler`]
//!
//! # Example
//!
//! ```ignore
//! // Sample the dense layer using the swim algorithm
//! let sampler_config = SWIMSamplerConfig {
//!     inputs: inputs.view(),
//!     outputs: truths.view(),
//!     param_sampler: swim::ParamSampler::Relu,
//!     input_sampler_prob_dist: swim::InputSamplerProbDist::SWIM(swim::InputNorm::L2, swim::OutputNorm::Max),
//! };
//! let sampler_config = SamplerConfig::SWIM(sampler_config);
//! let sampler = sampler_config.new();
//! sampler.sample(&mut model);
//! ```
//!
//! [associated_paper]: https://arxiv.org/abs/2306.16830

mod config;
mod sampler;

pub use config::SWIMSamplerConfig;
pub use sampler::SWIMSampler;

/// Parameter sampler used in the SWIM algorithm.
#[derive(Clone, Copy)]
pub enum ParamSampler {
    Relu, Tanh
}

/// Probability distribution used for input data sampling from the given inputs for the SWIM
/// algorithm.
#[derive(Clone, Copy)]
pub enum InputSamplerProbDist {
    Uniform,
    SWIM(InputNorm, OutputNorm),
}

/// norms over the input space coming from the previous layer
/// used in swim data sampler when sampling the data points on the divident
#[derive(Clone, Copy)]
pub enum InputNorm {
    L2,
}

/// norms over the function values
/// used in the data sampler when sampling the data points
#[derive(Clone, Copy)]
pub enum OutputNorm {
    Max,
}
