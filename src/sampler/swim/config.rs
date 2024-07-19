use ndarray::ArrayView2;
use crate::swim::{InputSamplerProbDist, ParamSampler, SWIMSampler};

/// Configuration for [`SWIMSampler`]
#[derive(Clone, Copy)]
pub struct SWIMSamplerConfig<'a> {
    /// Inputs to the model. Required for sampling using the SWIM algorithm.
    pub inputs: ArrayView2<'a, f64>,
    /// True function values of the inputs to the model (supervised learning). Required for the
    /// SWIM algorithm.
    pub outputs: ArrayView2<'a, f64>,
    /// Parameter sampler used in the SWIM algorithm.
    pub param_sampler: ParamSampler,
    /// The probability distribution used when sampling from the input space.
    pub input_sampler_prob_dist: InputSamplerProbDist,
}

impl<'a> SWIMSamplerConfig<'a> {
    /// Creates a new [`SWIMSampler`] given configuration. But directly using this
    /// function is discouraged. See [`crate::SamplerConfig::new`] instead.
    /// [`crate::SamplerConfig`] wraps a specific sampler config.
    pub fn new(self) -> SWIMSampler<'a> {
        SWIMSampler::new(self)
    }
}
