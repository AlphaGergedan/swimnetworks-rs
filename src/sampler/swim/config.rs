use ndarray::ArrayView2;
use crate::swim::{InputSamplerProbDist, ParamSampler, SWIMSampler};

#[derive(Clone, Copy)]
pub struct SWIMSamplerConfig<'a> {
    pub inputs: ArrayView2<'a, f64>,
    pub outputs: ArrayView2<'a, f64>,
    pub param_sampler: ParamSampler,
    pub input_sampler_prob_dist: InputSamplerProbDist,
}

impl<'a> SWIMSamplerConfig<'a> {
    pub fn new(self) -> SWIMSampler<'a> {
        SWIMSampler::new(self)
    }
}
