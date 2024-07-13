use crate::random_feature::{WeightSampler, BiasSampler, RandomFeatureSampler};

#[derive(Clone, Copy)]
pub struct RandomFeatureSamplerConfig {
    pub weight_sampler: WeightSampler,
    pub bias_sampler: BiasSampler,
}

impl RandomFeatureSamplerConfig {
    pub fn new(self) -> RandomFeatureSampler {
        RandomFeatureSampler::new(self.weight_sampler, self.bias_sampler)
    }
}
