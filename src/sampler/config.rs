use crate::{RandomFeatureSamplerConfig, SWIMSamplerConfig, Sampler};

pub enum SamplerConfig {
    RandomFeature(RandomFeatureSamplerConfig),
    SWIM(SWIMSamplerConfig),
}

impl SamplerConfig {
    pub fn new(self) -> Sampler {
        match self {
            SamplerConfig::RandomFeature(config) => {
                Sampler {
                    sampler_config: self,
                    sampler: Box::new(config.new()),
                }
            },
            SamplerConfig::SWIM(config) => {
                Sampler {
                    sampler_config: self,
                    sampler: Box::new(config.new()),
                }
            }
        }
    }
}
