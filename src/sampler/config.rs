use crate::{RandomFeatureSamplerConfig, SWIMSamplerConfig, Sampler};

pub enum SamplerConfig<'a> {
    RandomFeature(RandomFeatureSamplerConfig),
    SWIM(SWIMSamplerConfig<'a>),
}

impl<'a> SamplerConfig<'a> {
    pub fn new(self) -> Sampler<'a> {
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
