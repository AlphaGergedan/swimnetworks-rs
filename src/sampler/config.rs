use crate::{RandomFeatureSamplerConfig, SWIMSamplerConfig, Sampler};

/// Sampler configuration for hidden layer sampling in neural network models.
pub enum SamplerConfig<'a> {
    RandomFeature(RandomFeatureSamplerConfig),
    SWIM(SWIMSamplerConfig<'a>),
}

impl<'a> SamplerConfig<'a> {
    /// Cretates [`Sampler`] given configuration (wraps a specific sampling method).
    ///
    /// # Examples
    ///
    /// * See [`crate::swim`] and [`crate::random_feature`] for examples.
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
