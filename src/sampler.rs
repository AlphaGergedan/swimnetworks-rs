use crate::{Model, Sample};

pub mod config;
pub mod random_feature;
pub mod swim;

pub use random_feature::RandomFeatureSamplerConfig;
pub use swim::SWIMSamplerConfig;
pub use config::SamplerConfig;

/// Hidden layer sampler for neural network models.
pub struct Sampler<'a> {
    sampler_config: SamplerConfig<'a>,
    sampler: Box<dyn Sample + 'a>,
}

impl Sampler<'_> {
    pub fn name(&self) -> String {
        match self.sampler_config {
            SamplerConfig::RandomFeature(_) => "Random Feature".to_string(),
            SamplerConfig::SWIM(_) => "SWIM".to_string(),
        }
    }
}

impl Sample for Sampler<'_> {
    fn sample(&self, model: &mut Model) {
        self.sampler.sample(model);
    }
}
