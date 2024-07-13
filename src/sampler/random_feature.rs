mod config;
mod sampler;

pub use config::RandomFeatureSamplerConfig;
pub use sampler::RandomFeatureSampler;

#[derive(Clone, Copy)]
pub enum WeightSampler {
    Normal,
}

#[derive(Clone, Copy)]
pub enum BiasSampler {
    Uniform,
}
