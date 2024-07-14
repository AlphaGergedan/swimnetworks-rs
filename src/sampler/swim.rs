mod config;
mod sampler;

pub use config::SWIMSamplerConfig;
pub use sampler::SWIMSampler;

#[derive(Clone, Copy)]
pub enum ParamSampler {
    Relu, Tanh
}

// Probability distribution used for input data sampling from the given inputs
#[derive(Clone, Copy)]
pub enum InputSamplerProbDist {
    Uniform,
    SWIM(InputNorm, OutputNorm),
}

// norms over the input space coming from the previous layer
// used in swim data sampler when sampling the data points on the divident
// TODO equation
// TODO remove public
#[derive(Clone, Copy)]
pub enum InputNorm {
    L2,
}

// norms over the function values
// used in the data sampler when sampling the data points
// TODO: equation
// TODO: remove public
#[derive(Clone, Copy)]
pub enum OutputNorm {
    Max,
}
