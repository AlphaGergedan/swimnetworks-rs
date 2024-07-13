// TODO: copyright notice
//! General purpose neural network sampling library, Sample Where It Matters (SWIM), implementation in Rust.
//!
//! Original repository: https://gitlab.com/felix.dietrich/swimnetworks
//! Assosicated paper: https://arxiv.org/abs/2306.16830
//!
//! ## Neural Network Sampling
//!
//! **swimnetworks** supports both random feature and SWIM sampling methods. Sampling neural
//! networks allows allows quick training on a CPU. For more details and theoretical background you
//! can both chech the original repository and the associated paper. Sampled networks can be used
//! for regression and classification problem.

mod model;
mod activation;
mod linear;
mod layer;
mod sampler;
mod sample;

pub use model::*;
pub use activation::*;
pub use linear::*;
pub use layer::*;
pub use sampler::*;
pub use sample::*;
