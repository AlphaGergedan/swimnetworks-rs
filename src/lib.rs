//! The `swimnetworks` crate provides algorithms to quickly train neural networks on a CPU.
//!
//! We include configuration to different sampling algorithms such as Sample Where It Matter (SWIM)
//! and random features. Sampling neural networks allows quick training on a CPU, as they do not
//! rely on traditional neural network training methods. See the [associated paper][associated_paper]
//! for more details. Note that this crate is the reimplementation of the
//! [original repository][original_repo] that implements the SWIM algorithm.
//!
//! ## Documentation
//!
//! See the docs for [`Model`] for model creation and training, and the module docs
//! [`random_feature`] and [`swim`] for specific sampling algorithm implementation.
//!
//! [original_repo]: https://gitlab.com/felix.dietrich/swimnetworks
//! [associated_paper]: https://arxiv.org/abs/2306.16830
//!
//! ## Crate Status
//!
//! Current implementation almost covers all the features from the original library, but there is
//! still no guarantee at this stage if you decide to use this crate.

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
