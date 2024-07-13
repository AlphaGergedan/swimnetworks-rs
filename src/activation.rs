use ndarray::Array2;
use crate::Layer;

/// Activation functions.
#[derive(Clone)]
pub enum Activation {
    Relu, Tanh
}

impl Layer for Activation {
    fn forward(&self, inputs: Array2<f64>) -> Array2<f64> {
        match self {
            Activation::Relu => inputs.mapv(|x| x.max(0.)),
            Activation::Tanh => inputs.mapv(|x| x.tanh()),
        }
    }
}
