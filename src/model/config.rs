use ndarray::Array;
use crate::{Activation, Linear, Model};

pub struct ModelConfig {
    pub layer_width: usize,
    pub input_size: usize,
    pub output_size: usize,
    pub activation: Activation,
}

impl ModelConfig {
    pub fn new(self) -> Model {
        // of shape (input_size, layer_width), because we want to have data matrix
        // X mult by weights for the forward pass
        let linear1_weights = Array::zeros((self.layer_width, self.input_size)).reversed_axes();
        let linear1_biases = Array::zeros((1, self.layer_width));

        let linear2_weights = Array::zeros((self.output_size, self.layer_width)).reversed_axes();
        let linear2_biases = Array::zeros((1, self.output_size));

        Model {
            linear1: Linear::new(linear1_weights, linear1_biases),
            activation: self.activation,
            linear2: Linear::new(linear2_weights, linear2_biases),
        }
    }
}
