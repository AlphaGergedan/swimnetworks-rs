use ndarray::Array;
use crate::{Activation, Linear, Model};

/// Model configuration, used to create a new shallow neural network model.
pub struct ModelConfig {
    pub layer_width: usize,
    pub input_size: usize,
    pub output_size: usize,
    pub activation: Activation,
}

impl ModelConfig {
    /// Creates a new [`Model`] with the given configuration.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Constructs a mutable shallow neural network model with 512 neurons and ReLU activation
    /// // function that take one-dimensional inputs and produce one-dimensional outputs.
    /// let model_config = ModelConfig {
    ///     activation: Activation::Relu,
    ///     input_size: 1,
    ///     output_size: 1,
    ///     layer_width: 512,
    /// };
    /// let mut model = model_config.new();
    /// ```
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
