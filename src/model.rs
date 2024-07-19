use crate::{Activation, Linear};

mod config;
mod layer;
mod fit;

pub use config::*;

/// Shallow neural network model with one linear layer followed by an activation function and
/// another linear layer.
///
/// # Examples
///
/// * See [`ModelConfig`] and [`ModelConfig::new`] for model creation.
/// * See [`crate::random_feature`] and [`crate::swim`] for sampling.
/// * See [`Model::fit`] for training.
pub struct Model {
    linear1: Linear,
    activation: Activation,
    linear2: Linear,
}

/// Methods for encapsulation.
impl Model {
    /// First (linear) layer of the network.
    pub fn first_layer(&self) -> &Linear {
        &self.linear1
    }

    /// First (linear) layer of the network as a mutable reference.
    /// Useful for updating weights after initialization.
    pub fn first_layer_mut(&mut self) -> &mut Linear {
        &mut self.linear1
    }

    /// First (usually non-linear) activation function of the network.
    pub fn first_activation(&self) -> &Activation {
        &self.activation
    }

    /// Last (linear) layer of the network.
    pub fn last_layer(&self) -> &Linear {
        &self.linear2
    }

    /// Network width (number of neurons in the first linear layer).
    pub fn layer_width(&self) -> usize {
        self.first_layer().weights().ncols()
    }

    /// Size (number of dimensions) of the input to the network.
    pub fn input_size(&self) -> usize {
        self.first_layer().weights().nrows()
    }

    /// Size (number of dimensions) of the output of the network.
    pub fn output_size(&self) -> usize {
        self.last_layer().weights().ncols()
    }

    /// Total number of layers in the network.
    pub fn number_of_layers(&self) -> usize {
        2
    }

    /// Total number of parameters in the network.
    pub fn number_of_params(&self) -> usize {
        self.number_of_weights() + self.number_of_biases()
    }

    // Total number of parameters in the linear layer weights of the network.
    fn number_of_weights(&self) -> usize {
        let dense_weights = self.first_layer().weights().len();
        let linear_weights = self.last_layer().weights().len();

        dense_weights + linear_weights
    }

    // Total number of parameters in the linear layer biases of the network.
    fn number_of_biases(&self) -> usize {
        let dense_biases = self.first_layer().biases().len();
        let linear_biases = self.last_layer().biases().len();

        dense_biases + linear_biases
    }
}
