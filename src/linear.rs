use ndarray::Array2;
use crate::Layer;

/// Linear layer representation which is a dense layer without the activation. Includes weights and
/// biases, and linearly maps the input.
pub struct Linear {
    weights: Array2<f64>,
    biases: Array2<f64>,
}

/// Methods for encapsulation.
impl Linear {
    /// Creates a new layer given parameters.
    pub fn new(weights: Array2<f64>, biases: Array2<f64>) -> Self {
        Linear { weights, biases }
    }

    /// Returns the weights of the linear layer.
    pub fn weights(&self) -> &Array2<f64> {
        &self.weights
    }

    /// Returns the biases of the linear layer.
    pub fn biases(&self) -> &Array2<f64> {
        &self.biases
    }

    /// Sets the weights of the linear layer.
    pub fn set_weights(&mut self, new_weights: Array2<f64>) {
        self.weights = new_weights;
    }

    /// Sets the biases of the linear layer.
    pub fn set_biases(&mut self, new_biases: Array2<f64>) {
        self.biases = new_biases;
    }
}

impl Layer for Linear {
    fn forward(&self, inputs: Array2<f64>) -> Array2<f64> {
        inputs.dot(self.weights()) + self.biases()
    }
}
