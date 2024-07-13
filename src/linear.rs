use ndarray::Array2;
use crate::Layer;

/// Linear layer.
pub struct Linear {
    weights: Array2<f64>,
    biases: Array2<f64>,
}

impl Linear {
    pub fn new(weights: Array2<f64>, biases: Array2<f64>) -> Self {
        Linear { weights, biases }
    }

    pub fn weights(&self) -> &Array2<f64> {
        &self.weights
    }

    pub fn biases(&self) -> &Array2<f64> {
        &self.biases
    }

    pub fn set_weights(&mut self, new_weights: Array2<f64>) {
        self.weights = new_weights;
    }

    pub fn set_biases(&mut self, new_biases: Array2<f64>) {
        self.biases = new_biases;
    }
}

impl Layer for Linear {
    fn forward(&self, inputs: Array2<f64>) -> Array2<f64> {
        inputs.dot(self.weights()) + self.biases()
    }
}
