use ndarray::Array2;

/// Linear layer.
pub struct Linear {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
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
}
