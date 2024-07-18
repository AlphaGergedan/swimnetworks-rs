use ndarray::{concatenate, Array, Array2, ArrayView2, Axis, s};
use ndarray_linalg::LeastSquaresSvdInPlace;
use crate::{Model, Layer};

/// Methods for fitting the model.
impl Model {
    /// Fits the last layer (linear2) using least squares solution, in a supervised setting.
    pub fn fit(&mut self, inputs: ArrayView2<f64>, truths: ArrayView2<f64>) {
        let inputs = self.linear1.forward(inputs.into_owned());
        let inputs = self.activation.forward(inputs);

        // now inputs to the linear 2 layer should be mapped with minimum error to the truths

        let mut input_with_ones = concatenate![Axis(1), inputs, Array::ones((inputs.nrows(), 1))];
        let params: Array2<f64> = input_with_ones.least_squares_in_place(&mut truths.into_owned()).unwrap().solution;

        let weights = params.slice(s![..-1, ..]).into_owned();
        self.linear2.set_weights(weights);

        let biases = params.row(params.nrows() - 1).insert_axis(Axis(1)).t().into_owned();
        self.linear2.set_biases(biases);
    }
}
