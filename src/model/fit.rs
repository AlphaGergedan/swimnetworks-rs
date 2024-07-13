use ndarray::{concatenate, Array, Array2, ArrayView2, Axis, s};
use ndarray_linalg::LeastSquaresSvdInPlace;
use crate::{Model, Layer};

/// Methods for fitting the model.
impl Model {
    /// Fits the last layer (linear2) using least squares solution, in a supervised setting.
    pub fn fit(&mut self, inputs: ArrayView2<f64>, outputs: ArrayView2<f64>) {
        let inputs = self.linear1.forward(inputs.into_owned());
        let inputs = self.activation.forward(inputs);

        // now inputs to the linear 2 layer should be mapped with minimum error to the outputs

        let mut input_with_ones = concatenate![Axis(1), inputs, Array::ones((inputs.nrows(), 1))];
        //println!("inputs dims: {:?}", inputs.dim());
        //println!("outputs dims: {:?}", outputs.dim());
        //println!("inputs with ones dims: {:?}", input_with_ones.dim());
        let params: Array2<f64> = input_with_ones.least_squares_in_place(&mut outputs.into_owned()).unwrap().solution;
        //println!("A x = b <==> {} x = {}", input_with_ones, outputs);
        //println!("FITTED WEIGHTS = {:?}", params);

        //println!("-> params dim = {:?}", params.dim());

        //println!("weights dim BEFORE = {:?}", self.linear2.weights.dim());
        let weights = params.slice(s![..-1, ..]).into_owned();
        //println!("weights dim = {:?}", weights.dim());
        self.linear2.set_weights(weights);

        //println!("biases dim BEFORE = {:?}", self.linear2.biases.dim());
        let biases = params.row(params.nrows() - 1).insert_axis(Axis(1)).into_owned();
        //println!("biases dim = {:?}", biases.dim());
        self.linear2.set_biases(biases);
    }
}
