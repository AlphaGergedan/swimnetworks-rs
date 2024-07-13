use ndarray::{Array, Array1, Array2, ArrayView2, Axis};
use ndarray_linalg::{norm::NormalizeAxis, normalize};
use ndarray_rand::{rand_distr::{Normal, Uniform}, RandomExt};

use crate::Model;

type Distances = Array1<f64>;
type NormalizedDirections = Array2<f64>;

enum SwimSampler {
    Relu, Tanh
}

impl Model {
    /// Samples dense layer parameters (weights and biases) using the SWIM algorithm.
    fn sample_dense_layer_swimly(&mut self, sampler: SwimSampler, inputs: ArrayView2<f64>, outputs: Option<ArrayView2<f64>>) {
        match outputs {
            Some(outputs) => {
                // sample candidates as the size of inputs (FIXME make it variable?)
                let (candidate_input_from, candidate_normalized_directions, candidate_distances) = Self::sample_candidates(inputs, inputs.nrows());
                // TODO: pick the best candidates
                // TODO: compute probabilities
                // TODO: select from candidates using probabilities
                // TODO: return directions, distances, i_input_from, i_input_to
                todo!();
            },
            None => {
                // sample uniformly, without picking from candidates
                let (selected_input_from, normalized_directions, distances) = Self::sample_candidates(inputs, self.layer_width());
                self.sample_params(selected_input_from, normalized_directions, distances, sampler);
            }
        }
    }

    // inputs has shape (N, D), where N is the number of sampled and D is the dimension of the inputs
    fn sample_params(&mut self, selected_input_from: Array2<f64>, normalized_directions: Array2<f64>, distances: Array1<f64>, sampler: SwimSampler) {
        let scale_1 = match sampler {
            SwimSampler::Relu => 1.,
            SwimSampler::Tanh => 3.0_f64.ln(),  // ln(3)
        };

        let weights = scale_1 * (normalized_directions / distances);
        self.linear1.weights = weights;

        let scale_2 = match sampler {
            SwimSampler::Relu => 0.,
            SwimSampler::Tanh => 2.0_f64 * scale_1,
        };

        let biases = -self.linear1.weights.dot(&selected_input_from) - scale_2;
        drop(selected_input_from);

        self.linear1.biases = biases;
    }

    /// Sample directions from points to other points in the given dataset (inputs, outputs).
    fn sample_candidates(inputs: ArrayView2<f64>, num_candidates: usize) -> (Array2<f64>, NormalizedDirections, Distances) {
        let input_size = inputs.nrows();

        // ensure we have enough data points to sample
        assert!(num_candidates >= input_size);

        // candidate x0 list ~ Uniform[0, N-1]
        let candidates_i_input_from = Array::random(num_candidates, Uniform::new(0, input_size));

        // candidate x1 list using delta ~ Uniform[1, N-2] and ((x0 + delta) % N)
        let delta = Array::random(num_candidates, Uniform::new(1, input_size - 1));
        let candidates_i_input_to = (candidates_i_input_from.clone() + delta) % input_size;

        // compute directions
        let candidates_input_from = inputs.select(Axis(0), candidates_i_input_from.as_slice().unwrap());
        drop(candidates_i_input_from);

        let candidates_input_to = inputs.select(Axis(0), candidates_i_input_to.as_slice().unwrap());
        drop(candidates_i_input_to);

        let directions = candidates_input_to - candidates_input_from.clone();

        // compute distances, and normalize the directions
        let (normalized_directions, distances) = normalize(directions, NormalizeAxis::Column);
        let distances = Array1::from_vec(distances);

        (candidates_input_from, normalized_directions, distances)
    }
}

/// Methods for sampling the model parameters using random features.
impl Model {
    pub fn sample_dense_layer_weights_normally(&mut self, mean: f64, std_dev: f64) {
        let weights = Array::<f64, _>::random(self.linear1.weights.dim(), Normal::new(mean, std_dev).unwrap());
        self.linear1.weights = weights;
    }

    pub fn sample_dense_layer_biases_uniformly(&mut self, min: f64, max: f64) {
        let biases = Array::<f64, _>::random(self.linear1.biases.dim(), Uniform::new(min, max));
        self.linear1.biases = biases;
    }
}
