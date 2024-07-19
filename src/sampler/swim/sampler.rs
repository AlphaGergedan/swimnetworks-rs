use ndarray::prelude::*;
use ndarray_linalg::{normalize, NormalizeAxis};
use ndarray_rand::rand::prelude::*;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use crate::{swim::{ParamSampler, InputSamplerProbDist}, Model, SWIMSamplerConfig, Sample};

type Inputs = Array2<f64>;
type Outputs = Array2<f64>;

type Indices = Array1<usize>;
type Distances = Array1<f64>;

// Directions are same as Inputs: there are as many directions as input features
type Directions = Inputs;
type NormalizedDirections = Directions;

/// SWIM sampler.
pub struct SWIMSampler<'a> {
    inputs: ArrayView2<'a, f64>,
    outputs: ArrayView2<'a, f64>,
    param_sampler: ParamSampler,
    input_sampler_prob_dist: InputSamplerProbDist,
}

impl<'a> SWIMSampler<'a> {
    /// Creates a new [`SWIMSampler`] given configuration. But directly using this
    /// function is discouraged. See [`crate::SamplerConfig::new`] instead.
    /// [`crate::SamplerConfig`] wraps a specific sampler config.
    pub fn new(config: SWIMSamplerConfig<'a>) -> Self {
        Self {
            inputs: config.inputs,
            outputs: config.outputs,
            param_sampler: config.param_sampler,
            input_sampler_prob_dist: config.input_sampler_prob_dist,
        }
    }
}

impl<'a> Sample for SWIMSampler<'a> {
    fn sample(&self, model: &mut crate::Model) {

        // Only sample as many neurons as there are in the uniform case,
        // this is because for uniform we do not need any candidate points,
        // sample_candidates function already picks from the input uniformly
        let num_candidates = match self.input_sampler_prob_dist {
            InputSamplerProbDist::Uniform => model.layer_width(),
            InputSamplerProbDist::SWIM(_, _) => self.inputs.nrows(),
        };

        // Sample input candidates (owned representations, can we improve here? TODO)
        let (candidate_indices_from, candidate_indices_to, candidate_inputs_from, candidate_inputs_to, candidate_normalized_directions, candidate_distances) = sample_candidates(self.inputs, num_candidates);

        match self.input_sampler_prob_dist {
            InputSamplerProbDist::Uniform => {
                // early drop of one of the candidate parts, as we only need one
                drop(candidate_inputs_to);
                // early drop of the indices
                drop(candidate_indices_from);
                drop(candidate_indices_to);

                sample_params(model, candidate_inputs_from, candidate_normalized_directions, candidate_distances, self.param_sampler);
            },
            // TODO: use input_norm and output_norm when sampling points, and selection weights
            InputSamplerProbDist::SWIM(_, _) => {
                //todo!("SWIM SAMPLING CALLED on INPUTS");
                // Compute function value differences of the candidate points
                let candidate_outputs_from = self.outputs.select(Axis(0), candidate_indices_from.as_slice().unwrap());
                let candidate_outputs_to = self.outputs.select(Axis(0), candidate_indices_to.as_slice().unwrap());

                // early drop of one of the candidate parts, as we only need one
                drop(candidate_inputs_to);
                // early drop of the indices
                drop(candidate_indices_from);
                drop(candidate_indices_to);

                let candidate_d_outputs = candidate_outputs_to - candidate_outputs_from;

                // candidate indices to select from the candidate_* variables
                let candidate_indices: Vec<usize> = (0..candidate_normalized_directions.nrows()).collect();
                let candidate_weights = candidate_weights(candidate_d_outputs, candidate_distances.clone());

                let selected_indices_iter = candidate_indices.choose_multiple_weighted(&mut ThreadRng::default(), model.layer_width(), |&item| candidate_weights[item]).unwrap();
                let selected_indices = selected_indices_iter.into_iter().map(|&i| i).collect::<Vec<usize>>();
                let selected_inputs_from = candidate_inputs_from.select(Axis(0), selected_indices.as_slice());
                let selected_normalized_directions = candidate_normalized_directions.select(Axis(0), selected_indices.as_slice());
                let selected_distances = candidate_distances.select(Axis(0), selected_indices.as_slice());

                sample_params(model, selected_inputs_from, selected_normalized_directions, selected_distances, self.param_sampler);
            },
        }
    }
}

// Samples the parameters of the hidden layer of the given model using SWIM algorithm.
fn sample_params(model: &mut Model, selected_input_from: Array2<f64>, normalized_directions: Array2<f64>, distances: Array1<f64>, param_sampler: ParamSampler) {
    assert_eq!(model.layer_width(), selected_input_from.nrows());
    assert_eq!(model.layer_width(), normalized_directions.nrows());
    assert_eq!(model.layer_width(), distances.len());

    let scale_1 = match param_sampler {
        ParamSampler::Relu => 1.,
        ParamSampler::Tanh => 3.0_f64.ln(),  // ln(3)
    };

    // reversed_axes makes (M, D) -> (D, M) where M == layer width
    // this is needed becuase in the end we will have data matrix X mult. weights
    // and X has shape (N,D)
    let weights = scale_1 * (normalized_directions / distances.insert_axis(Axis(1))).reversed_axes();

    let dims_before = model.first_layer().weights().dim();

    model.first_layer_mut().set_weights(weights);

    let dims_after = model.first_layer().weights().dim();
    assert_eq!(dims_after, dims_before);

    let scale_2 = match param_sampler {
        ParamSampler::Relu => 0.,
        ParamSampler::Tanh => 2.0_f64 * scale_1,
    };

    let dims_before = model.first_layer().biases().dim();

    // we take inner product btw. candidate inpute and the weights
    // (layer_width, num_features) * (num_features, layer_width).T
    let prod = selected_input_from * model.first_layer().weights().t();
    let inner_prod = prod.sum_axis(Axis(1));
    assert_eq!(inner_prod.len(), model.layer_width());

    let biases = -inner_prod - scale_2;
    let biases = biases.insert_axis(Axis(0));

    model.first_layer_mut().set_biases(biases);

    let dims_after = model.first_layer().biases().dim();
    assert_eq!(dims_after, dims_before);
}

// Sample directions from points to other points in the given dataset (inputs, outputs).
//
// # Returns
//
// * Candidates (*from* parts)
//
// * Candidates (*to* part)
//
// * Normalized directions, 2d array of directions computed using candidate inputs *from* and *to*
// parts, and normalized using L2-Norm.
//
// Distances between *from* and *to* candidates as 1d array.
fn sample_candidates(inputs: ArrayView2<f64>, num_candidates: usize) -> (Indices, Indices, Inputs, Inputs, NormalizedDirections, Distances) {
    let input_size = inputs.nrows();

    // ensure we have enough data points to sample
    assert!(num_candidates <= input_size);

    // candidate x0 list ~ Uniform[0, N-1]
    let candidate_indices_from: Indices = Array::random(num_candidates, Uniform::new(0, input_size));

    // candidate x1 list using delta ~ Uniform[1, N-2] and ((x0 + delta) % N)
    let delta = Array::random(num_candidates, Uniform::new(1, input_size - 1));
    let candidate_indices_to: Indices = (candidate_indices_from.clone() + delta) % input_size;

    // compute directions
    let candidate_inputs_from: Inputs = inputs.select(Axis(0), candidate_indices_from.as_slice().unwrap());
    let candidate_inputs_to: Inputs = inputs.select(Axis(0), candidate_indices_to.as_slice().unwrap());

    let directions: Directions = candidate_inputs_to.clone() - candidate_inputs_from.clone();

    // compute distances, and normalize the directions
    let (normalized_directions, distances): (NormalizedDirections, Vec<f64>) = normalize(directions, NormalizeAxis::Row);
    let distances: Distances = Array1::from_vec(distances);

    (candidate_indices_from, candidate_indices_to, candidate_inputs_from, candidate_inputs_to, normalized_directions, distances)
}

// Computes weights for each candidate depending on their differences using the probability
// distribution defined in [SWIM paper][associated_paper]. Candidates are not given as input to the
// function, rather, their distances and output differences (gradients) are given. Utilizing this
// function when picking the candidates we sample at the large gradients of the target function.
//
// # Returns
//
// * Array of probabilities for each candidate.
fn candidate_weights(candidate_d_outputs: Outputs, candidate_distances: Distances) -> Array1<f64> {
    // Compute the maximum change over all directions in the output to sample at large gradients
    // over all output directions

    let candidate_d_outputs_max = candidate_d_outputs.map_axis(Axis(1), |row| {
        row.fold(f64::NEG_INFINITY, |max, &val| val.abs().max(max))
    });
    let gradients = candidate_d_outputs_max / candidate_distances;

    // When all gradients are small avoid division by a small number
    // and default to uniform distribution.
    if gradients.sum() < 1e-10_f64 || gradients.sum().is_nan() {
        let ones = Array::<f64, Ix1>::ones(gradients.dim());
        let len = gradients.len() as f64;

        ones / len
    } else {
        gradients.clone() / gradients.sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_candidates_test() {
        let inputs: Inputs = Array::from_shape_vec((10, 1), (1..=10).map(|v| v as f64).collect()).unwrap();
        assert_eq!(inputs.dim(), (10, 1));

        let (i_from, i_to, from, to, normalized_directions, distances) = sample_candidates(inputs.view(), 1);

        assert_eq!(i_from.dim(), 1);
        assert_eq!(i_to.dim(), 1);
        assert_eq!(from.dim(), (1,1));
        assert_eq!(to.dim(), (1,1));
        assert_eq!(normalized_directions.dim(), (1,1));
        assert_eq!(distances.dim(), 1);

        let (i_from, i_to, from, to, normalized_directions, distances) = sample_candidates(inputs.view(), 10);

        assert_eq!(i_from.dim(), 10);
        assert_eq!(i_to.dim(), 10);
        assert_eq!(from.dim(), (10,1));
        assert_eq!(to.dim(), (10,1));
        assert_eq!(normalized_directions.dim(), (10,1));
        assert_eq!(distances.dim(), 10);
    }

    #[test]
    fn candidate_weights_test() {
        // all gradients are 0, it should fallback to uniform sampling
        let outputs: Outputs = Array::from_shape_vec((10, 1), vec![0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]).unwrap();
        let distances: Distances = Array::from_shape_vec((10,), vec![1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]).unwrap();
        let probabilities = candidate_weights(outputs, distances);
        let probs: Vec<f64> = probabilities.to_vec();

        let expected_prob = 1.0_f64 / (probs.len() as f64);

        for &prob in probs.iter() {
            assert!((prob - expected_prob).abs() < f64::EPSILON);
        }

        let expected_sum = 1.0_f64;
        assert!((probabilities.sum() - expected_sum) < f64::EPSILON);

        // gradients are very high at one candidate, it should be the most weighted one
        let outputs: Outputs = Array::from_shape_vec((10, 1), vec![0., 0., 0., 0., 0., 0., 0., 0., 0., 100.]).unwrap();
        let distances: Distances = Array::from_shape_vec((10,), vec![1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]).unwrap();
        let probabilities = candidate_weights(outputs, distances);
        let probs: Vec<f64> = probabilities.to_vec();

        let expected_high_prob = 1_f64;
        let expected_low_prob = 0_f64;

        for (i, &prob) in probs.iter().enumerate() {
            if i == probs.len() - 1 {
                assert!((prob - expected_high_prob) < f64::EPSILON);
            } else {
                assert!((prob - expected_low_prob) < f64::EPSILON);
            }
        }

        let expected_sum = 1.0_f64;
        assert!((probabilities.sum() - expected_sum) < f64::EPSILON);

        // gradients are very high at 3 candidates
        let outputs: Outputs = Array::from_shape_vec((10, 1), vec![0., 100., 0., 0., 0., 0., 0., 0., 100., 100.]).unwrap();
        let distances: Distances = Array::from_shape_vec((10,), vec![1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]).unwrap();
        let probabilities = candidate_weights(outputs, distances);
        let probs: Vec<f64> = probabilities.to_vec();

        let expected_high_prob = 1_f64 / 3_f64; // 0.3333...
        let expected_low_prob = 0_f64;

        for (i, &prob) in probs.iter().enumerate() {
            if i == 1 || i == probs.len() - 1 || i == probs.len() - 2 {
                dbg!("my prob at index {} is {}, expected is {}", i, prob, expected_high_prob);
                assert!((prob - expected_high_prob) < f64::EPSILON);
            } else {
                assert!((prob - expected_low_prob) < f64::EPSILON);
            }
        }

        let expected_sum = 1.0_f64;
        assert!((probabilities.sum() - expected_sum) < f64::EPSILON);
    }
}
