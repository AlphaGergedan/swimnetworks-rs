use std::f64::consts::PI;

use ndarray::Array;
use ndarray_rand::{rand_distr::{Normal, Uniform}, RandomExt};
use crate::{random_feature::{BiasSampler, WeightSampler}, Model, RandomFeatureSamplerConfig, Sample};

pub struct RandomFeatureSampler {
    weight_sampler: WeightSampler,
    bias_sampler: BiasSampler,
}

impl RandomFeatureSampler {
    pub fn new(config: RandomFeatureSamplerConfig) -> Self {
        Self { weight_sampler: config.weight_sampler, bias_sampler: config.bias_sampler }
    }
}

impl Sample for RandomFeatureSampler {
    fn sample(&self, model: &mut Model) {
        match self.weight_sampler {
            WeightSampler::Normal => sample_dense_layer_weights_normally(model, 0., 1.),
            _ => unimplemented!("-> Given weight sampling method for random feature is not implemented yet."),
        }

        match self.bias_sampler {
            // TODO: make uniform sampling range variable
            BiasSampler::Uniform => sample_dense_layer_biases_uniformly(model, -PI, PI),
            _ => unimplemented!("-> Given bias sampling method for random feature is not implemented yet."),
        }
    }
}

fn sample_dense_layer_weights_normally(model: &mut Model, mean: f64, std_dev: f64) {
    let weights = Array::<f64, _>::random(model.first_layer().weights().dim(), Normal::new(mean, std_dev).unwrap());
    model.first_layer_mut().set_weights(weights);
}

fn sample_dense_layer_biases_uniformly(model: &mut Model, min: f64, max: f64) {
    let biases = Array::<f64, _>::random(model.first_layer().biases().dim(), Uniform::new(min, max));
    model.first_layer_mut().set_biases(biases)
}

#[cfg(test)]
mod tests {
    use crate::{ModelConfig, Activation, Model};
    use super::*;

    #[test]
    fn sample_weights_with_correct_dimensions() {
        let mut model = get_shallow_model(1, 1, 256);

        let dims_before = model.first_layer().weights().dim();
        sample_dense_layer_weights_normally(&mut model, 0., 1.);
        let dims_after = model.first_layer().weights().dim();

        assert_eq!(dims_after, dims_before);

        let mut model = get_shallow_model(20, 20, 50);

        let dims_before = model.first_layer().weights().dim();
        sample_dense_layer_weights_normally(&mut model, 0., 1.);
        let dims_after = model.first_layer().weights().dim();

        assert_eq!(dims_after, dims_before);
    }

    #[test]
    fn sample_biases_with_correct_dimensions() {
        let mut model = get_shallow_model(1, 1, 256);

        let dims_before = model.first_layer().biases().dim();
        sample_dense_layer_biases_uniformly(&mut model, -10., 10.);
        let dims_after = model.first_layer().biases().dim();

        assert_eq!(dims_after, dims_before);

        let mut model = get_shallow_model(20, 20, 50);

        let dims_before = model.first_layer().biases().dim();
        sample_dense_layer_biases_uniformly(&mut model, -10., 10.);
        let dims_after = model.first_layer().biases().dim();

        assert_eq!(dims_after, dims_before);
    }

    fn get_shallow_model(input_size: usize, output_size: usize, layer_width: usize) -> Model {
        let model_config = ModelConfig {
            activation: Activation::Tanh,
            input_size,
            output_size,
            layer_width,
        };
        model_config.new()
    }
}
