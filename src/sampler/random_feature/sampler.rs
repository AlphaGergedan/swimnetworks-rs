use ndarray::Array;
use ndarray_rand::{rand_distr::{Normal, Uniform}, RandomExt};
use crate::{Model, Sample, random_feature::{WeightSampler, BiasSampler}};

pub struct RandomFeatureSampler {
    weight_sampler: WeightSampler,
    bias_sampler: BiasSampler,
}

impl RandomFeatureSampler {
    pub fn new(weight_sampler: WeightSampler, bias_sampler: BiasSampler) -> Self {
        Self { weight_sampler, bias_sampler }
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
            BiasSampler::Uniform => sample_dense_layer_biases_uniformly(model, 0., 10.),
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
