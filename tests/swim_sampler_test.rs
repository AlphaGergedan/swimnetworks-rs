use ndarray::{array, Array, Array2};
use swimnetworks::{
    Activation, Model, ModelConfig, Sample, SamplerConfig,
    swim, SWIMSamplerConfig,
};

#[test]
fn swim_sampler_dim_test() {
    let mut model = get_model(1, 3, 1);
    let inputs = array![[1.], [2.], [3.], [4.], [5.]];
    let outputs = array![[1.], [2.], [3.], [4.], [5.]];
    let sampler_config = SWIMSamplerConfig {
        inputs: inputs.view(),
        outputs: outputs.view(),
        input_sampler_prob_dist: swim::InputSamplerProbDist::Uniform,
        param_sampler: swim::ParamSampler::Tanh,
    };
    let sampler_config = SamplerConfig::SWIM(sampler_config);
    let sampler = sampler_config.new();

    let weights_dim_before = model.first_layer().weights().dim();
    let biases_dim_before = model.first_layer().biases().dim();
    sampler.sample(&mut model);
    let weights_dim_after = model.first_layer().weights().dim();
    let biases_dim_after = model.first_layer().biases().dim();

    assert_eq!(weights_dim_before, weights_dim_after);
    assert_eq!(biases_dim_before, biases_dim_after);

    let mut model = get_model(3, 3, 5);
    let inputs: Array2<f64> = Array::zeros((5, 3));
    let outputs: Array2<f64> = Array::zeros((5, 5));
    let sampler_config = SWIMSamplerConfig {
        inputs: inputs.view(),
        outputs: outputs.view(),
        input_sampler_prob_dist: swim::InputSamplerProbDist::Uniform,
        param_sampler: swim::ParamSampler::Tanh,
    };
    let sampler_config = SamplerConfig::SWIM(sampler_config);
    let sampler = sampler_config.new();

    let weights_dim_before = model.first_layer().weights().dim();
    let biases_dim_before = model.first_layer().biases().dim();
    sampler.sample(&mut model);
    let weights_dim_after = model.first_layer().weights().dim();
    let biases_dim_after = model.first_layer().biases().dim();

    assert_eq!(weights_dim_before, weights_dim_after);
    assert_eq!(biases_dim_before, biases_dim_after);

    let mut model = get_model(4, 3, 1);
    let inputs: Array2<f64> = Array::zeros((5, 4));
    let outputs: Array2<f64> = Array::zeros((5, 1));
    let sampler_config = SWIMSamplerConfig {
        inputs: inputs.view(),
        outputs: outputs.view(),
        input_sampler_prob_dist: swim::InputSamplerProbDist::Uniform,
        param_sampler: swim::ParamSampler::Tanh,
    };
    let sampler_config = SamplerConfig::SWIM(sampler_config);
    let sampler = sampler_config.new();

    let weights_dim_before = model.first_layer().weights().dim();
    let biases_dim_before = model.first_layer().biases().dim();
    sampler.sample(&mut model);
    let weights_dim_after = model.first_layer().weights().dim();
    let biases_dim_after = model.first_layer().biases().dim();

    assert_eq!(weights_dim_before, weights_dim_after);
    assert_eq!(biases_dim_before, biases_dim_after);
}

fn get_model(input_size: usize, layer_width: usize, output_size: usize) -> Model {
    let model_config = ModelConfig {
        input_size,
        output_size,
        layer_width,
        activation: Activation::Relu,
    };
    model_config.new()
}
