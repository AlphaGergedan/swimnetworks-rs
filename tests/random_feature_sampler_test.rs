use std::f64::consts::PI;
use swimnetworks_rs::{
    Activation, Model, ModelConfig,
    Sample, SamplerConfig,
    random_feature, RandomFeatureSamplerConfig,
};

#[test]
fn random_feature_sampler_dim_test() {
    let mut model = get_model(1, 128, 1);
    let sampler_config = RandomFeatureSamplerConfig {
        weight_sampler: random_feature::WeightSampler::Normal,
        bias_sampler: random_feature::BiasSampler::Uniform,
    };
    let sampler_config = SamplerConfig::RandomFeature(sampler_config);
    let sampler = sampler_config.new();

    let weights_dim_before = model.first_layer().weights().dim();
    let biases_dim_before = model.first_layer().biases().dim();
    sampler.sample(&mut model);
    let weights_dim_after = model.first_layer().weights().dim();
    let biases_dim_after = model.first_layer().biases().dim();

    assert_eq!(weights_dim_before, weights_dim_after);
    assert_eq!(biases_dim_before, biases_dim_after);

    let mut model = get_model(3, 128, 5);
    let sampler_config = RandomFeatureSamplerConfig {
        weight_sampler: random_feature::WeightSampler::Normal,
        bias_sampler: random_feature::BiasSampler::Uniform,
    };
    let sampler_config = SamplerConfig::RandomFeature(sampler_config);
    let sampler = sampler_config.new();

    let weights_dim_before = model.first_layer().weights().dim();
    let biases_dim_before = model.first_layer().biases().dim();
    sampler.sample(&mut model);
    let weights_dim_after = model.first_layer().weights().dim();
    let biases_dim_after = model.first_layer().biases().dim();

    assert_eq!(weights_dim_before, weights_dim_after);
    assert_eq!(biases_dim_before, biases_dim_after);

    let mut model = get_model(4, 128, 1);
    let sampler_config = RandomFeatureSamplerConfig {
        weight_sampler: random_feature::WeightSampler::Normal,
        bias_sampler: random_feature::BiasSampler::Uniform,
    };
    let sampler_config = SamplerConfig::RandomFeature(sampler_config);
    let sampler = sampler_config.new();

    let weights_dim_before = model.first_layer().weights().dim();
    let biases_dim_before = model.first_layer().biases().dim();
    sampler.sample(&mut model);
    let weights_dim_after = model.first_layer().weights().dim();
    let biases_dim_after = model.first_layer().biases().dim();

    assert_eq!(weights_dim_before, weights_dim_after);
    assert_eq!(biases_dim_before, biases_dim_after);
}

#[test]
fn random_feature_sampler_bias_range_test() {
    let mut model = get_model(1, 1, 128);

    let sampler_config = RandomFeatureSamplerConfig {
        weight_sampler: random_feature::WeightSampler::Normal,
        bias_sampler: random_feature::BiasSampler::Uniform,
    };
    let sampler_config = SamplerConfig::RandomFeature(sampler_config);
    let sampler = sampler_config.new();

    sampler.sample(&mut model);

    let (min_bias, max_bias)= model.first_layer().biases().fold((f64::INFINITY, f64::NEG_INFINITY), |(min_bias, max_bias), &bias| {
        (min_bias.min(bias), max_bias.max(bias))
    });

    // for now we sample biases fixed uniformly in range [-pi, pi)
    assert!(min_bias > -PI);
    assert!(max_bias < PI);
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
