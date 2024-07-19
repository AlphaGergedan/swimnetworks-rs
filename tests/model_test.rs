use ndarray::{array, ArrayView2};
use ndarray_linalg::Norm;
use swimnetworks::{
    Activation, Layer, ModelConfig, SamplerConfig, Sample,
    RandomFeatureSamplerConfig, random_feature,
};

#[test]
fn model_encapsulation_test() {
    let model_config = ModelConfig {
        input_size: 5,
        output_size: 2,
        layer_width: 98,
        activation: Activation::Relu,
    };
    let model = model_config.new();

    assert_eq!(model.input_size(), 5);
    assert_eq!(model.output_size(), 2);
    assert_eq!(model.layer_width(), 98);
    assert!(matches!(model.first_activation(), Activation::Relu));
    assert_eq!(model.number_of_layers(), 2);
    assert_eq!(model.number_of_params(), 98*5 + 98 + 2*98 + 2);
}

#[test]
fn model_fit_test() {
    let model_config = ModelConfig {
        input_size: 1,
        output_size: 1,
        layer_width: 128,
        activation: Activation::Tanh,
    };
    let mut model = model_config.new();

    // sample using random features, this part is not related to the tested method (fit)
    let sampler_config = RandomFeatureSamplerConfig {
        weight_sampler: random_feature::WeightSampler::Normal,
        bias_sampler: random_feature::BiasSampler::Uniform,
    };
    let sampler_config = SamplerConfig::RandomFeature(sampler_config);
    let sampler = sampler_config.new();

    sampler.sample(&mut model);

    // a linear function (line)
    let inputs = array![[0.], [1.], [2.], [3.], [4.]];
    let truths = array![[1.], [1.], [1.], [1.], [1.]];

    // forward without fitting
    let outputs = model.forward(inputs.clone());
    let initial_error = l2_error(outputs.view(), truths.view());

    // forward after fitting
    model.fit(inputs.view(), truths.view());

    let outputs = model.forward(inputs);
    let error = l2_error(outputs.view(), truths.view());

    assert!(error < initial_error);
}

fn l2_error(outputs: ArrayView2<f64>, truths: ArrayView2<f64>) -> f64 {
    (outputs.into_owned() - truths).norm_l2()
}
