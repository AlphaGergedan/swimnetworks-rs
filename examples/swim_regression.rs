use ndarray::prelude::*;
use ndarray_linalg::Norm;

use swimnetworks_rs::{swim, Activation, Layer, ModelConfig, SWIMSamplerConfig, Sample, SamplerConfig};

type Input = f64;
type Output = f64;
type TargetFunction = fn(Input) -> Output;

fn fit(rng_lower: Input, rng_upper: Input, rate: Input, target_fn: TargetFunction) {

    // Prepare your data to be 2 dimensional (1 for number of samples, 1 for number of features)
    let inputs = Array::range(rng_lower, rng_upper, rate).insert_axis(Axis(1)); // (N, D), where D=1
    let truths = inputs.mapv(target_fn);

    // Prepare your neural network model
    let model_config = ModelConfig {
        activation: Activation::Tanh,
        input_size: 1,
        output_size: 1,
        layer_width: 256,
    };
    let mut model = model_config.new();

    // Initial forward pass, to see how bad the weight init is (TODO currently params are
    // zero-initialized, make it just random and uniform without fitting the lstsq, or just
    // allocate memory if you can for minimum cost)
    let outputs = model.forward(inputs.clone());
    println!("-> L2 error after zero-initializing the parameters: {}", l2_error(truths.view(), outputs));

    // Sample the dense layer using random features
    let sampler_config = SWIMSamplerConfig {
        inputs: inputs.view(),
        outputs: truths.view(),
        param_sampler: swim::ParamSampler::Tanh,
        input_sampler_prob_dist: swim::InputSamplerProbDist::SWIM(swim::InputNorm::L2, swim::OutputNorm::Max),
        //input_sampler_prob_dist: swim::InputSamplerProbDist::Uniform,
    };
    let sampler_config = SamplerConfig::SWIM(sampler_config);
    let sampler = sampler_config.new();
    sampler.sample(&mut model);

    let outputs = model.forward(inputs.clone());
    println!("-> L2 error after random-feature sampling: {}", l2_error(truths.view(), outputs));

    // Fit the last linear layer of the model, using least squares solution given the truths (supervised learning)
    model.fit(inputs.view(), truths.view());

    let outputs = model.forward(inputs.clone());
    println!("-> L2 error after the last linear layer fitting lstsq: {}", l2_error(truths.view(), outputs));
}

fn main() {
    println!("------- Example Regression Task -------");
    println!("Sampler: Random Feature");
    println!();

    // global settings to sample the inputs
    const RNG_LOWER: Input = -10.;
    const RNG_UPPER: Input = 10.;
    const RATE: Input = 0.01;

    let target_fns: [TargetFunction; 7] = [polynomial_deg_1, polynomial_deg_2, polynomial_deg_3, polynomial_deg_4, polynomial_deg_5, polynomial_deg_6, polynomial_deg_7];

    for (i, target_fn) in target_fns.into_iter().enumerate() {
        println!("-> FITTING POLYNOMIAL DEG {}", i + 1);
        println!();
        fit(RNG_LOWER, RNG_UPPER, RATE, target_fn);
        println!();
    }
}

pub fn polynomial_deg_1(x: Input) -> Output {
    12. + 2.23 * x
}

pub fn polynomial_deg_2(x: Input) -> Output {
    -100. - 10. * x + 2. * x.powf(2.)
}

pub fn polynomial_deg_3(x: Input) -> Output {
    300. + 230. * x - 12. * x.powf(2.) + 80. * x.powf(3.)
}

pub fn polynomial_deg_4(x: Input) -> Output {
    300. + 230. * x - 12. * x.powf(2.) + 80. * x.powf(3.) - 120. * x.powf(4.)
}

pub fn polynomial_deg_5(x: Input) -> Output {
    300. + 230. * x - 12. * x.powf(2.) + 80. * x.powf(3.) - 120. * x.powf(4.) + 100. * x.powf(5.)
}

pub fn polynomial_deg_6(x: Input) -> Output {
    300. + 230. * x - 12. * x.powf(2.) + 80. * x.powf(3.) - 120. * x.powf(4.) + 100. * x.powf(5.) - 50. * x.powf(6.)
}

pub fn polynomial_deg_7(x: Input) -> Output {
    0.252 + 0.12 * x - 12.5 * x.powf(2.) + 10. * x.powf(3.) - 0.5 * x.powf(4.) + 0.25 * x.powf(5.) - 0.9 * x.powf(6.) + 1. * x.powf(7.)
}

pub fn l2_error(input: ArrayView2<Input>, output: Array2<Output>) -> f64 {
    (output - input).norm()
}
