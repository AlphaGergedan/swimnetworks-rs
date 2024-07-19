use std::{mem, collections::HashSet};

use ndarray::{Array, Array2};
use ndarray_linalg::Norm;
use ndarray_rand::{rand_distr::Uniform, RandomExt};

use swimnetworks::{
    // neural network basis
    Activation, Layer, Model, ModelConfig, Sample, SamplerConfig,

    // random feature sampler
    random_feature, RandomFeatureSamplerConfig,

    // swim sampler
    swim, SWIMSamplerConfig
};

type Input = f64;
type Output = f64;

type Inputs = Array2<Input>;
type Outputs = Array2<Input>;

type TargetFunction = fn(Input) -> Output;

fn main() {
    println!("------- Example Regression -------\n");

    let polynomial_deg_1 = Target {
        target_fn: |x| polynomial(x, 1),
        name: "Polynomial of degree 1".to_string(),
        equation: "f(x) = 1 + x".to_string(),
    };

    let polynomial_deg_2 = Target {
        target_fn: |x| polynomial(x, 2),
        name: "Polynomial of degree 2".to_string(),
        equation: "f(x) = 1 + x + x^2".to_string(),
    };

    let polynomial_deg_3 = Target {
        target_fn: |x| polynomial(x, 3),
        name: "Polynomial of degree 3".to_string(),
        equation: "f(x) = 1 + x + x^2 + x^3".to_string(),
    };

    let trigonometric = Target {
        target_fn: |x| trigonometric(x),
        name: "Trigonometric".to_string(),
        equation: "f(x) = sin(x) + cos^2(x) + tanh(x)".to_string(),
    };

    let targets: [Target; 4] = [polynomial_deg_1, polynomial_deg_2, polynomial_deg_3, trigonometric];
    let (train_inputs, test_inputs) = sample_train_test_set_inputs(-10_f64, 10_f64, 9000, 3000);

    for target in targets.into_iter() {
        let train_truths = get_truths(target.target_fn, train_inputs.clone());
        let test_truths = get_truths(target.target_fn, test_inputs.clone());

        println!("-> Fitting {}", target.name);
        println!("-> Equation: {}\n", target.equation);
        println!("L2 Error Relative:");

        let random_feature_sampled_model = get_random_feature_sampled_model(train_inputs.clone(), train_truths.clone());
        let train_outputs = random_feature_sampled_model.forward(train_inputs.clone());
        let test_outputs = random_feature_sampled_model.forward(test_inputs.clone());
        let train_error = l2_error_relative(&train_truths, &train_outputs);
        let test_error = l2_error_relative(&test_truths, &test_outputs);
        drop(test_outputs);
        drop(random_feature_sampled_model);
        println!("-> Random Feature Sampled Model: \t train error = {}, test error = {}", train_error, test_error);

        let swim_sampled_model = get_swim_sampled_model(train_inputs.clone(), train_truths.clone());
        let train_outputs = swim_sampled_model.forward(train_inputs.clone());
        let test_outputs = swim_sampled_model.forward(test_inputs.clone());
        let train_error = l2_error_relative(&train_truths, &train_outputs);
        let test_error = l2_error_relative(&test_truths, &test_outputs);
        drop(test_outputs);
        drop(swim_sampled_model);
        println!("-> SWIM Sampled Model: \t\t\t train error = {}, test error = {}\n", train_error, test_error);
    }
    println!("----------------------------------\n");
}

fn l2_error_relative(truths: &Outputs, outputs: &Outputs) -> f64 {
    l2_error(truths, outputs) / truths.norm()
}

fn l2_error(truths: &Outputs, outputs: &Outputs) -> f64 {
    (truths - outputs).norm()
}

// Returns a swim sampled shallow neural network
fn get_random_feature_sampled_model(inputs: Inputs, truths: Outputs) -> Model {
    let model_config = ModelConfig {
        activation: Activation::Relu,
        input_size: 1,
        output_size: 1,
        layer_width: 512,
    };
    let mut model = model_config.new();

    // sample the dense layer using random features
    let sampler_config = RandomFeatureSamplerConfig {
        weight_sampler: random_feature::WeightSampler::Normal,
        bias_sampler: random_feature::BiasSampler::Uniform,
    };
    let sampler_config = SamplerConfig::RandomFeature(sampler_config);
    let sampler = sampler_config.new();
    sampler.sample(&mut model);

    // fit the last linear layer of the model,
    // using least squares solution given the truths (supervised learning)
    model.fit(inputs.view(), truths.view());

    model
}

fn get_swim_sampled_model(inputs: Inputs, truths: Outputs) -> Model {
    let model_config = ModelConfig {
        activation: Activation::Relu,
        input_size: 1,
        output_size: 1,
        layer_width: 512,
    };
    let mut model = model_config.new();

    // Sample the dense layer using the swim algorithm
    let sampler_config = SWIMSamplerConfig {
        inputs: inputs.view(),
        outputs: truths.view(),
        param_sampler: swim::ParamSampler::Relu,
        input_sampler_prob_dist: swim::InputSamplerProbDist::SWIM(swim::InputNorm::L2, swim::OutputNorm::Max),
    };
    let sampler_config = SamplerConfig::SWIM(sampler_config);
    let sampler = sampler_config.new();
    sampler.sample(&mut model);

    // fit the last linear layer of the model,
    // using least squares solution given the truths (supervised learning)
    model.fit(inputs.view(), truths.view());

    model
}

// Structure of the target functions, including their name and equation as string for pretty print
struct Target {
    pub target_fn: TargetFunction,
    pub name: String,
    pub equation: String,
}

// Returns polynomial of degree `deg` given input using coefficients `1`.
fn polynomial(x: Input, mut deg: usize) -> Output {
    let mut result: Output = 1.0;
    while deg > 0 {
        result += x.powf(deg as f64);
        deg -= 1;
    }

    result
}

// Returns output given input using a trigonometric function including `sin`, `cos` and `tan`
fn trigonometric(x: Input) -> Output {
    x.sin() + x.cos().powf(2.) + x.tanh()
}

// Returns truths given the target function and inputs
fn get_truths(target_fn: TargetFunction, inputs: Inputs) -> Outputs {
    inputs.mapv_into(target_fn)
}

// Returns distinct train and test set inputs to the target function.
fn sample_train_test_set_inputs(lower_bound: Input, upper_bound: Input, train_size: usize, test_size: usize) -> (Inputs, Inputs) {
    let uniform_dist = Uniform::<Input>::new(lower_bound, upper_bound);

    let mut train_set_inputs: Inputs;
    let mut test_set_inputs: Inputs;

    let mut unique_inputs = HashSet::<DoubleIEEE>::new();

    loop {
        train_set_inputs = Array::random((train_size, 1), uniform_dist);
        test_set_inputs = Array::random((test_size, 1), uniform_dist);

        unique_inputs.clear();

        for &val in train_set_inputs.iter().chain(test_set_inputs.iter()) {
            unique_inputs.insert(DoubleIEEE::new(val));
        }

        // all values are unique
        if unique_inputs.len() == (train_size + test_size) {
            break;
        }
    }

    (train_set_inputs, test_set_inputs)
}

// The following code is for using floating point numbers in HashMap, as floats do not implement the hash trait.

#[derive(Hash, Eq, PartialEq)]
struct DoubleIEEE(u64, i16, i8);
impl DoubleIEEE {
    fn new(val: f64) -> DoubleIEEE {
        integer_decode(val)
    }
}

// f64 consists of 1 sign bit ++ 11 exponent bits ++ 52 mantissa bits
// Reference: https://github.com/rust-lang/rust/blob/5c674a11471ec0569f616854d715941757a48a0a/src/libcore/num/f64.rs#L203-L216
fn integer_decode(val: f64) -> DoubleIEEE {
    let bits: u64 = unsafe { mem::transmute(val) };
    let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
    let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0xfffffffffffff) << 1
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };

    exponent -= 1023 + 52;
    DoubleIEEE(mantissa, exponent, sign)
}
