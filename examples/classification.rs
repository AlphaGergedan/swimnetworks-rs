use std::{collections::HashSet, mem};

use ndarray::{concatenate, s, stack, Array, Array2, ArrayView1, Axis};
use ndarray_rand::{rand::{seq::SliceRandom, thread_rng}, rand_distr::Normal, RandomExt};

use swimnetworks::{
    // neural network basis
    Activation, Layer, Model, ModelConfig, Sample, SamplerConfig,

    // random feature sampler
    random_feature, RandomFeatureSamplerConfig,

    // swim sampler
    swim, SWIMSamplerConfig
};

// Represents classes for the samples.
#[derive(PartialEq)]
enum Class {
    A, B, C
}

impl From<usize> for Class {
    fn from(index: usize) -> Self {
        match index {
            0 => Self::A,
            1 => Self::B,
            2 => Self::C,
            _ => panic!("Unknown class index."),
        }
    }
}

// Class distributions
const CLASS_A_MEAN: (f64, f64) = (-2_f64, 0_f64);
const CLASS_B_MEAN: (f64, f64) = (2_f64, 0_f64);
const CLASS_C_MEAN: (f64, f64) = (0_f64, 2_f64);

const CLASS_A_COV: f64 = 1_f64;
const CLASS_B_COV: f64 = 1_f64;
const CLASS_C_COV: f64 = 1_f64;

type Input = f64;

// use one-hot encoding for the outputs
type Output = f64;

type Inputs = Array2<Input>;
type Outputs = Array2<Output>;

fn main() {
    println!("------- Example Classification -------\n");

    let ((train_class_a, train_class_b, train_class_c),
         (test_class_a, test_class_b, test_class_c)) = get_train_test_sets(9000, 3000);
    assert_eq!(train_class_a.nrows() + train_class_b.nrows() + train_class_c.nrows(), 9000);
    assert_eq!((train_class_a.ncols(), train_class_b.ncols(), train_class_c.ncols()), (2, 2, 2));
    assert_eq!(test_class_a.nrows() + test_class_b.nrows() + test_class_c.nrows(), 3000);
    assert_eq!((test_class_a.ncols(), test_class_b.ncols(), test_class_c.ncols()), (2, 2, 2));

    /*
        use one-hot encoding to indicate the classes, example class a truths of the train set
        [
            [1, 0, 0],
            [1, 0, 0],
            ...
            [1, 0, 0]
        ]
        with shape = (train_class_a.len(), 3)
    */

    let train_truths: Array2<f64> = concatenate![Axis(0),
        stack![Axis(1), Array::ones(train_class_a.nrows()), Array::zeros(train_class_a.nrows()), Array::zeros(train_class_a.nrows())],
        stack![Axis(1), Array::zeros(train_class_b.nrows()), Array::ones(train_class_b.nrows()), Array::zeros(train_class_b.nrows())],
        stack![Axis(1), Array::zeros(train_class_c.nrows()), Array::zeros(train_class_c.nrows()), Array::ones(train_class_c.nrows())],
    ];
    assert_eq!(train_truths.nrows(), train_class_a.nrows() + train_class_b.nrows() + train_class_c.nrows());
    assert_eq!(train_truths.ncols(), 3);

    let test_truths: Array2<f64> = concatenate![Axis(0),
        stack![Axis(1), Array::ones(test_class_a.nrows()), Array::zeros(test_class_a.nrows()), Array::zeros(test_class_a.nrows())],
        stack![Axis(1), Array::zeros(test_class_b.nrows()), Array::ones(test_class_b.nrows()), Array::zeros(test_class_b.nrows())],
        stack![Axis(1), Array::zeros(test_class_c.nrows()), Array::zeros(test_class_c.nrows()), Array::ones(test_class_c.nrows())],
    ];
    assert_eq!(test_truths.nrows(), test_class_a.nrows() + test_class_b.nrows() + test_class_c.nrows());
    assert_eq!(test_truths.ncols(), 3);

    let train_inputs = concatenate![Axis(0), train_class_a, train_class_b, train_class_c];
    drop(train_class_a);
    drop(train_class_b);
    drop(train_class_c);
    assert_eq!(train_inputs.nrows(), 9000);
    assert_eq!(train_inputs.ncols(), 2);

    let test_inputs = concatenate![Axis(0), test_class_a, test_class_b, test_class_c];
    drop(test_class_a);
    drop(test_class_b);
    drop(test_class_c);
    assert_eq!(test_inputs.nrows(), 3000);
    assert_eq!(test_inputs.ncols(), 2);

    // shuffle the sets, actually we do not need this for this example because we are handling the
    // full dataset at once when fitting using least squares
    let (train_inputs, train_truths) = shuffle_input_output_sets(train_inputs, train_truths);
    let (test_inputs, test_truths) = shuffle_input_output_sets(test_inputs, test_truths);

    println!("-> Fitting 3 class classification");
    println!("-> Class Distributions: a ~ N((-2,0), Identity), b ~ N((2,0), Identity), c ~ N(0,2), Identity)");
    println!("Accuracy:");

    let random_feature_sampled_model = get_random_feature_sampled_model(train_inputs.clone(), train_truths.clone());
    let train_outputs = random_feature_sampled_model.forward(train_inputs.clone());
    let test_outputs = random_feature_sampled_model.forward(test_inputs.clone());
    let train_acc = accuracy(&train_truths, &train_outputs);
    let test_acc = accuracy(&test_truths, &test_outputs);
    drop(test_outputs);
    drop(random_feature_sampled_model);
    println!("-> Random Feature Sampled Model: \t train accuracy = {}, test accuracy = {}", train_acc, test_acc);

    let swim_sampled_model = get_swim_sampled_model(train_inputs.clone(), train_truths.clone());
    let train_outputs = swim_sampled_model.forward(train_inputs.clone());
    let test_outputs = swim_sampled_model.forward(test_inputs.clone());
    let train_acc = accuracy(&train_truths, &train_outputs);
    let test_acc = accuracy(&test_truths, &test_outputs);
    drop(test_outputs);
    drop(swim_sampled_model);
    println!("-> SWIM Sampled Model: \t\t\t train accuracy = {}, test accuracy = {}\n", train_acc, test_acc);

    println!("--------------------------------------");
}

// Returns one-hot encoding of the predicted class
fn get_predicted_class(output: ArrayView1<Output>) -> Class {
    let (max_index, _) = output.iter().enumerate().fold((usize::MAX, f64::NEG_INFINITY), |(max_index, max_val), (index, &val)| {
        if max_val < val {
            return (index, val);
        }
        (max_index, max_val)
    });

    Class::from(max_index)
}

// Computes accuracy given truths and predicted outputs. Inputs to this function should be one-hot
// encoded
fn accuracy(truths: &Outputs, outputs: &Outputs) -> f64 {
    assert_eq!(truths.dim(), outputs.dim());
    let num_samples = truths.nrows();

    let truths = truths.map_axis(Axis(1), |one_hot_encoding| {
        get_predicted_class(one_hot_encoding)
    });

    let outputs = outputs.map_axis(Axis(1), |class_logits| {
        get_predicted_class(class_logits)
    });

    // count the correctly predicted classes
    let mut count_correct = 0_usize;
    for (index, pred_class) in outputs.iter().enumerate() {
        let true_class = &truths[index];
        if pred_class == true_class {
            count_correct += 1;
        }
    }

    count_correct as f64 / num_samples as f64
}

// Creates a shallow neural network, samples its hidden layer using random features and
// fits the last layer parameters using least squares
fn get_random_feature_sampled_model(inputs: Inputs, truths: Outputs) -> Model {
    let model_config = ModelConfig {
        activation: Activation::Tanh,
        input_size: 2,
        output_size: 3,
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

// Creates a shallow neural network, samples its hidden layer using SWIM algorithm and
// fits the last layer parameters using least squares
fn get_swim_sampled_model(inputs: Inputs, truths: Outputs) -> Model {
    let model_config = ModelConfig {
        activation: Activation::Tanh,
        input_size: 2,
        output_size: 3,
        layer_width: 512,
    };
    let mut model = model_config.new();

    // Sample the dense layer using the swim algorithm
    let sampler_config = SWIMSamplerConfig {
        inputs: inputs.view(),
        outputs: truths.view(),
        param_sampler: swim::ParamSampler::Tanh,
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

// Returns train and test sets containing samples belonging to 3 classes, with equal number of
// samples in each class.
fn get_train_test_sets(train_size: usize, test_size: usize) -> ((Inputs, Inputs, Inputs), (Inputs, Inputs, Inputs)) {
    assert_eq!(train_size % 3, 0);
    assert_eq!(test_size % 3, 0);

    let class_train_size = train_size / 3;
    let class_test_size = test_size / 3;

    let mut class_a_train_inputs: Inputs;
    let mut class_a_test_inputs: Inputs;

    let mut class_b_train_inputs: Inputs;
    let mut class_b_test_inputs: Inputs;

    let mut class_c_train_inputs: Inputs;
    let mut class_c_test_inputs: Inputs;

    let mut unique_inputs = HashSet::<DoubleIEEE>::new();

    loop {
        class_a_train_inputs = sample_class_a_inputs(class_train_size);
        class_a_test_inputs = sample_class_a_inputs(class_test_size);

        class_b_train_inputs = sample_class_b_inputs(class_train_size);
        class_b_test_inputs = sample_class_b_inputs(class_test_size);

        class_c_train_inputs = sample_class_c_inputs(class_train_size);
        class_c_test_inputs = sample_class_c_inputs(class_test_size);

        unique_inputs.clear();

        assert_eq!(class_a_train_inputs.len(), class_train_size * 2);
        assert_eq!(class_b_train_inputs.len(), class_train_size * 2);
        assert_eq!(class_c_train_inputs.len(), class_train_size * 2);

        assert_eq!(class_a_test_inputs.len(), class_test_size * 2);
        assert_eq!(class_b_test_inputs.len(), class_test_size * 2);
        assert_eq!(class_c_test_inputs.len(), class_test_size * 2);

        assert_eq!(class_train_size * 3, train_size);
        assert_eq!(class_test_size * 3, test_size);

        assert_eq!(class_a_train_inputs.len() + class_b_train_inputs.len() + class_c_train_inputs.len(), train_size * 2);
        assert_eq!(class_a_test_inputs.len() + class_b_test_inputs.len() + class_c_test_inputs.len(), test_size * 2);

        // create chain iterator for all inputs
        let chain_iter = class_a_train_inputs.iter()
            .chain(class_a_test_inputs.iter())
            .chain(class_b_train_inputs.iter())
            .chain(class_b_test_inputs.iter())
            .chain(class_c_train_inputs.iter())
            .chain(class_c_test_inputs.iter());

        for &val in chain_iter {
            unique_inputs.insert(DoubleIEEE::new(val));
        }

        println!("-> unique_inputs.len() = {}", unique_inputs.len());

        // all values are unique, there are 3 classes, for each class we have train and test sets,
        // and each set contains 2 dimensional inputs
        if unique_inputs.len() == ((train_size + test_size) * 2) {
            break;
        }
    }

    ((class_a_train_inputs, class_b_train_inputs, class_c_train_inputs), (class_a_test_inputs, class_b_test_inputs, class_c_test_inputs))
}

// Samples class a samples (inputs that belong to class a) using a multivariate distribution with
// mean=(-1,0) and covariance=Identity
fn sample_class_a_inputs(num_samples: usize) -> Inputs {
    let class_a_dist_first_dimension = Normal::new(CLASS_A_MEAN.0, CLASS_A_COV).unwrap();
    let class_a_dist_second_dimension = Normal::new(CLASS_A_MEAN.1, CLASS_A_COV).unwrap();

    let class_a_train_input_first_dimensions = Array::random(num_samples, class_a_dist_first_dimension);
    let class_a_train_input_second_dimensions = Array::random(num_samples, class_a_dist_second_dimension);

    stack![Axis(1), class_a_train_input_first_dimensions, class_a_train_input_second_dimensions]
}

// Samples class b samples (inputs that belong to class b) using a multivariate distribution with
// mean=(1,0) and covariance=Identity
fn sample_class_b_inputs(num_samples: usize) -> Inputs {
    let class_b_dist_first_dimension = Normal::new(CLASS_B_MEAN.0, CLASS_B_COV).unwrap();
    let class_b_dist_second_dimension = Normal::new(CLASS_B_MEAN.1, CLASS_B_COV).unwrap();

    let class_b_train_input_first_dimensions = Array::random(num_samples, class_b_dist_first_dimension);
    let class_b_train_input_second_dimensions = Array::random(num_samples, class_b_dist_second_dimension);

    stack![Axis(1), class_b_train_input_first_dimensions, class_b_train_input_second_dimensions]
}

// Samples class c samples (inputs that belong to class c) using a multivariate distribution with
// mean=(0,1) and covariance=Identity
fn sample_class_c_inputs(num_samples: usize) -> Inputs {
    let class_c_dist_first_dimension = Normal::new(CLASS_C_MEAN.0, CLASS_C_COV).unwrap();
    let class_c_dist_second_dimension = Normal::new(CLASS_C_MEAN.1, CLASS_C_COV).unwrap();

    let class_c_train_input_first_dimensions = Array::random(num_samples, class_c_dist_first_dimension);
    let class_c_train_input_second_dimensions = Array::random(num_samples, class_c_dist_second_dimension);

    stack![Axis(1), class_c_train_input_first_dimensions, class_c_train_input_second_dimensions]
}

/*
   Shuffles the given sets together:

        merge [9000, 2] and [9000, 3] into [(9000, 5)]

        merge [9000, 2] and [9000, 3] into: [9000, 5]
        shuffle [9000, 5] along Axis(0)
        [9000, 5] => [9000, 2] and [9000, 3]
*/
fn shuffle_input_output_sets(inputs: Outputs, truths: Outputs) -> (Outputs, Outputs) {
    let inputs_dim = inputs.dim();
    let truths_dim = truths.dim();
    let num_samples = inputs_dim.0;
    assert_eq!(inputs_dim.0, truths_dim.0);

    let mut unit_inputs = Vec::<Vec<f64>>::new();
    for (index, axis) in inputs.axis_iter(Axis(0)).enumerate() {
        let mut unit_inputs_item = Vec::<f64>::new();

        // append input dimensions
        for &val in axis.iter() {
            unit_inputs_item.push(val);
        }

        // append truth dimensions
        for &val in truths.slice(s![index, ..]).iter() {
            unit_inputs_item.push(val);
        }

        unit_inputs.push(unit_inputs_item);
    }

    assert_eq!(unit_inputs.len(), num_samples);

    let mut rng = thread_rng();
    unit_inputs.as_mut_slice().shuffle(&mut rng);
    let unit_inputs: Vec<f64> = unit_inputs.into_iter().flatten().collect();
    let inputs = Array::from_shape_vec((num_samples, inputs_dim.1 + truths_dim.1), unit_inputs).unwrap();

    let sample_view = inputs.slice(s![.., 0..2]);
    let truth_view = inputs.slice(s![.., 2..5]);

    assert_eq!(sample_view.dim(), inputs_dim);
    assert_eq!(truth_view.dim(), truths_dim);

    (sample_view.to_owned(), truth_view.to_owned())
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
