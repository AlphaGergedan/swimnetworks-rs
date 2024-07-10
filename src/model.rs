use ndarray::{s, Array, Array1, Array2, ArrayView, ArrayView2, Axis, Dim, Ix1, Ix2, IxDyn};
use ndarray_linalg::{norm::NormalizeAxis, normalize, Norm};
use ndarray_rand::{rand::distributions::Slice, rand_distr::{uniform::{UniformInt, UniformSampler}, Normal, Uniform}, RandomExt};

pub struct ModelConfig {
    layer_width: usize,
    input_size: usize,
    output_size: usize,
    activation: Activation,
}

impl ModelConfig {
    fn new(self) -> Model {
        Model {
            linear1: Linear {
                // of shape (input_size, layer_width), because we want to have data matrix
                // X mult by weights for the forward pass
                weights: Array::zeros((self.layer_width, self.input_size)).reversed_axes(),
                biases: Array::zeros((1, self.layer_width)),
            },
            activation: self.activation,
            linear2: Linear {
                weights: Array::zeros((self.output_size, self.layer_width)).reversed_axes(),
                biases: Array::zeros((1, self.output_size)),
            },
            i_input_from: None,
            i_input_to: None,
        }
    }
}

pub struct Model {
    linear1: Linear,
    activation: Activation,
    linear2: Linear,
    i_input_from: Option<Array1<usize>>,
    i_input_to: Option<Array1<usize>>,

}

type Indices = Array1<usize>;
type Distances = Array1<f64>;
type NormalizedDirections = Array2<f64>;

/// Methods for gathering model information.
impl Model {
    fn layer_width(&self) -> usize {
        let (_, layer_width) = self.linear1.weights.dim();
        layer_width
    }

    fn input_size(&self) -> usize {
        let (input_size, _) = self.linear1.weights.dim();
        input_size
    }

    fn output_size(&self) -> usize {
        let (_, output_size) = self.linear2.weights.dim();
        output_size
    }

    fn number_of_layers(&self) -> usize {
        // TODO
        2
    }

    fn number_of_weights(&self) -> usize {
        let dense_weights = self.linear1.weights.shape().iter().fold(0, |acc, dim_size| acc + dim_size);
        let linear_weights = self.linear2.weights.shape().iter().fold(0, |acc, dim_size| acc + dim_size);
        dense_weights + linear_weights
    }

    fn number_of_biases(&self) -> usize {
        let dense_biases = self.linear1.biases.shape().iter().fold(0, |acc, dim_size| acc + dim_size);
        let linear_biases = self.linear2.biases.shape().iter().fold(0, |acc, dim_size| acc + dim_size);
        dense_biases + linear_biases
    }

    fn number_of_params(&self) -> usize {
        self.number_of_weights() + self.number_of_biases()
    }
}

/// Methods for sampling the model parameters using random features.
impl Model {
    fn sample_dense_layer_weights_normally(&mut self, mean: f64, std_dev: f64) {
        let weights = Array::<f64, _>::random(self.linear1.weights.dim(), Normal::new(mean, std_dev).unwrap());
        self.linear1.weights = weights;
    }

    fn sample_dense_layer_biases_uniformly(&mut self, min: f64, max: f64) {
        let biases = Array::<f64, _>::random(self.linear1.biases.dim(), Uniform::new(min, max));
        self.linear1.biases = biases;
    }
}


#[derive(Clone)]
enum Activation {
    Relu, Tanh
}

struct Linear {
    weights: Array2<f64>,
    biases: Array2<f64>,
}

// forward backward implementations

enum BackwardWithRespectTo {
    Weights,
    Inputs,
}

trait Layer {
    fn forward(&self, input: Array1<f64>) -> Array1<f64>;
}

trait Differentiable {
    fn backward(&self, output: Array1<f64>, wrt: BackwardWithRespectTo) -> Array1<f64>;
}

//impl Differentiable for Activation {
    ////fn backward(&self) { todo!() }
//}

//impl Layer for Activation {
    //fn forward(&self, input: Array1<f64>) {
        ////match self {
            ////Activation::Relu => ,
            ////Activation::Tanh => todo!(),
        ////}
    //}
//}
