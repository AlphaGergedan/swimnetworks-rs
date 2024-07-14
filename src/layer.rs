use ndarray::Array2;

type Inputs = Array2<f64>;
type Outputs = Array2<f64>;

pub trait Layer {
    fn forward(&self, inputs: Inputs) -> Outputs;
}
