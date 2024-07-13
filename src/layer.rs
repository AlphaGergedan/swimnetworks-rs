use ndarray::Array2;

type Input = Array2<f64>;
type Output = Array2<f64>;

pub trait Layer {
    fn forward(&self, inputs: Input) -> Output;
}
