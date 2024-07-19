use ndarray::Array2;

type Inputs = Array2<f64>;
type Outputs = Array2<f64>;

/// [`Layer`] trait includes [`Layer::forward`] that should forward the given inputs
/// through, applying operations like activation, linear mapping etc.
pub trait Layer {
    fn forward(&self, inputs: Inputs) -> Outputs;
}
