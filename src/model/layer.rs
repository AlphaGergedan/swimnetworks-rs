use ndarray::Array2;
use crate::{Layer, Model};

impl Layer for Model {
    fn forward(&self, inputs: Array2<f64>) -> Array2<f64> {
        //println!("FORWARDING MODEL WITH MEAN DENSE WEIGHTS: {:?}", self.linear1.weights.mean());
        let outputs = self.first_layer().forward(inputs);
        let outputs = self.first_activation().forward(outputs);
        //println!("FORWARDING MODEL WITH MEAN LINEAR WEIGHTS: {:?}", self.linear2.weights.mean());
        let outputs = self.last_layer().forward(outputs);

        outputs
    }
}
