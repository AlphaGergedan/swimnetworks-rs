use crate::Model;

pub trait Sample {
    // TODO: better would be to receive a list of mutable references to the weights and biases.
    // &mut Model makes the whole model to be mutable.
    fn sample(&self, model: &mut Model);
}
