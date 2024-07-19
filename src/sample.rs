use crate::Model;

/// [`Sample`] trait includes [`Sample::sample`] that should update the hidden layer weights
/// and biases in the given model using a sampling method.
pub trait Sample {
    fn sample(&self, model: &mut Model);
}
