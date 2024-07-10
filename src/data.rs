use ndarray::Array2;
use ndarray_linalg::Norm;

type Input = f64;
type Output = f64;

pub fn f_poly_deg_1(x: Input) -> Output {
    12. + 2.23 * x
}

pub fn f_poly_deg_2(x: Input) -> Output {
    -100. - 10. * x + 2. * x.powf(2.)
}

pub fn f_poly_deg_3(x: Input) -> Output {
    300. + 230. * x - 12. * x.powf(2.) + 80. * x.powf(3.)
}

pub fn f_poly_deg_4(x: Input) -> Output {
    300. + 230. * x - 12. * x.powf(2.) + 80. * x.powf(3.) - 120. * x.powf(4.)
}

pub fn f_poly_deg_7(x: Input) -> Output {
    0.252 + 0.12 * x - 12.5 * x.powf(2.) + 10. * x.powf(3.) - 0.5 * x.powf(4.) + 0.25 * x.powf(5.) - 0.9 * x.powf(6.) + 1. * x.powf(7.)
}

pub fn l2_error(input: Array2<f64>, output: Array2<f64>) -> f64 {
    (output - input).norm()
}
