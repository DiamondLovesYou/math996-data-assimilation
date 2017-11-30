
extern crate ndarray as nd;
extern crate na_discrete_filtering as na_df;

use nd::prelude::*;

#[derive(Debug, Clone)]
pub struct L63Model {
  pub rho: f64,
  pub sigma: f64,
  pub beta: f64,
  pub noise_variance: Option<Array<f64, Ix1>>,
}
impl na_df::Model<f64> for L63Model {
  fn workspace_size(&self) -> usize { 0 }
  fn run_model(&self, _step: u64,
               _: ArrayViewMut<f64, Ix1>,
               y: ArrayView<f64, Ix1>,
               mut yp: ArrayViewMut<f64, Ix1>) {
    debug_assert!(y.len() == 3);
    debug_assert!(yp.len() == 3);

    yp[0] = self.sigma * (y[1] - y[0]);
    yp[1] = -self.sigma * y[0] - y[1] - y[0] * y[2];
    yp[2] = y[0] * y[1] - self.beta * y[2] - self.beta * (self.rho + self.sigma);
  }
}
impl Default for L63Model {
  fn default() -> Self {
    L63Model {
      rho: 28.0,
      sigma: 10.0,
      beta: 8.0 / 3.0,
      noise_variance: None,
    }
  }
}
