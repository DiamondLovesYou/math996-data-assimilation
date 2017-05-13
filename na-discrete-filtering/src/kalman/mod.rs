
use linxal::types::{LinxalImplScalar};
use nd::prelude::*;

pub use super::utils::{extend_dim_mut, extend_dim_ref};

pub mod etkf;
pub mod kf;

pub struct Model<E, F1, F2>
  where F1: for<'r, 's> Fn(u64, ArrayView<'r, E, Ix1>, ArrayViewMut<'s, E, Ix1>),
        F2: FnMut(u64) -> Option<Array<E, Ix1>>,
        E: LinxalImplScalar,
{
  /// Don't modify this; safe to read though.
  pub calls: u64,
  pub model: F1,

  /// Will be called at most once per iteration.
  pub next_observation: F2,
}

impl<E, F1, F2> Model<E, F1, F2>
  where F1: for<'r, 's> Fn(u64, ArrayView<'r, E, Ix1>, ArrayViewMut<'s, E, Ix1>),
        F2: FnMut(u64) -> Option<Array<E, Ix1>>,
        E: LinxalImplScalar,
{
  pub fn new(model: F1, observation: F2) -> Model<E, F1, F2> {
    Model {
      calls: 0,
      model: model,
      next_observation: observation,
    }
  }
}
