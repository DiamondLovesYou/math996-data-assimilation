
//! You probably don't want to use unmodified Kalman filters.
//! This is here because it was required for one of my Data
//! Assimilation homework problems.

use nd::{Array, ArrayView, Ix2, Ix1,};

use linxal::types::{LinxalImplScalar};

pub struct Init<'a, E>
  where E: LinxalImplScalar,
{
  pub initial_mean: ArrayView<'a, E, Ix1>,
  pub initial_covariance: ArrayView<'a, E, Ix2>,
  pub observation_operator: ArrayView<'a, E, Ix2>,
  pub gamma: ArrayView<'a, E::RealPart, Ix1>,
  pub sigma: ArrayView<'a, E::RealPart, Ix1>,
}
impl<'a, E> ::Initializer for Init<'a, E>
  where E: LinxalImplScalar,
{ }

#[derive(Debug)]
pub struct Workspace<E>
  where E: LinxalImplScalar,
{
  mean: Array<E, Ix1>,
  covariance: Array<E, Ix2>,

  estimator_predict: Array<E, Ix2>,
  covariance_predict: Array<E, Ix2>,
}