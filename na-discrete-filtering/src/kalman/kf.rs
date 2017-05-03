
//! You probably don't want to use unmodified Kalman filters.
//! This is here because it was required for one of my Data
//! Assimilation homework problems.

use nd::{Array, ArrayBase, ArrayView, ArrayViewMut, Ix2, Ix1,
         Axis};
use nd::linalg::general_mat_mul;

use linxal::types::{LinxalImplScalar};
use linxal::solve_linear::general::SolveLinear;
use linxal::solve_linear::symmetric::SymmetricSolveLinear;
use linxal::eigenvalues::general::Eigen;
use num_traits::{NumCast, One, Zero, Float};
use num_complex::Complex;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use rand::distributions::normal::Normal;
use std::ops::{Add, Sub, Mul, Div,
               AddAssign, SubAssign,
               DivAssign, MulAssign,};

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