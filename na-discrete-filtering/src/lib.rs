
#![feature(associated_type_defaults)]

#[macro_use]
extern crate ndarray as nd;
extern crate linxal;
extern crate rand;
extern crate num_traits;
extern crate num_complex;
extern crate rayon;
extern crate ndarray_parallel as nd_par;
extern crate na_core as nac;

#[cfg(test)]
extern crate gnuplot;

use rand::Rng;
use nd::{ArrayViewMut, ArrayView, Ix1, Ix2, ArrayBase, Data};

pub use nac::error;
pub use nac::error::Result;
pub use nac::{Model, ModelStats, Operator, LinearizedOperator};


pub mod utils;
pub mod variational;
pub mod kalman;
pub mod ensemble;
pub mod forcing;
pub mod particle;

/// In the example programs provided in Data Assimilation:
/// A Mathematical Introduction, the algorithm assumes the
/// number of steps is known up front. It's possible to move
/// to a code that doesn't make this assumption, but for
/// simplicity I'm ignoring this. I'll eventually get to this.
///
/// I want to have a generic trait to allow users
/// to ask for an arbitrary number of extra predictor steps on
/// every loop.
///
/// Additionally, I've excluded any consideration of errors, ie
/// those generated by LAPACK under the hood. TODO.
///
/// ndarray doesn't expose interfaces for using permutation
/// matrices (LAPACK has the capability, of course). XXX
///
/// Ugh. Upon more thought, ndarray needs work (ie general
/// gemm panics if dims are wrong. the function in question
/// should instead return a Result). XXX


pub trait Workspace<I>: Sized
  where I: Initializer,
{
  fn alloc(i: I, rand: &mut Rng, total_steps: u64) -> Result<Self>;
}
pub trait Initializer { }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State<'a, E>
  where E: 'a,
{
  pub mean: ArrayView<'a, E, Ix1>,
  pub covariance: ArrayView<'a, E, Ix2>,
}

pub trait Algorithm<M, Ob>: Sized
{
  type Init: Initializer;
  type WS: Workspace<Self::Init>;

  /// observer should be unused. It's just here for type inference.
  fn init(i: &Self::Init,
          rand: &mut Rng,
          model: &mut ModelStats<M>,
          _observer: &Ob,
          steps: u64) -> Result<Self>;

  fn next_step(&self,
               current_step: u64,
               total_steps: u64,
               rand: &mut Rng,
               workspace: &mut Self::WS,
               model: &mut ModelStats<M>,
               observer: &Ob)
               -> Result<()>;
}

pub trait Observer<E, Dim>: Send + Sync
  where Dim: nd::Dimension,
{
  fn observe_into(&self, step: u64, out: ArrayViewMut<E, Dim>) -> bool;
}


pub struct SimpleObserver<D>(pub ArrayBase<D, Ix2>)
  where D: Data;
impl<D> Observer<D::Elem, Ix1> for SimpleObserver<D>
  where D: Data + Send + Sync,
        D::Elem: Clone,
{
  fn observe_into(&self, idx: u64,
                  mut out: ArrayViewMut<D::Elem, Ix1>) -> bool {
    out.assign(&self.0.subview(nd::Axis(0),
                               idx as usize));
    true
  }
}
