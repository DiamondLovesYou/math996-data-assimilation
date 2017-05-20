
extern crate ndarray as nd;
extern crate ndarray_parallel as nd_par;
extern crate rayon;
extern crate linxal;
extern crate num_traits;

use nd::prelude::*;
use nd::linalg::general_mat_vec_mul;
use nd::{Data, LinalgScalar, RemoveAxis};

use num_traits::{One, Zero};

pub use error::{Result, Error};

pub mod error;

pub trait Model<E>: Send + Sync {
  /// Will be called once.
  fn workspace_size(&self) -> usize;
  fn run_model(&self, step: u64,
               workspace: ArrayViewMut<E, Ix1>,
               mean: ArrayView<E, Ix1>,
               out: ArrayViewMut<E, Ix1>);
}
#[derive(Debug)]
pub struct ModelStats<M> {
  pub model: M,
  pub calls: u64,
}
impl<M> From<M> for ModelStats<M> {
  fn from(v: M) -> ModelStats<M> {
    ModelStats {
      model: v,
      calls: 0,
    }
  }
}

pub trait Operator<E, Dim>: Send + Sync
  where Dim: nd::Dimension,
{
  fn operator_input_dim(&self) -> usize;
  fn operator_output_dim(&self) -> usize;
  fn eval_at(&self, x: ArrayView<E, Dim>,
             out: ArrayViewMut<E, Dim>) -> Result<()>;
}
pub trait LinearizedOperator<E, Dim>: Operator<E, Dim::Smaller> + Send + Sync
  where Dim: nd::Dimension,
{
  fn eval_jacobian_at(&self,
                      x: ArrayView<E, Dim::Smaller>,
                      out: ArrayViewMut<E, Dim>) -> Result<()>;
}

impl<D, E> Operator<E, Ix1> for ArrayBase<D, Ix2>
  where D: nd::Data<Elem = E> + Send + Sync,
        E: LinalgScalar + One + Zero + Sync + Send,
{
  fn operator_input_dim(&self) -> usize { self.raw_dim().remove_axis(Axis(0))[0] }
  fn operator_output_dim(&self) -> usize { self.raw_dim().remove_axis(Axis(1))[0] }
  fn eval_at(&self, x: ArrayView<E, Ix1>,
             mut out: ArrayViewMut<E, Ix1>) -> Result<()> {
    general_mat_vec_mul(One::one(),
                        self, &x, Zero::zero(),
                        &mut out);

    Ok(())
  }
}
impl<D, E> LinearizedOperator<E, Ix2> for ArrayBase<D, Ix2>
  where D: Data<Elem = E> + Send + Sync,
        E: LinalgScalar + One + Send + Sync,
{
  fn eval_jacobian_at(&self, _: ArrayView<E, Ix1>,
                      mut out: ArrayViewMut<E, Ix2>) -> Result<()> {
    out.assign(self);
    Ok(())
  }
}