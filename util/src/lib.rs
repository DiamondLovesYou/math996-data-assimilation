#![feature(unboxed_closures)]
#![feature(fn_traits)]

extern crate ndarray as nd;
extern crate ndarray_parallel as nd_par;
extern crate na_discrete_filtering as na_df;
extern crate na_quadrature as na_q;
extern crate rand;
extern crate rayon;
extern crate num_traits;
extern crate linxal;

use nd::{ArrayView, Array, Axis, Ix2, Ix3};

use std::marker::PhantomData;

pub mod data;

pub trait ModelTruth<E> {
  fn truth(&self) -> ArrayView<E, Ix2>;
  fn observations(&self) -> ArrayView<E, Ix2>;
}

#[derive(Clone, Debug)]
pub struct StateSteps {
  pub means: Array<f64, Ix2>,
  pub covariances: Array<f64, Ix3>,
  pub ensembles: Array<f64, Ix3>,
}
impl StateSteps {
  pub fn new(steps: usize, ensemble_count: usize, n: usize) -> StateSteps {
    StateSteps {
      means: Array::zeros((steps, n)),
      covariances: Array::zeros((steps, n, n)),
      ensembles: Array::zeros((steps, ensemble_count, n)),
    }
  }

  pub fn store_state<WS>(&mut self, step: usize, ws: &WS)
    where WS: na_df::kalman::EnsembleWorkspace<f64>,
  {
    self.means
      .subview_mut(Axis(0), step)
      .assign(&ws.mean_view());
    self.covariances
      .subview_mut(Axis(0), step)
      .assign(&ws.covariance_view());
    self.ensembles
      .subview_mut(Axis(0), step)
      .assign(&ws.ensembles_view());
  }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Lambda<T, Args, Ret>(pub T, PhantomData<(Args, Ret)>)
  where T: LambdaObject<Args, Ret>;
impl<T, Args, Ret> From<T> for Lambda<T, Args, Ret>
  where T: LambdaObject<Args, Ret>,
{
  fn from(f: T) -> Lambda<T, Args, Ret> { Lambda(f, PhantomData) }
}
pub trait LambdaObject<Args, Ret>: Into<Lambda<Self, Args, Ret>> {
  fn run_lambda(&self, args: Args) -> Ret;
}

impl<T, Args, Ret> FnOnce<Args> for Lambda<T, Args, Ret>
  where T: LambdaObject<Args, Ret>,
{
  type Output = Ret;
  extern "rust-call" fn call_once(self, a: Args) -> Ret {
    self.0.run_lambda(a)
  }
}
impl<T, Args, Ret> FnMut<Args> for Lambda<T, Args, Ret>
  where T: LambdaObject<Args, Ret>,
{
  extern "rust-call" fn call_mut(&mut self, a: Args) -> Ret {
    self.0.run_lambda(a)
  }
}
impl<T, Args, Ret> Fn<Args> for Lambda<T, Args, Ret>
  where T: LambdaObject<Args, Ret>,
{
  extern "rust-call" fn call(&self, a: Args) -> Ret {
    self.0.run_lambda(a)
  }
}