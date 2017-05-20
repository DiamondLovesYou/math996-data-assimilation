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
extern crate pbr;

use nd::{ArrayView, Array, Axis, Ix2, Ix3};

use std::marker::PhantomData;

pub mod data;
pub mod progress;

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
    where WS: na_df::ensemble::EnsembleWorkspace<f64>,
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

pub struct MeshGrid {
  pub grid: Array<f64, Ix3>,
  pub scale: (f64, f64),
  pub xmean: f64,
  pub ymean: f64,
}
impl MeshGrid {
  pub fn dim(&self) -> (usize, usize) {
    (self.grid.dim().1, self.grid.dim().2)
  }

  pub fn x_max(&self) -> f64 { self.dim().0 as f64 * self.scale.0 }
  pub fn y_max(&self) -> f64 { self.dim().1 as f64 * self.scale.1 }

  pub fn dx(&self) -> f64 { self.scale.0 }
  pub fn dy(&self) -> f64 { self.scale.1 }

  pub fn x_varying(&self) -> ArrayView<f64, Ix2> {
    self.grid.subview(Axis(0), 0)
  }
  pub fn y_varying(&self) -> ArrayView<f64, Ix2> {
    self.grid.subview(Axis(0), 1)
  }
}

pub fn meshgrid(x: usize, x_scale: f64, y: usize, y_scale: f64) -> MeshGrid {
  let mut grid = Array::zeros((2, x, y));

  {
    let gv = grid.view_mut();
    let (mut x_grid0, mut y_grid0) = gv.split_at(Axis(0), 1);
    let mut x_grid = x_grid0.subview_mut(Axis(0), 0);
    let mut y_grid = y_grid0.subview_mut(Axis(0), 0);
    for i in 0..x {
      for j in 0..y {
        x_grid[(i, j)] = (i as f64) * x_scale;
        y_grid[(i, j)] = (j as f64) * y_scale;
      }
    }
  }

  let xsum: usize = (0..x).sum();
  let ysum: usize = (0..y).sum();
  let xmean = (xsum as f64 * x_scale) / (x as f64);
  let ymean = (ysum as f64 * y_scale) / (y as f64);

  MeshGrid {
    grid: grid,
    scale: (x_scale, y_scale),
    xmean: xmean,
    ymean: ymean,
  }
}
