#![feature(associated_consts)]
#![feature(step_by)]

extern crate na_discrete_filtering as na_df;
extern crate rgsl as gsl;
#[macro_use]
extern crate ndarray as nd;
extern crate ndarray_parallel as nd_par;
extern crate rayon;
extern crate roots;
extern crate num_traits;
extern crate num_complex;
extern crate arrayfire as af;
extern crate util;
extern crate gnuplot;
extern crate plot_helper;
extern crate ordered_float;

use ordered_float::OrderedFloat;

use num_complex::Complex;
use nd::prelude::*;
use nd_par::prelude::*;

use gnuplot::{Figure, PlotOption};

mod lax_wendroff;

use na_df::Model;

use plot_helper::xdmf::Xdmf;

type C64 = Complex<f64>;

const INITIALLY_GEOSTROPHIC: bool = true;

fn max<T: PartialOrd>(a: T, b: T) -> T { if a > b { a } else { b } }
fn min<T: PartialOrd>(a: T, b: T) -> T { if a < b { a } else { b } }

pub fn test_l95() {

}

pub fn main() {
  use lax_wendroff::{SweModel, STD_GRID_SIZE, STD_DX_DY, STD_DT};

  //test_l95();
  //return;

  const FORECAST_LENGTH_DAYS: f64 = 2.0;
  const FORECAST_STEPS: u64 = (FORECAST_LENGTH_DAYS * 24.0 * 3600.0 / STD_DT) as u64 + 1;
  const CHECKPOINT_INTERVAL: u64 = 60;

  let grid_size = STD_GRID_SIZE;
  let (dx, dy) = STD_DX_DY;
  let mesh = util::meshgrid(grid_size.0, dx,
                            grid_size.1, dy);

  let initer = lax_wendroff::HeightFieldInit::ZonalJet {
    mean_height: 10000.0,
  };
  let orography = lax_wendroff::Orography::GaussianMountain {
    std_dev: (5.0 * dx, 5.0 * dy),
    scale: 4000.0,
  };

  let swe = SweModel::new(&mesh, orography,
                          None, None, Some(grid_size),
                          None, Some((dx, dy)), None);
  let ihf = swe.initial_height_field(initer, &mesh);
  let caph = swe.orography();

  const U_IDX: usize = 0;
  const V_IDX: usize = 1;
  const H_IDX: usize = 2;
  let mut state = Array::zeros((2, 3, grid_size.0, grid_size.1));
  {
    let mut state_snapshot = state.subview_mut(Axis(0), 0);
    let (mut u0, rest) = state_snapshot.view_mut().split_at(Axis(0), 1);
    let (mut v0, rest) = rest.split_at(Axis(0), 1);
    let (mut h0, _) = rest.split_at(Axis(0), 1);

    let mut u = u0.subview_mut(Axis(0), 0);
    let mut v = v0.subview_mut(Axis(0), 0);
    let mut h = h0.subview_mut(Axis(0), 0);

    if INITIALLY_GEOSTROPHIC {
      let ycrop = s![.., 1..grid_size.1 as isize - 1];
      let xcrop = s![1..grid_size.0 as isize - 1, ..];

      {
        let mut ucrop = u.slice_mut(ycrop);
        ucrop.assign(&ihf.slice(s![.., 2..]));
        ucrop.scaled_add(-1.0, &ihf.slice(s![.., ..ihf.dim().1 as isize - 2]));
        ucrop *= -0.5 * swe.gravity() / dx;
        ucrop /= &swe.coriolis_parameters.slice(s![.., 1..swe.coriolis_parameters.dim().1 as isize - 1]);

        let mut vcrop = v.slice_mut(xcrop);
        vcrop.assign(&ihf.slice(s![2.., ..]));
        vcrop.scaled_add(-1.0, &ihf.slice(s![..ihf.dim().0 as isize - 2, ..]));
        vcrop *= -0.5 * swe.gravity() / dy;
        vcrop /= &swe.coriolis_parameters.slice(s![1..swe.coriolis_parameters.dim().0 as isize - 1, ..]);
      }

      {
        let (mut first, rest) = u.view_mut().split_at(Axis(0), 1);
        let rest_dim = rest.dim();
        let (middle, mut last) = rest.split_at(Axis(0), rest_dim.0 - 1);
        first.assign(&middle.subview(Axis(0), middle.dim().0 - 1));
        last.assign(&middle.subview(Axis(0), 0));
      }

      v.subview_mut(Axis(1), 0)
        .fill(0.0);
      v.subview_mut(Axis(1), grid_size.1 - 1)
        .fill(0.0);

      const MAX_WIND: f64 = 200.0;
      u.mapv_inplace(|v| {
        let v = min(v, MAX_WIND);
        max(v, -MAX_WIND)
      });
      v.mapv_inplace(|v| {
        let v = min(v, MAX_WIND);
        max(v, -MAX_WIND)
      });
    }

    h.assign(&ihf);
    h.scaled_add(-1.0, &caph);
  }

  let mut xdmf = Xdmf::new("swe-lax-wendroff-model-test");

  let mut ws = Array::zeros(swe.workspace_size());
  for i in (0..FORECAST_STEPS).step_by(CHECKPOINT_INTERVAL) {

    {
      let s = state.subview(Axis(0), 0);
      let s0 = s.subview(Axis(0), H_IDX);
      xdmf.next_timestep(i, &mesh, s0);

      let u = s.subview(Axis(0), U_IDX);
      let flat_u = u.into_shape(u.dim().0 * u.dim().1).unwrap();
      let v = s.subview(Axis(0), V_IDX);
      let flat_v = v.into_shape(v.dim().0 * v.dim().1).unwrap();

      let maxu = flat_u.axis_iter(Axis(0)).into_par_iter()
        .zip(flat_v.axis_iter(Axis(0)).into_par_iter())
        .map(|(u, v)| {
          OrderedFloat(u[()] * u[()] + v[()] * v[()])
        })
        .max()
        .unwrap()
        .into_inner();

      println!("Time = {}, max(|u|) = {}",
               (i as f64 * STD_DT / 3600.0) as u64,
               maxu);
    }


    for j in 0..CHECKPOINT_INTERVAL + 1 {
      let split = state.view_mut().split_at(Axis(0), 1);
      let (current_state, next_state) = split;

      let nsl = next_state.len();
      let mut next_state = next_state.into_shape(nsl)
        .unwrap();
      let csl = current_state.len();
      let mut current_state = current_state.into_shape(csl)
        .unwrap();

      next_state.fill(0.0);
      swe.run_model(i, ws.view_mut(), current_state.view(),
                    next_state.view_mut());

      current_state.assign(&next_state);
    }
  }


  let mut fig = Figure::new();
  fig.set_terminal("wxt", "");

  {
    let mut a = fig.axes3d();

    let plot_dims = Some((0.0, 0.0, mesh.x_max(), mesh.y_max()));
    let h0 = state.subview(Axis(0), 0);
    let h = h0.subview(Axis(0), H_IDX);
    a.surface(h.iter(), h.dim().0, h.dim().1, plot_dims,
              &[PlotOption::Caption("height"),]);
  }

  fig.show();
}