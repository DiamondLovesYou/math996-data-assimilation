
use num_traits::Float;

use nd::{Array, ArrayView, ArrayViewMut, Ix1, Ix2, Ix3, Axis};
use nd::{self, Zip, aview1, aview2};
use nd_par::prelude::*;
use rayon::prelude::*;

use na_df;

use util::MeshGrid;

const A0: Axis = Axis(0);
const A1: Axis = Axis(1);

fn par_scaled_add(dest: ArrayViewMut<f64, Ix2>,
                  factor: f64, added: ArrayView<f64, Ix2>) {
  par_scaled_add_impl(dest, factor, added, false)
}
fn par_scaled_add_sq(dest: ArrayViewMut<f64, Ix2>,
                  factor: f64, added: ArrayView<f64, Ix2>) {
  par_scaled_add_impl(dest, factor, added, true)
}

fn par_scaled_add_impl(mut dest: ArrayViewMut<f64, Ix2>,
                       factor: f64, added: ArrayView<f64, Ix2>,
                       sq_elem_wise_to_add: bool) {
  if sq_elem_wise_to_add {
    Zip::from(&mut dest)
      .and(&added)
      .par_apply(|mut dest, to_add| {
        *dest += factor * to_add * to_add;
      });
  } else {
    Zip::from(&mut dest)
      .and(&added)
      .par_apply(|mut dest, to_add| {
        *dest += factor * to_add;
      });
  }
}

fn par_multi_scaled_multi_add(mut dest: ArrayViewMut<f64, Ix2>,
                              factors: ArrayView<f64, Ix1>,
                              add: &[ArrayView<f64, Ix2>]) {
  debug_assert!(factors.len() == add.len());
  // xxx
  for (factor, add) in factors.axis_iter(A0).zip(add.iter()) {
    Zip::from(&mut dest)
      .and(add)
      .par_apply(|dest, &add| {
        *dest = factor[()] * add;
      });
  }
}

#[derive(Debug)]
pub struct SweModel {
  pub grid_size: (usize, usize),

  pub coriolis_parameter: f64,
  pub coriolis_parameters: Array<f64, Ix2>,
  pub orography: Array<f64, Ix2>,

  pub gravity: f64,

  pub dt: f64,
  pub zonal_grid_spacing: f64,
  pub meridional_grid_spacing: f64,
}

pub const STD_GRID_SIZE: (usize, usize) = (254, 50);
pub const STD_DX_DY: (f64, f64) = (100_000.0, 100_000.0);
pub const STD_DT: f64 = 60.0;

pub enum HeightFieldInit {
  UniformWesterly {
    mean_wind_speed: f64,
    mean_height: f64,
  },
  ZonalJet {
    mean_height: f64,
  }
}
pub enum Orography {
  GaussianMountain {
    std_dev: (f64, f64),
    scale: f64,
  }
}

impl SweModel {
  pub fn new(mesh: &MeshGrid,
             orography: Orography,
             coriolis_parameter: Option<f64>,
             beta: Option<f64>,
             grid_dim: Option<(usize, usize)>,
             dt: Option<f64>,
             dx_dy: Option<(f64, f64)>,
             gravity: Option<f64>) -> SweModel {
    let coriolis_parameter = coriolis_parameter.unwrap_or(0.0001);
    let beta = beta.unwrap_or(1.6 * (10.0).powi(-11));
    let grid_dim = grid_dim.unwrap_or(STD_GRID_SIZE);
    let dt = dt.unwrap_or(STD_DT);
    let (dx, dy) = dx_dy.unwrap_or(STD_DX_DY);
    let gravity = gravity.unwrap_or(9.81);

    let mut coriolis_parameters = Array::zeros(mesh.dim());
    coriolis_parameters.fill(coriolis_parameter);
    coriolis_parameters.scaled_add(beta, &mesh.y_varying());
    let means_ = [[-beta * mesh.ymean; 1]; 1];
    let means = aview2(&means_[..]);
    coriolis_parameters += &means;

    let orography = match orography {
      Orography::GaussianMountain {
        std_dev: (stdmx, stdmy),
        scale,
      } => {
        let mut h = &mesh.x_varying() - &aview2(&[[mesh.xmean]]);
        h /= stdmx;
        h *= &h.to_owned();

        let mut t2 = &mesh.y_varying() - &aview2(&[[mesh.ymean]]);
        t2 /= stdmy;
        t2 *= &t2.to_owned();

        h *= -0.5;
        h.scaled_add(-0.5, &t2);

        h.par_mapv_inplace(f64::exp);
        h *= scale;

        h
      },
    };

    SweModel {
      grid_size: grid_dim,
      coriolis_parameter: coriolis_parameter,
      coriolis_parameters: coriolis_parameters,
      orography: orography,
      gravity: gravity,
      dt: dt,
      zonal_grid_spacing: dx,
      meridional_grid_spacing: dy,
    }
  }

  pub fn checkpoint_step_feq(&self, interval_mins: f64) -> u64 {
    let output_interval = interval_mins * 60.0;
    (output_interval / self.dt()) as u64
  }
  pub fn total_steps(&self, forcast_length_days: f64) -> u64 {
    let forecast_length = forcast_length_days * 24.0 * 3600.0;
    (forecast_length / self.dt()) as u64
  }
  pub fn grid_dims(&self) -> (usize, usize) { self.grid_size }

  pub fn orography(&self) -> ArrayView<f64, Ix2> {
    self.orography.view()
  }

  pub fn initial_height_field(&self, initer: HeightFieldInit,
                              mesh: &MeshGrid) -> Array<f64, Ix2> {
    let mut h = Array::zeros(mesh.dim());

    match initer {
      HeightFieldInit::UniformWesterly {
        mean_wind_speed, mean_height,
      } => {
        h.assign(&mesh.y_varying());
        h -= &aview2(&[[mesh.ymean]]);
        h *= -(mean_wind_speed * self.coriolis_parameter / self.gravity);
        h += &aview2(&[[10000.0]]);
      },
      HeightFieldInit::ZonalJet {
        mean_height,
      } => {
        h.assign(&mesh.y_varying());
        h.par_mapv_inplace(|v| {
          let v = 20.0 * ((v - mesh.ymean) / mesh.y_max());
          mean_height - v.tanh() * 400.0
        });
      }
    }

    h
  }

  pub fn gravity(&self) -> f64 { self.gravity }

  pub fn dt(&self) -> f64 { self.dt }
  pub fn dx(&self) -> f64 { self.zonal_grid_spacing }
  pub fn dy(&self) -> f64 { self.meridional_grid_spacing }

  fn state_shape(&self) -> (usize, usize, usize) {
    (3, self.grid_size.0, self.grid_size.1)
  }
  fn workspace_shape(&self) -> (usize, usize, usize) {
    (2, self.grid_size.0 - 2, self.grid_size.1 - 2)
  }

  fn advance(&self, u: ArrayView<f64, Ix2>,
             v: ArrayView<f64, Ix2>,
             h: ArrayView<f64, Ix2>,
             u_accel: ArrayView<f64, Ix2>,
             v_accel: ArrayView<f64, Ix2>,

             mut next_u: ArrayViewMut<f64, Ix2>,
             mut next_v: ArrayViewMut<f64, Ix2>,
             mut next_h: ArrayViewMut<f64, Ix2>)
  {
    let uh = &u * &h;
    let vh = &v * &h;

    let mut capu_x = &uh * &u;
    par_scaled_add_sq(capu_x.view_mut(), 0.5 * self.gravity, h.view());
    let capu_x = capu_x;
    let capu_y = &uh * &v;

    let capv_x = capu_y.view();
    let mut capv_y = &vh * &v;
    par_scaled_add_sq(capv_y.view_mut(), 0.5 * self.gravity, h.view());
    let capv_y = capv_y;

    macro_rules! lsv {
      ($idx:expr, $h:expr) => {
        $h.slice(&match $idx {
          0 | 1 => *s![1..$h.dim().0 as isize, ..],
          2 | 3 => *s![.., 1..$h.dim().1 as isize],
          _ => unreachable!(),
        })
      }
    }
    macro_rules! rsv {
      ($idx:expr, $h:expr) => {
        $h.slice(&match $idx {
          0 | 1 => *s![..$h.dim().0 as isize - 1, ..],
          2 | 3 => *s![.., ..$h.dim().1 as isize - 1],
          _ => unreachable!(),
        })
      }
    }

    fn calculate_midpoint_pair(this: &SweModel,
                               nonlinear_term: ArrayView<f64, Ix2>,
                               jacobi0: ArrayView<f64, Ix2>,
                               jacobi1: ArrayView<f64, Ix2>)
      -> [Array<f64, Ix2>; 2]
    {
      let fx = -0.5 * this.dt() / this.dx();
      let fy = -0.5 * this.dt() / this.dy();
      let factorsx = [
        0.5, 0.5,
        fx, -fx,
      ];
      let factorsy = [
        0.5, 0.5,
        fy, -fy,
      ];
      let factors = [
        factorsx, factorsy,
      ];
      let factors = aview2(&factors[..]);

      let lhsx = &[
        lsv!(0, nonlinear_term), rsv!(0, nonlinear_term),
        lsv!(1, jacobi0),        rsv!(1, jacobi0),
      ];
      let lhsy = &[
        lsv!(2, nonlinear_term), rsv!(2, nonlinear_term),
        lsv!(3, jacobi1),        rsv!(3, jacobi1),
      ];

      let lhss = [
        lhsx, lhsy,
      ];

      let dim = nonlinear_term.dim();
      let mid_x = Array::zeros((dim.0 - 1, dim.1));
      let mid_y = Array::zeros((dim.0, dim.1 - 1));

      let mut mid = [
        mid_x,
        mid_y,
      ];
      mid.par_iter_mut()
        .enumerate()
        .zip(factors.axis_iter(A0).into_par_iter())
        .for_each(|((idx, mut mid), factors)| {
          par_multi_scaled_multi_add(mid.view_mut(),
                                     factors,
                                     lhss[idx]);
        });

      mid
    }

    let h_mid = calculate_midpoint_pair(self,
                                        h.view(),
                                        uh.view(),
                                        vh.view());
    let uh_mid = calculate_midpoint_pair(self,
                                         uh.view(),
                                         capu_x.view(),
                                         capu_y.view());
    let vh_mid = calculate_midpoint_pair(self,
                                         vh.view(),
                                         capv_x.view(),
                                         capv_y.view());
    let (h_mid_xt, h_mid_yt) = (&h_mid[0], &h_mid[1]);
    let (uh_mid_xt, uh_mid_yt) = (&uh_mid[0], &uh_mid[1]);
    let (vh_mid_xt, vh_mid_yt) = (&vh_mid[0], &vh_mid[1]);

    #[inline(always)]
    fn indices((xend, yend): (usize, usize)) -> [[nd::Si; 2]; 4] {
      let xend = xend as isize;
      let yend = yend as isize;
      let idxs: [_; 4] = [
        *s![1..xend, 1..yend - 1], *s![..xend - 1, 1..yend - 1],
        *s![1..xend - 1, 1..yend], *s![1..xend - 1, ..yend - 1],
      ];

      idxs
    }

    let xend = self.grid_size.0 as isize;
    let yend = self.grid_size.1 as isize;
    let crop = s![1..xend - 1, 1..yend - 1];
    let old_h_cropped = h.slice(crop);
    next_h.assign(&old_h_cropped);

    let dt = self.dt();
    let dtdx = self.dt() / self.dx();
    let dtdy = self.dt() / self.dy();

    let idxs = indices(uh_mid_xt.dim());
    par_scaled_add(next_h.view_mut(), -dtdx, uh_mid_xt.slice(&idxs[0]));
    par_scaled_add(next_h.view_mut(), dtdx, uh_mid_xt.slice(&idxs[1]));
    let idxs = indices(vh_mid_yt.dim());
    par_scaled_add(next_h.view_mut(), -dtdy, vh_mid_yt.slice(&idxs[2]));
    par_scaled_add(next_h.view_mut(), dtdy, vh_mid_yt.slice(&idxs[3]));

    let mut capu_x_mid_xt = uh_mid_xt * uh_mid_xt;
    capu_x_mid_xt /= h_mid_xt;
    par_scaled_add_sq(capu_x_mid_xt.view_mut(), 0.5 * self.gravity,
                      h_mid_xt.view());
    let capu_x_mid_xt = capu_x_mid_xt;

    let mut capu_y_mid_yt = uh_mid_yt * vh_mid_yt;
    capu_y_mid_yt /= h_mid_yt;
    let capu_y_mid_yt = capu_y_mid_yt;

    let xidxs = indices(capu_x_mid_xt.dim());
    let yidxs = indices(capu_y_mid_yt.dim());

    next_u.assign(&uh.slice(crop));
    let mut next_uh = next_u.view_mut();
    let fs: [f64; 4] = [-dtdx, dtdx, -dtdy, dtdy,];
    let factors = aview1(&fs[..]);
    par_multi_scaled_multi_add(next_uh.view_mut(),
                               factors,
                               &[capu_x_mid_xt.slice(&xidxs[0]),
                                 capu_x_mid_xt.slice(&xidxs[1]),
                                 capu_y_mid_yt.slice(&yidxs[2]),
                                 capu_y_mid_yt.slice(&yidxs[3])]);
    Zip::from(&mut next_uh)
      .and(&u_accel)
      .and(&old_h_cropped)
      .and(&next_h)
      .par_apply(|next_uh, &u_accel, &old_h_cropped, &next_h| {
        *next_uh += dt * 0.5 * u_accel * (old_h_cropped + next_h);
        *next_uh /= next_h;
      });

    let mut capv_x_mid_xt = uh_mid_xt * vh_mid_xt;
    capv_x_mid_xt /= h_mid_xt;
    let capv_x_mid_xt = capv_x_mid_xt;

    let mut capv_y_mid_yt = vh_mid_yt * vh_mid_yt;
    capv_y_mid_yt /= h_mid_yt;
    par_scaled_add_sq(capv_y_mid_yt.view_mut(), 0.5 * self.gravity,
                      h_mid_yt.view());
    let capv_y_mid_yt = capv_y_mid_yt;

    let xidxs = indices(capv_x_mid_xt.dim());
    let yidxs = indices(capv_y_mid_yt.dim());

    next_v.assign(&vh.slice(crop));
    let mut next_vh = next_v.view_mut();
    par_multi_scaled_multi_add(next_uh.view_mut(),
                               factors,
                               &[capv_x_mid_xt.slice(&xidxs[0]),
                                 capv_x_mid_xt.slice(&xidxs[1]),
                                 capv_y_mid_yt.slice(&yidxs[2]),
                                 capv_y_mid_yt.slice(&yidxs[3])]);
    Zip::from(&mut next_vh)
      .and(&v_accel)
      .and(&old_h_cropped)
      .and(&next_h)
      .par_apply(|next_vh, &v_accel, &old_h_cropped, &next_h| {
        *next_vh += dt * 0.5 * v_accel * (old_h_cropped + next_h);
        *next_vh /= next_h;
      });

  }
}

impl na_df::Model<f64> for SweModel {
  fn workspace_size(&self) -> usize {
    let shape = self.workspace_shape();
    shape.0 * shape.1 * shape.2
  }

  fn run_model(&self, step: u64,
               mut flat_ws: ArrayViewMut<f64, Ix1>,
               flat_state: ArrayView<f64, Ix1>,
               mut flat_out: ArrayViewMut<f64, Ix1>) {

    let grid = flat_state.into_shape(self.state_shape())
      .expect("failed to reshape flat state into Ix3");
    let mut out = flat_out.into_shape(self.state_shape())
      .expect("failed to reshape flat output state into Ix3");
    let ws = flat_ws.into_shape(self.workspace_shape())
      .expect("failed to reshape flat workspace into Ix3");

    if step != 0 { panic!(); }

    let (u, v, h) = {
      let (u, rest) = grid.view().split_at(A0, 1);
      let (v, h)    = rest.split_at(A0, 1);

      (u.into_subview(A0, 0),
       v.into_subview(A0, 0),
       h.into_subview(A0, 0))
    };
    let (mut next_u, mut next_v, mut next_h) = {
      let (u, rest) = out.view_mut().split_at(A0, 1);
      let (v, h)    = rest.split_at(A0, 1);

      (u.into_subview(A0, 0),
       v.into_subview(A0, 0),
       h.into_subview(A0, 0))
    };

    let bd_idx = s![1..self.grid_size.0 as isize - 1,
                    1..self.grid_size.1 as isize - 1];
    let cps_bd = self.coriolis_parameters
      .slice(bd_idx);
    let u_bd = u.slice(bd_idx);
    let v_bd = v.slice(bd_idx);

    println!("u[(0,0)] = {:.16e}", u_bd[(0,0)]);
    println!("v[(0,0)] = {:.16e}", v_bd[(0,0)]);

    let (mut u_accel0, mut v_accel0) = ws.split_at(A0, 1);
    let mut u_accel = u_accel0.subview_mut(A0, 0);
    let mut v_accel = v_accel0.subview_mut(A0, 0);
    u_accel.assign(&cps_bd.view());
    u_accel *= &v_bd;
    v_accel.assign(&cps_bd.view());
    v_accel *= &u_bd;
    v_accel *= -1.0;

    let xend = self.grid_size.0 as isize;
    let yend = self.grid_size.1 as isize;
    let h_idxs: [_; 4] = [
      s![2..xend, 1..yend - 1],
      s![..xend - 2, 1..yend - 1],
      s![1..xend - 1, 2..yend],
      s![1..xend - 1, ..yend - 2],
    ];

    let u_factor = -self.gravity / (2.0 * self.dx());
    let v_factor = -self.gravity / (2.0 * self.dy());

    par_scaled_add(u_accel.view_mut(), u_factor,
                   self.orography.slice(h_idxs[0]));
    par_scaled_add(u_accel.view_mut(), -u_factor,
                   self.orography.slice(h_idxs[1]));
    par_scaled_add(v_accel.view_mut(), v_factor,
                   self.orography.slice(h_idxs[2]));
    par_scaled_add(v_accel.view_mut(), -v_factor,
                   self.orography.slice(h_idxs[3]));

    println!("v_accel: {:?}", v_accel[(0,0)]);

    {
      let crop = s![1..xend - 1, 1..yend - 1];
      let mut cropped_next_u = next_u.slice_mut(crop);
      let mut cropped_next_v = next_v.slice_mut(crop);
      let mut cropped_next_h = next_h.slice_mut(crop);

      self.advance(u.view(), v.view(), h.view(),
                   u_accel.view(), v_accel.view(),
                   cropped_next_u.view_mut(),
                   cropped_next_v.view_mut(),
                   cropped_next_h.view_mut());
    }

    fn connect_bdcs(mut uv: ArrayViewMut<f64, Ix2>,
                    column_only: bool) {
      // rows and columns are flipped here.
      let dims = uv.dim();
      for i in 1..dims.1 - 1 {
        uv[(0, i)] = uv[(dims.0 - 2, i)];
        uv[(dims.0 - 1, i)] = uv[(1, i)];
      }

      if column_only { return; }

      for j in 0..dims.0 {
        uv[(j, 0)] = uv[(j, 1)];
        uv[(j, dims.1 - 1)] = uv[(j, dims.1 - 2)];
      }
    }
    connect_bdcs(next_u.view_mut(), false);
    connect_bdcs(next_v.view_mut(), false);
    next_v.subview_mut(A1, 0)
      .fill(0.0);
    next_v.subview_mut(A1, yend as usize - 1)
      .fill(0.0);

    connect_bdcs(next_h.view_mut(), true);
  }
}