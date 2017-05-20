
use linxal::types::{LinxalImplScalar};

use nd::{ArrayViewMut, Ix2};

use rand::Rng;
use rand::distributions::IndependentSample;

use std::ops::MulAssign;

use utils::Diagonal;

pub trait ResampleForcing<E>
  where E: LinxalImplScalar + From<f64>,
        E: MulAssign<<E as LinxalImplScalar>::RealPart>,
{
  type Disc;
  fn forcing_view_mut(&mut self,
                      disc: Self::Disc) -> ArrayViewMut<E, Ix2>;

  fn resample_forcing<R, S>(&mut self,
                            disc: Self::Disc,
                            diag: Diagonal<<E as LinxalImplScalar>::RealPart>,
                            sampler: &mut S,
                            mut rand: R)
    where R: Rng,
          S: IndependentSample<f64>,
  {
    let mut r = self.forcing_view_mut(disc);
    for i in 0..r.dim().0 {
      for j in 0..r.dim().1 {
        r[[i, j]] = From::from(sampler.sample(&mut rand));
        r[[i, j]] *= diag[i];
      }
    }
  }
}