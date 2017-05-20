
extern crate na_core as nac;
extern crate linxal;
extern crate num_traits;

use std::ops::{Range, Sub, Div};

use num_traits::NumCast;
use linxal::types::LinxalImplScalar;

pub struct DiscretizedInterval<E> {
  pub range: Range<E>,
  pub delta: f64,
}
impl<E> DiscretizedInterval<E>
  where E: LinxalImplScalar,
{
  pub fn new(range: Range<E>, steps: u64) -> Self {
    DiscretizedInterval {
      range: range,
      delta: steps,
    }
  }

  pub fn d(&self) -> E {
    let cap_t = self.range.end - self.range.start;
    let t: E::RealPart = NumCast::from(self.steps).unwrap();
    cap_t / E::from_real(t)
  }
}