
#[macro_use]
extern crate ndarray as nd;
extern crate alga;
extern crate num;

pub use cholesky::*;
pub use qr::*;

pub mod cholesky;
pub mod qr;