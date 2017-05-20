
use nd::{ArrayBase, Ix0, Ix1, Ix2, Array, Data, DataMut, ViewRepr, ArrayView, ArrayViewMut, Axis};
use nd_par::prelude::*;
use linxal::factorization::cholesky::*;
use linxal::eigenvalues::Solution;
use linxal::types::Symmetric;
use linxal::types::{LinxalImplScalar, c32, c64};
use num_complex::Complex;
use num_traits::{Float, Zero, One, NumCast};

use rand::Rng;
use rand::distributions::{Normal, IndependentSample};

use std::ops::{Index, MulAssign};

pub trait CholeskyLDL: Cholesky + Sized {
  fn compute<D1: Data>(a: &ArrayBase<D1, Ix2>, uplo: Symmetric)
                       -> Result<(Array<Self, Ix2>, Array<Self, Ix1>), CholeskyError>
    where D1: Data<Elem=Self>;
}
impl CholeskyLDL for f64 {
  fn compute<D1: Data>(a: &ArrayBase<D1, Ix2>, uplo: Symmetric)
                       -> Result<(Array<Self, Ix2>,
                                  Array<Self, Ix1>), CholeskyError>
    where D1: Data<Elem=Self>,
  {
    let mut u = Cholesky::compute_into(a.to_owned(), uplo)?;
    let n = u.dim();
    let mut d: Array<f64, _> = ArrayBase::zeros(n.0);
    for i in 0..n.0 {
      d[[i]] = u[[i, i]];
      if u[[i, i]] == 1.0 { continue; }

      let factor = d[[i]];
      assert!(factor != 0.0, "TODO: think about this");
      match uplo {
        Symmetric::Lower => {
          u.column_mut(i)
            .slice_mut(s![i as isize .. n.1 as isize])
            .mapv_inplace(|v| v / factor );
        },
        Symmetric::Upper => {
          u.row_mut(i)
            .slice_mut(s![i as isize .. n.1 as isize])
            .mapv_inplace(|v| v / factor );
        },
      }
    }

    Ok((u, d))
  }
}

pub fn extend_dim_mut<D>(d: &mut ArrayBase<D, Ix1>, t: bool)
  -> ArrayBase<ViewRepr<&mut D::Elem>, Ix2>
  where D: DataMut,
{
  let d_dim = d.dim();
  let d_slice = d.as_slice_mut().expect("d_slice");
  let dim = if !t {
    (d_dim, 1)
  } else {
    (1, d_dim)
  };
  let d2 = ArrayBase::<ViewRepr<&mut D::Elem>, _>::from_shape(dim, d_slice)
    .expect("d2");

  d2
}
pub fn extend_dim_ref<D>(d: &ArrayBase<D, Ix1>, t: bool)
  -> ArrayBase<ViewRepr<&D::Elem>, Ix2>
  where D: Data,
{
  let d_dim = d.dim();
  let d_slice = d.as_slice().expect("d_slice");
  let dim = if !t {
    (d_dim, 1)
  } else {
    (1, d_dim)
  };
  let d2 = ArrayBase::<ViewRepr<&D::Elem>, _>::from_shape(dim, d_slice)
    .expect("d2");

  d2
}

pub trait Sqrt {
  fn _sqrt(&self) -> Self;
}
impl Sqrt for c32 {
  fn _sqrt(&self) -> Self { self.sqrt() }
}
impl Sqrt for c64 {
  fn _sqrt(&self) -> Self { self.sqrt() }
}
impl Sqrt for f32 {
  fn _sqrt(&self) -> Self { self.sqrt() }
}
impl Sqrt for f64 {
  fn _sqrt(&self) -> Self { self.sqrt() }
}

pub trait Exp {
  fn _exp(&self) -> Self;
}
impl Exp for c32 {
  fn _exp(&self) -> c32 { self.exp() }
}
impl Exp for c64 {
  fn _exp(&self) -> c64 { self.exp() }
}
impl Exp for f32 {
  fn _exp(&self) -> f32 { self.exp() }
}
impl Exp for f64 {
  fn _exp(&self) -> f64 { self.exp() }
}

pub trait SolutionHelper<EV, IV> {
  fn values(self) -> Array<IV, Ix1>;
  fn values_and_left_vectors(&mut self) -> (ArrayViewMut<IV, Ix1>, ArrayView<EV, Ix2>);
}

impl<T> SolutionHelper<T, Complex<<T as LinxalImplScalar>::RealPart>> for Solution<T, Complex<<T as LinxalImplScalar>::RealPart>>
  where T: Float + LinxalImplScalar,
{
  fn values(self) -> Array<Complex<<T as LinxalImplScalar>::RealPart>, Ix1> {
    let Solution {
      values, ..
    } = self;
    values
  }
  fn values_and_left_vectors(&mut self) -> (ArrayViewMut<Complex<<T as LinxalImplScalar>::RealPart>, Ix1>, ArrayView<T, Ix2>) {
    (self.values.view_mut(), self.left_vectors.as_ref().unwrap().view())
  }
}

#[derive(Debug)]
pub enum Diagonal<'a, E>
  where E: 'a,
{
  Single(E),
  Multiple(ArrayView<'a, E, Ix1>)
}
impl<'a, E> Index<usize> for Diagonal<'a, E> {
  type Output = E;
  fn index(&self, idx: usize) -> &E {
    match self {
      &Diagonal::Single(ref e) => e,
      &Diagonal::Multiple(ref v) => &v[idx],
    }
  }
}
impl<'a, E> From<E> for Diagonal<'a, E> {
  fn from(v: E) -> Diagonal<'a, E> {
    Diagonal::Single(v)
  }
}
impl<'a, E> From<ArrayView<'a, E, Ix1>> for Diagonal<'a, E> {
  fn from(v: ArrayView<'a, E, Ix1>) -> Self {
    Diagonal::Multiple(v)
  }
}

pub trait PartialEqWithinTol<Rhs, Tol> {
  const STD_TOL: Tol;
  fn partial_eq_within_tol(&self, rhs: &Rhs, tol: Tol) -> bool;

  fn partial_neq_within_tol(&self, rhs: &Rhs, tol: Tol) -> bool {
    !self.partial_eq_within_tol(rhs, tol)
  }

  fn partial_eq_within_std_tol(&self, rhs: &Rhs) -> bool {
    self.partial_eq_within_tol(rhs, Self::STD_TOL)
  }
  fn partial_neq_within_std_tol(&self, rhs: &Rhs) -> bool {
    !self.partial_eq_within_std_tol(rhs)
  }
}

impl PartialEqWithinTol<f64, f64> for f64 {
  const STD_TOL: Self = ::std::f64::EPSILON;
  fn partial_eq_within_tol(&self, rhs: &f64, tol: f64) -> bool {
    (self - rhs).mag() <= tol
  }
}
impl PartialEqWithinTol<f32, f32> for f32 {
  const STD_TOL: Self = ::std::f32::EPSILON;
  fn partial_eq_within_tol(&self, rhs: &f32, tol: f32) -> bool {
    (self - rhs).mag() <= tol
  }
}
impl<'a, T> PartialEqWithinTol<ArrayView<'a, T, Ix0>, T> for ArrayView<'a, T, Ix0>
  where T: PartialEqWithinTol<T, T> + LinxalImplScalar,
{
  const STD_TOL: T = T::STD_TOL;
  fn partial_eq_within_tol(&self, rhs: &ArrayView<'a, T, Ix0>, tol: T) -> bool {
    self[()].partial_eq_within_tol(&rhs[()], tol)
  }
}
impl<'a, T> PartialEqWithinTol<ArrayView<'a, T, Ix1>, T> for ArrayView<'a, T, Ix1>
  where T: PartialEqWithinTol<T, T> + LinxalImplScalar,
{
  const STD_TOL: T = T::STD_TOL;
  fn partial_eq_within_tol(&self, rhs: &ArrayView<'a, T, Ix1>, tol: T) -> bool {
    self.axis_iter(Axis(0))
      .zip(rhs.axis_iter(Axis(0)))
      .all(|(l, r)| {
        l.partial_eq_within_tol(&r, tol)
      })
  }
}

pub fn par_conj_t<E>(dest: &mut ArrayViewMut<E, Ix2>, src: &ArrayView<E, Ix2>)
  where E: LinxalImplScalar + Send + Sync,
{
  assert_eq!(dest.dim().0, src.dim().1);
  assert_eq!(dest.dim().1, src.dim().0);

  dest.axis_iter_mut(Axis(0))
    .into_par_iter()
    .zip(src.axis_iter(Axis(1)).into_par_iter())
    .for_each(|(mut dest, src)| {
      dest.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(src.axis_iter(Axis(0)).into_par_iter())
        .for_each(|(mut dest, src)| {
          dest[()] = src[()].cj();
        });
    });
}

pub fn make_2d_randn<E, R>(dim: (usize, usize),
                           d: Diagonal<E>,
                           mut rand: &mut R) -> Array<E, Ix2>
  where E: LinxalImplScalar + MulAssign + Zero + One + NumCast,
        R: Rng,
{
  let normal = Normal::new(Zero::zero(), One::one());
  let mut r: Array<E, Ix2> =
    ArrayBase::zeros(dim);
  for i in 0..dim.0 {
    for j in 0..dim.1 {
      r[[i, j]] = NumCast::from(normal.ind_sample(&mut rand))
        .unwrap();
      r[[i, j]] *= d[j];
    }
  }

  r
}