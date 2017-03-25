
use nd::{ArrayBase, Ix1, Ix2, Array, Data, DataMut, ViewRepr};
use linxal::factorization::cholesky::*;
use linxal::types::Symmetric;
use linxal::types::{c32, c64};

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
    let mut d: ArrayBase<Vec<f64>, _> = ArrayBase::zeros(n.0);
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