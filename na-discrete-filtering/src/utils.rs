
use nd::{ArrayBase, Ix1, Ix2, Array, Data,};
use linxal::factorization::cholesky::*;
use linxal::types::Symmetric;

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