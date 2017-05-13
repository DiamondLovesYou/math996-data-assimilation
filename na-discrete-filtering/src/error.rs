
use linxal;

pub type Result<T> = ::std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
  Linxal(linxal::types::Error),
}

impl From<linxal::types::Error> for Error {
  fn from(v: linxal::types::Error) -> Error {
    Error::Linxal(v)
  }
}
