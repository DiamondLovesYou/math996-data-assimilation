
use std::borrow::Cow;
use std::iter::{ExactSizeIterator};
use std::io::Stdout;
use std::time::Duration;

use pbr::ProgressBar;

pub struct ReportingIterator<I>
  where I: ExactSizeIterator,
{
  name: Cow<'static, str>,
  progress: ProgressBar<Stdout>,
  inner: I,
}
impl<I> ReportingIterator<I>
  where I: ExactSizeIterator,
{
  pub fn new(inner: I, name: Cow<'static, str>) -> ReportingIterator<I> {
    let mut p = ProgressBar::new(inner.len() as _);
    p.show_speed = true;
    p.show_percent = true;
    p.show_counter = true;
    p.show_time_left = true;

    let fps = Duration::new(1, 0) / 60;
    p.set_max_refresh_rate(Some(fps));

    let msg = format!("{}: ", name);
    p.message(&msg[..]);
    ReportingIterator {
      name: name,
      progress: p,
      inner: inner,
    }
  }
}

impl<I> Iterator for ReportingIterator<I>
  where I: ExactSizeIterator,
{
  type Item = I::Item;
  fn next(&mut self) -> Option<Self::Item> {
    self.progress.inc();

    match self.inner.next() {
      Some(v) => Some(v),
      None => {
        let msg = format!("{} done!\n", self.name);
        self.progress.finish_println(&msg);

        None
      },
    }
  }
}
