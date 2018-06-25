//# TODO: put global doc

#[macro_use]
extern crate failure;
extern crate snips_fst;
extern crate serde_json;
extern crate serde;

#[macro_use]
extern crate serde_derive;

mod constants;
mod data;
mod resolver;
mod utils;

pub mod errors;
pub use resolver::{Resolver, ResolvedValue};
pub use data::{Gazetteer, EntityValue};
