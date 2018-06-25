//# TODO: put global doc

#[macro_use]
extern crate failure;
extern crate snips_fst;
extern crate serde_json;
extern crate serde;

#[macro_use]
extern crate serde_derive;

pub mod constants;
pub mod data;
pub mod errors;
pub mod resolver;
pub use resolver::Resolver;
mod utils;
