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
mod parser;
mod utils;

pub mod errors;
pub use parser::{Parser, ParsedValue};
pub use data::{Gazetteer, EntityValue};
