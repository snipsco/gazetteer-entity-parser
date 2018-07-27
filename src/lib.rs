//! ## Getting Started
//!
//! This crate exposes a parser to match and resolve entity values, drawn from a gazetteer, inside
//! written queries. The parser is based on a finite state transducer (FST) and built from a
//! ordered list of entity values. The parser will attempt to find and resolve maximal substrings
//! of the input queries against the gazetteer values, allowing to skip some of the tokens
//! composing the entity value. More precisely, when several resolutions are possible
//! - the entity value sharing the most tokens with the input is preferred.
//! - in case of a tie, the entity value with smallest rank in the gazetteer is preferred. The
//! sorting of the gazetteer therefore matters.
//!
//!```rust
//!
//! use gazetteer_entity_parser::{Gazetteer, Parser, EntityValue, ParsedValue};
//!
//! let mut gazetteer = Gazetteer::new();
//! // We fill the gazetteer with artists, sorted by popularity
//! gazetteer.add(EntityValue {
//!     resolved_value: "The Rolling Stones".to_string(),
//!     raw_value: "the rolling stones".to_string(),
//! });
//! gazetteer.add(EntityValue {
//!     resolved_value: "The Strokes".to_string(),
//!     raw_value: "the strokes".to_string(),
//! });
//! gazetteer.add(EntityValue {
//!     resolved_value: "Jacques Brel".to_string(),
//!     raw_value: "jacques brel".to_string(),
//! });
//! gazetteer.add(EntityValue {
//!     resolved_value: "Daniel Brel".to_string(),
//!     raw_value: "daniel brel".to_string(),
//! });
//! let parser = Parser::from_gazetteer(&gazetteer).unwrap();
//! let parsed_stones = parser.run("I want to listen to the stones", 0.5).unwrap();
//! assert_eq!(
//!     parsed_stones,
//!     vec![ParsedValue {
//!         raw_value: "the stones".to_string(),
//!         resolved_value: "The Rolling Stones".to_string(),
//!         range: 20..30,
//!     }]
//! );
//! // Example with an ambiguity, where the artist with smaller rank is preferred
//! let parsed_brel = parser.run("I want to listen to brel", 0.5).unwrap();
//! assert_eq!(
//!     parsed_brel,
//!     vec![ParsedValue {
//!         raw_value: "brel".to_string(),
//!         resolved_value: "Jacques Brel".to_string(),
//!         range: 20..24,
//!     }]
//! );
//!```

#[macro_use]
extern crate failure;
extern crate serde;
extern crate serde_json;
extern crate rmp_serde as rmps;
extern crate snips_fst;

#[macro_use]
extern crate serde_derive;

mod constants;
mod data;
mod parser;
mod utils;

pub use data::{EntityValue, Gazetteer};
pub use parser::{ParsedValue, Parser};
pub mod errors;
