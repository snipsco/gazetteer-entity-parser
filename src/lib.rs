//! ## Getting Started
//!
//! This crate exposes a parser to match and resolve entity values, drawn from a gazetteer, inside
//! written queries. The parser is built from an
//! ordered list of entity values. It will attempt to find and resolve maximal substrings
//! of the input queries against the gazetteer values, allowing to skip some of the tokens
//! composing the entity value. More precisely, when several resolutions are possible
//! - the entity value sharing the most tokens with the input is preferred.
//! - in case of a tie, the entity value with smallest rank in the gazetteer is preferred. The
//! sorting of the gazetteer therefore matters.
//!
//!```rust
//!
//! use gazetteer_entity_parser::{Gazetteer, ParserBuilder, EntityValue, ParsedValue};
//!
//! let mut gazetteer = Gazetteer::default();
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
//!     resolved_value: "The Hives".to_string(),
//!     raw_value: "the hives".to_string(),
//! });
//! gazetteer.add(EntityValue {
//!     resolved_value: "Jacques Brel".to_string(),
//!     raw_value: "jacques brel".to_string(),
//! });
//! gazetteer.add(EntityValue {
//!     resolved_value: "Daniel Brel".to_string(),
//!     raw_value: "daniel brel".to_string(),
//! });
//!
//! // The Parser is then instantiated using a builder pattern. The ParserBuilder is instantiated
//! // from a gazetteer and a decoding threshold, i.e. the minimal fraction of matched tokens for
//! // a parsing to be possible. Additional methods allow to set the stop words of the Parser,
//! // e.g. here the most common word of the gazetteer, plus "a" and "for".
//! let parser = ParserBuilder::default()
//!     .minimum_tokens_ratio(0.5)
//!     .gazetteer(gazetteer)
//!     .n_stop_words(1)
//!     .additional_stop_words(vec!["a".to_string(), "for".to_string()]).build().unwrap();
//!
//! // Parse a sentence
//! let parsed_stones = parser.run("I want to listen to the stones").unwrap();
//! assert_eq!(
//!     parsed_stones,
//!     vec![ParsedValue {
//!         raw_value: "the stones".to_string(),
//!         resolved_value: "The Rolling Stones".to_string(),
//!         matched_value: "the rolling stones".to_string(),
//!         range: 20..30,
//!     }]
//! );
//!
//! // Example with an ambiguity, where the artist with smaller rank is preferred
//! let parsed_brel = parser.run("I want to listen to brel").unwrap();
//! assert_eq!(
//!     parsed_brel,
//!     vec![ParsedValue {
//!         raw_value: "brel".to_string(),
//!         resolved_value: "Jacques Brel".to_string(),
//!         matched_value: "jacques brel".to_string(),
//!         range: 20..24,
//!     }]
//! );
//!```

#[macro_use]
extern crate failure;
extern crate fnv;
extern crate rmp_serde as rmps;
extern crate serde;
extern crate serde_json;

#[macro_use]
extern crate serde_derive;

mod constants;
mod data;
mod parser;
mod parser_builder;
mod symbol_table;
mod utils;

pub use data::{EntityValue, Gazetteer};
pub use parser::{ParsedValue, Parser};
pub use parser_builder::ParserBuilder;
pub mod errors;
