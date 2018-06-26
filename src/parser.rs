use constants::RESTART_IDX;
use constants::{EPS, EPS_IDX, RESTART, SKIP, SKIP_IDX, METADATA_FILENAME};
use data::EntityValue;
use data::Gazetteer;
use errors::GazetteerParserResult;
use snips_fst::string_paths_iterator::{StringPath, StringPathsIterator};
use snips_fst::symbol_table::SymbolTable;
use snips_fst::{fst, operations};
use std::ops::Range;
use utils::whitespace_tokenizer;
use utils::{check_threshold, fst_format_resolved_value, fst_unformat_resolved_value};
use std::path::Path;
use std::fs;
use serde_json;
use std::io::Write;

#[derive(Debug)]
pub struct InternalEntityValue<'a> {
    pub weight: f32,
    pub resolved_value: &'a str,
    pub raw_value: &'a str,
}

impl<'a> InternalEntityValue<'a> {
    pub fn new(entity_value: &'a EntityValue, rank: usize) -> InternalEntityValue {
        InternalEntityValue {
            resolved_value: &entity_value.resolved_value,
            raw_value: &entity_value.raw_value,
            weight: 1.0 - 1.0 / (1.0 + rank as f32), // Adding 1 ensures rank is > 0
        }
    }
}

/// Struct representing the parser. The `fst` attribute holds the finite state transducer
/// representing the logic of the transducer, and its symbol table is held by `symbol_table`.
/// `decoding_threshold` is the minimum fraction of words to match for an entity to be parsed.
/// The Parser will match the longest possible contiguous substrings of a query that match entity
/// values. The order in which the values are added to the parser matters: In case of ambiguity
/// between two parsings, the Parser will output the value that was added first (see Gazetteer).
pub struct Parser {
    fst: fst::Fst,
    symbol_table: SymbolTable,
    decoding_threshold: f32,
}

#[derive(Serialize, Deserialize)]
struct ParserConfig {
    decoding_threshold: f32,
    fst_filename: String,
    symbol_table_filename: String
}

/// Struct holding an individual parsing result. The result of a run of the parser on a query
/// will be a vector of ParsedValue. The `range` attribute is the range of the characters
/// composing the raw value in the input query.
#[derive(Debug, PartialEq)]
pub struct ParsedValue {
    pub resolved_value: String,
    pub range: Range<usize>, // character-level
    pub raw_value: String,
}

impl Parser {
    /// Create an empty parser. Its parser has a single, start state. Its symbol table has
    /// epsilon and a skip symbol
    fn new(decoding_threshold: f32) -> GazetteerParserResult<Parser> {
        // Add a FST with a single state and set it as start
        let mut fst = fst::Fst::new();
        let start_state = fst.add_state();
        fst.set_start(start_state);
        // Add a symbol table with epsilon and skip symbols
        let mut symbol_table = SymbolTable::new();
        let eps_idx = symbol_table.add_symbol(EPS)?;
        assert_eq!(eps_idx, EPS_IDX);
        let skip_idx = symbol_table.add_symbol(SKIP)?;
        assert_eq!(skip_idx, SKIP_IDX);
        let restart_idx = symbol_table.add_symbol(RESTART)?;
        assert_eq!(restart_idx, RESTART_IDX);
        Ok(Parser {
            fst,
            symbol_table,
            decoding_threshold,
        })
    }

    /// This function returns a FST that checks that at least one of the words in
    /// `verbalized_value` is matched. This allows to disable many branches during the
    /// composition. It returns the state at the output of the bottleneck. On the left, the
    /// bottleneck is connected to the start state of the fst. Each token is consumed with weight
    /// `weight_by_token` and is returned. The bottleneck starts by consuming a RESTART symbol or
    /// nothing. The presence of the restart symbol allows forcing the parser to restart between
    /// non-contiguous chunks of text (see `build_input_fst`).
    fn make_bottleneck(
        &mut self,
        verbalized_value: &str,
        weight_by_token: f32,
    ) -> GazetteerParserResult<i32> {
        let start_state = self.fst.start();
        let current_head = self.fst.add_state();
        self.fst
            .add_arc(start_state, RESTART_IDX, EPS_IDX, 0.0, current_head);
        self.fst
            .add_arc(start_state, EPS_IDX, EPS_IDX, 0.0, current_head);
        let next_head = self.fst.add_state();
        for (_, token) in whitespace_tokenizer(verbalized_value) {
            let token_idx = self.symbol_table.add_symbol(&token)?;
            self.fst.add_arc(
                current_head,
                token_idx,
                token_idx,
                weight_by_token,
                next_head,
            );
        }
        Ok(next_head)
    }

    /// This function creates the transducer that maps a subset of the verbalized value onto
    /// the resolved value.
    fn make_value_transducer(
        &mut self,
        mut current_head: i32,
        entity_value: &InternalEntityValue,
        weight_by_token: f32,
    ) -> GazetteerParserResult<()> {
        let mut next_head: i32;
        // First we consume the raw value
        for (_, token) in whitespace_tokenizer(&entity_value.raw_value) {
            next_head = self.fst.add_state();
            let token_idx = self.symbol_table.add_symbol(&token)?;
            // Each arc can either consume a token, and output it...
            self.fst
                .add_arc(current_head, token_idx, token_idx, 0.0, next_head);
            // Or skip the word, with a certain weight, outputting skip
            self.fst
                .add_arc(current_head, EPS_IDX, SKIP_IDX, weight_by_token, next_head);
            // Update current head
            current_head = next_head;
        }
        // Next we output the resolved value
        next_head = self.fst.add_state();
        // The symbol table cannot be deserialized if some symbols contain whitespaces. So we
        // replace them with underscores.
        let token_idx = self
            .symbol_table
            .add_symbol(&fst_format_resolved_value(&entity_value.resolved_value))?;
        self.fst
            .add_arc(current_head, EPS_IDX, token_idx, 0.0, next_head);
        // Make current head final, with weight given by entity value
        self.fst.set_final(next_head, entity_value.weight);
        Ok(())
    }

    /// Add a single entity value to the parser. This function is kept private to promote
    /// creating the parser with a higher level function (such as `from_gazetteer`) that
    /// performs additional global optimizations.
    fn add_value(&mut self, entity_value: &InternalEntityValue) -> GazetteerParserResult<()> {
        // compute weight for each arc based on size of string
        let weight_by_token = 1.0;
        let current_head = self.make_bottleneck(&entity_value.raw_value, -weight_by_token)?;
        self.make_value_transducer(current_head, &entity_value, weight_by_token)?;
        Ok(())
    }

    /// Create a Parser from a Gazetteer, which represents an ordered list of entity values.
    /// This function adds the entity values from the gazetteer
    /// and performs several optimizations on the resulting FST. This is the recommended method
    /// to define a parser. The `parser_threshold` argument sets the minimum fraction of words
    /// to match for an entity to be parsed.
    pub fn from_gazetteer(
        gazetteer: &Gazetteer,
        parser_threshold: f32,
    ) -> GazetteerParserResult<Parser> {
        let mut parser = Parser::new(parser_threshold)?;
        for (rank, entity_value) in gazetteer.data.iter().enumerate() {
            parser.add_value(&InternalEntityValue::new(entity_value, rank))?;
        }
        parser.fst.optimize();
        parser.fst.closure_plus();
        parser.fst.arc_sort(true);
        Ok(parser)
    }

    /// Create an input fst from a string to be parsed. Outputs the input fst and a vec of ranges
    // of the tokens composing it
    fn build_input_fst(&self, input: &str) -> GazetteerParserResult<(fst::Fst, Vec<Range<usize>>)> {
        // build the input fst
        let mut input_fst = fst::Fst::new();
        let mut tokens_ranges: Vec<Range<usize>> = vec![];
        let mut current_head = input_fst.add_state();
        input_fst.set_start(current_head);
        let mut restart_to_be_inserted: bool = false;
        for (token_range, token) in whitespace_tokenizer(input) {
            match self.symbol_table.find_symbol(&token)? {
                Some(value) => {
                    if !restart_to_be_inserted {
                        let next_head = input_fst.add_state();
                        input_fst.add_arc(current_head, EPS_IDX, RESTART_IDX, 0.0, next_head);
                        current_head = next_head;
                        restart_to_be_inserted = true;
                    }
                    let next_head = input_fst.add_state();
                    input_fst.add_arc(current_head, value, value, 0.0, next_head);
                    tokens_ranges.push(token_range);
                    current_head = next_head;
                }
                None => {
                    restart_to_be_inserted = false;
                    // if the word is not in the symbol table, there is no
                    // chance of matching it: we skip
                    continue;
                }
            }
        }
        // Set final state
        input_fst.set_final(current_head, 0.0);
        input_fst.optimize();
        input_fst.arc_sort(false);
        Ok((input_fst, tokens_ranges))
    }

    /// Decode the single shortest path
    fn decode_shortest_path(
        &self,
        shortest_path: &fst::Fst,
        tokens_range: &Vec<Range<usize>>,
    ) -> GazetteerParserResult<Vec<ParsedValue>> {
        let mut path_iterator = StringPathsIterator::new(
            &shortest_path,
            &self.symbol_table,
            &self.symbol_table,
            true,
            true,
        );

        let path = path_iterator
            .next()
            .ok_or_else(|| format_err!("Empty string path iterator"))??;
        Ok(Parser::format_string_path(
            &path,
            &tokens_range,
            self.decoding_threshold,
        )?)
    }

    /// Format the shortest path as a vec of ParsedValue
    fn format_string_path(
        string_path: &StringPath,
        tokens_range: &Vec<Range<usize>>,
        threshold: f32,
    ) -> GazetteerParserResult<Vec<ParsedValue>> {
        let mut input_iterator = whitespace_tokenizer(&string_path.istring);
        let mut parsed_values: Vec<ParsedValue> = vec![];
        let mut input_value_until_now: Vec<String> = vec![];
        let mut current_ranges: Vec<&Range<usize>> = vec![];
        let mut advance_input = false;
        let (_, mut current_input_token) = input_iterator
            .next()
            .ok_or_else(|| format_err!("Empty input string"))?;
        let mut current_input_token_idx: usize = 0;
        let mut n_skips: usize = 0;
        for (_, token) in whitespace_tokenizer(&string_path.ostring) {
            if token == SKIP {
                n_skips += 1;
                continue;
            }
            if advance_input {
                if let Some((_, value)) = input_iterator.next() {
                    current_input_token = value;
                    current_input_token_idx += 1;
                }
            }
            if current_input_token != token {
                let range_start = current_ranges.first().unwrap().start;
                let range_end = current_ranges.last().unwrap().end;
                if check_threshold(input_value_until_now.len(), n_skips, threshold) {
                    parsed_values.push(ParsedValue {
                        raw_value: input_value_until_now.join(" "),
                        resolved_value: fst_unformat_resolved_value(&token),
                        range: range_start..range_end,
                    });
                }
                // Reinitialize accumulators
                n_skips = 0;
                input_value_until_now.clear();
                current_ranges.clear();
                advance_input = false;
            } else {
                input_value_until_now.push(token);
                current_ranges.push(
                    tokens_range
                        .get(current_input_token_idx)
                        .ok_or_else(|| format_err!("Decoding went wrong"))?,
                );
                advance_input = true;
            }
        }
        Ok(parsed_values)
    }

    /// Parse the input string `input` and output a vec of `ParsedValue`
    pub fn run(&self, input: &str) -> GazetteerParserResult<Vec<ParsedValue>> {
        let (input_fst, tokens_range) = self.build_input_fst(input)?;
        // Compose with the parser fst
        let composition = operations::compose(&input_fst, &self.fst);
        if composition.num_states() == 0 {
            return Ok(vec![]);
        }
        let shortest_path = composition.shortest_path(1, false, false);

        let parsing = self.decode_shortest_path(&shortest_path, &tokens_range)?;

        Ok(parsing)
    }

    fn get_parser_config(&self) -> ParserConfig {
        ParserConfig {
            fst_filename: "fst".to_string(),
            symbol_table_filename: "symbol_table".to_string(),
            decoding_threshold: self.decoding_threshold
        }
    }

    /// Dump the resolver to a folder
    pub fn dump<P: AsRef<Path>>(&self, folder_name: P) -> GazetteerParserResult<()> {
        try!(fs::create_dir(folder_name.as_ref()).map_err(|e| format_err!("Error dumping parser: {}", e.to_string())));
        let config = self.get_parser_config();
        let config_string = serde_json::to_string(&config)?;
        // let folder_name_2 = Path::new((&folder_name).as_ref());
        let mut buffer = fs::File::create(folder_name.as_ref().join(METADATA_FILENAME))?;
        buffer.write(config_string.as_bytes())?;
        self.fst.write_file(folder_name.as_ref().join(config.fst_filename))?;
        self.symbol_table.write_file(folder_name.as_ref().join(config.symbol_table_filename), true)?;
        Ok(())
    }

    /// Load a resolver from a folder
    pub fn from_folder<P: AsRef<Path>>(folder_name: P) -> GazetteerParserResult<Parser> {
        let metadata_file = try!(fs::File::open(folder_name.as_ref().join(METADATA_FILENAME)).map_err(|e| format_err!("Error loading parser: {}", e.to_string())));
        let config: ParserConfig = serde_json::from_reader(metadata_file)?;
        let fst = fst::Fst::from_path(folder_name.as_ref().join(config.fst_filename))?;
        let symbol_table = SymbolTable::from_path(folder_name.as_ref().join(config.symbol_table_filename), true)?;
        Ok(Parser {
            fst,
            symbol_table,
            decoding_threshold: config.decoding_threshold
        })
    }
}

#[cfg(test)]
mod tests {
    extern crate tempfile;

    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use data::EntityValue;
    use self::tempfile::tempdir;

    #[test]
    fn test_seralization_deserialization() {
        let tdir = tempdir().unwrap();
        let mut gazetteer = Gazetteer { data: Vec::new() };
        gazetteer.add(EntityValue {
            resolved_value: "The Flying Stones".to_string(),
            raw_value: "the flying stones".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        let parser = Parser::from_gazetteer(&gazetteer, 0.0).unwrap();
        parser.dump(tdir.as_ref().join("parser")).unwrap();
        let reloaded_parser = Parser::from_folder(tdir.as_ref().join("parser")).unwrap();
        tdir.close().unwrap();
        assert!(parser.fst.equals(&reloaded_parser.fst));
        assert_eq!(parser.decoding_threshold, reloaded_parser.decoding_threshold);
    }

    #[test]
    fn test_parser() {
        let mut gazetteer = Gazetteer { data: Vec::new() };
        gazetteer.add(EntityValue {
            resolved_value: "The Flying Stones".to_string(),
            raw_value: "the flying stones".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Blink-182".to_string(),
            raw_value: "blink one eight two".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Je Suis Animal".to_string(),
            raw_value: "je suis animal".to_string(),
        });
        let parser = Parser::from_gazetteer(&gazetteer, 0.0).unwrap();

        let mut parsed = parser.run("je veux écouter les rolling stones").unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    raw_value: "je".to_string(),
                    resolved_value: "Je Suis Animal".to_string(),
                    range: 0..2,
                },
                ParsedValue {
                    raw_value: "rolling stones".to_string(),
                    resolved_value: "The Rolling Stones".to_string(),
                    range: 20..34,
                },
            ]
        );

        parsed = parser.run("je veux ecouter les \t rolling stones").unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    raw_value: "je".to_string(),
                    resolved_value: "Je Suis Animal".to_string(),
                    range: 0..2,
                },
                ParsedValue {
                    raw_value: "rolling stones".to_string(),
                    resolved_value: "The Rolling Stones".to_string(),
                    range: 22..36,
                },
            ]
        );

        parsed = parser
            .run("i want to listen to rolling stones and blink eight")
            .unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    raw_value: "rolling stones".to_string(),
                    resolved_value: "The Rolling Stones".to_string(),
                    range: 20..34,
                },
                ParsedValue {
                    raw_value: "blink eight".to_string(),
                    resolved_value: "Blink-182".to_string(),
                    range: 39..50,
                },
            ]
        );
        parsed = parser.run("joue moi quelque chose").unwrap();
        assert_eq!(parsed, vec![]);
    }

    #[test]
    fn test_parser_with_ranking() {
        /* Weight is here a proxy for the ranking of an artist in a popularity
        index */

        let mut gazetteer = Gazetteer { data: Vec::new() };
        gazetteer.add(EntityValue {
            resolved_value: "Jacques Brel".to_string(),
            raw_value: "jacques brel".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Flying Stones".to_string(),
            raw_value: "the flying stones".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Daniel Brel".to_string(),
            raw_value: "daniel brel".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Jacques".to_string(),
            raw_value: "jacques".to_string(),
        });
        let parser = Parser::from_gazetteer(&gazetteer, 0.5).unwrap();

        /* When there is a tie in terms of number of token matched, match the most popular choice */
        let parsed = parser.run("je veux écouter the stones").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "the stones".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
                range: 16..26,
            }]
        );
        let parsed = parser.run("je veux écouter brel").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "brel".to_string(),
                resolved_value: "Jacques Brel".to_string(),
                range: 16..20,
            }]
        );

        // Resolve to the value with more words matching regardless of popularity
        let parsed = parser.run("je veux écouter the flying stones").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "the flying stones".to_string(),
                resolved_value: "The Flying Stones".to_string(),
                range: 16..33,
            }]
        );
        let parsed = parser.run("je veux écouter daniel brel").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "daniel brel".to_string(),
                resolved_value: "Daniel Brel".to_string(),
                range: 16..27,
            }]
        );
        let parsed = parser.run("je veux écouter jacques").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "jacques".to_string(),
                resolved_value: "Jacques".to_string(),
                range: 16..23,
            }]
        );
    }

    #[test]
    fn test_parser_with_restart() {
        let mut gazetteer = Gazetteer { data: Vec::new() };
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        let parser = Parser::from_gazetteer(&gazetteer, 0.5).unwrap();

        let parsed = parser
            .run("the music I want to listen to is rolling on stones")
            .unwrap();
        assert_eq!(parsed, vec![]);
    }

    #[test]
    #[ignore]
    fn test_parser_with_mixed_ordered_entity() {
        let mut gazetteer = Gazetteer { data: Vec::new() };
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        let parser = Parser::from_gazetteer(&gazetteer, 0.5).unwrap();

        let parsed = parser.run("rolling the stones").unwrap();
        assert_eq!(parsed, vec![]);
    }

    #[test]
    fn test_parser_with_threshold() {
        let mut gazetteer = Gazetteer { data: Vec::new() };
        gazetteer.add(EntityValue {
            resolved_value: "The Flying Stones".to_string(),
            raw_value: "the flying stones".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Blink-182".to_string(),
            raw_value: "blink one eight two".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Je Suis Animal".to_string(),
            raw_value: "je suis animal".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Les Enfoirés".to_string(),
            raw_value: "les enfoirés".to_string(),
        });
        let parser = Parser::from_gazetteer(&gazetteer, 0.5).unwrap();
        let parsed = parser.run("je veux écouter les rolling stones").unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    resolved_value: "Les Enfoirés".to_string(),
                    range: 16..19,
                    raw_value: "les".to_string(),
                },
                ParsedValue {
                    raw_value: "rolling stones".to_string(),
                    resolved_value: "The Rolling Stones".to_string(),
                    range: 20..34,
                },
            ]
        );

        let parser = Parser::from_gazetteer(&gazetteer, 0.3).unwrap();
        let parsed = parser.run("je veux écouter les rolling stones").unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    raw_value: "je".to_string(),
                    resolved_value: "Je Suis Animal".to_string(),
                    range: 0..2,
                },
                ParsedValue {
                    resolved_value: "Les Enfoirés".to_string(),
                    range: 16..19,
                    raw_value: "les".to_string(),
                },
                ParsedValue {
                    raw_value: "rolling stones".to_string(),
                    resolved_value: "The Rolling Stones".to_string(),
                    range: 20..34,
                },
            ]
        );

        let parser = Parser::from_gazetteer(&gazetteer, 0.6).unwrap();
        let parsed = parser.run("je veux écouter les rolling stones").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "rolling stones".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
                range: 20..34,
            }]
        );
    }
}
