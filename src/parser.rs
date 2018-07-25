use std::collections::{HashMap, HashSet};
use constants::RESTART_IDX;
use constants::{EPS, EPS_IDX, METADATA_FILENAME, RESTART, SKIP, SKIP_IDX, CONSUMED, CONSUMED_IDX};
use data::EntityValue;
use data::Gazetteer;
use errors::GazetteerParserResult;
use failure::ResultExt;
use serde_json;
use snips_fst::string_paths_iterator::{StringPath, StringPathsIterator};
use snips_fst::symbol_table::SymbolTable;
use snips_fst::{fst, operations};
use std::fs;
use std::io::Write;
use std::ops::Range;
use std::path::Path;
use utils::whitespace_tokenizer;
use utils::{check_threshold, fst_format_resolved_value, fst_unformat_resolved_value};
// use std::intrinsics::breakpoint;

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
/// The Parser will match the longest possible contiguous substrings of a query that match entity
/// values. The order in which the values are added to the parser matters: In case of ambiguity
/// between two parsings, the Parser will output the value that was added first (see Gazetteer).
pub struct Parser {
    fst: fst::Fst,
    symbol_table: SymbolTable,
    word_to_value: HashMap<i32, HashSet<i32>>
}

#[derive(Serialize, Deserialize)]
struct ParserConfig {
    fst_filename: String,
    symbol_table_filename: String,
    version: String,
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
    /// Create an empty parser. Its FST has a single, start state, and it has minimal
    /// symbol table.
    fn new() -> GazetteerParserResult<Parser> {
        // Add a FST with a single state and set it as start
        let mut fst = fst::Fst::new();
        let start_state = fst.add_state();
        fst.set_start(start_state);
        // Add a symbol table with epsilon and skip symbols
        let mut symbol_table = SymbolTable::new();
        let eps_idx = symbol_table.add_symbol(EPS)?;
        if eps_idx != EPS_IDX {
            return Err(format_err!("Wrong epsilon index: {}", eps_idx));
        }
        let skip_idx = symbol_table.add_symbol(SKIP)?;
        if skip_idx != SKIP_IDX {
            return Err(format_err!("Wrong skip index: {}", skip_idx));
        }
        let consumed_idx = symbol_table.add_symbol(CONSUMED)?;
        if consumed_idx != CONSUMED_IDX {
            return Err(format_err!("Wrong consumed index: {}", consumed_idx));
        }
        let restart_idx = symbol_table.add_symbol(RESTART)?;
        if restart_idx != RESTART_IDX {
            return Err(format_err!("Wrong restart index: {}", restart_idx));
        }
        Ok(Parser { fst, symbol_table, word_to_value: HashMap::new() })
    }

    /// This function returns a FST that checks that at least one of the words in
    /// `verbalized_value` is matched. This allows to disable many branches during the
    /// composition. It returns the state at the output of the bottleneck. On the left, the
    /// bottleneck is connected to the start state of the fst. Each token is consumed with weight
    /// `weight_by_token` and is returned. The bottleneck starts by consuming a RESTART symbol or
    /// nothing. The presence of the restart symbol allows forcing the parser to restart between
    /// non-contiguous chunks of text (see `build_input_fst`).
    // fn make_bottleneck(
    //     &mut self,
    //     verbalized_value: &str,
    //     weight_by_token: f32,
    // ) -> GazetteerParserResult<i32> {
    //     let start_state = self.fst.start();
    //     let current_head = self.fst.add_state();
    //     self.fst
    //         .add_arc(start_state, RESTART_IDX, EPS_IDX, 0.0, current_head);
    //     self.fst
    //         .add_arc(start_state, EPS_IDX, EPS_IDX, 0.0, current_head);
    //     let next_head = self.fst.add_state();
    //     for (_, token) in whitespace_tokenizer(verbalized_value) {
    //         let token_idx = self.symbol_table.add_symbol(&token)?;
    //         self.fst.add_arc(
    //             current_head,
    //             token_idx,
    //             token_idx,
    //             weight_by_token,
    //             next_head,
    //         );
    //     }
    //     Ok(next_head)
    // }

    /// This function creates the transducer that maps a subset of the verbalized value onto
    /// the resolved value.
    fn make_value_transducer(
        &mut self,
        mut current_head: i32,
        entity_value: &InternalEntityValue,
        weight_by_token: f32,
    ) -> GazetteerParserResult<()> {
        let mut next_head: i32;
        // The symbol table cannot be deserialized if some symbols contain whitespaces. So we
        // replace them with underscores.
        let resolved_value_idx = self
            .symbol_table
            .add_symbol(&fst_format_resolved_value(&entity_value.resolved_value))?;
        // First we consume the raw value
        let mut bottleneck_inserted = false;
        for (_, token) in whitespace_tokenizer(&entity_value.raw_value) {
            next_head = self.fst.add_state();
            let token_idx = self.symbol_table.add_symbol(&token)?;
            // Each arc can either consume a token, and output it...
            // self.fst
            //     .add_arc(current_head, resolved_value_idx, resolved_value_idx, 0.0, next_head);
            self.fst
                .add_arc(current_head, resolved_value_idx, CONSUMED_IDX, 0.0, next_head);
            // Or skip the word, with a certain weight, outputting skip
            // We want at least one word belonging to the entity value
            // So while subsequent symbols are optional, the first entity symbol must be here
            if !bottleneck_inserted {
                bottleneck_inserted = true;
            } else {
                self.fst
                    .add_arc(current_head, EPS_IDX, SKIP_IDX, weight_by_token, next_head);
            }
            // Update current head
            current_head = next_head;
            // Add the mapping from token to resolved value
            self.word_to_value.entry(token_idx)
                .and_modify(|e| { (*e).insert(resolved_value_idx); })
                .or_insert({
                    let mut h = HashSet::new();
                    h.insert(resolved_value_idx);
                    h
                });
        }
        // Next we output the resolved value
        next_head = self.fst.add_state();

        self.fst
            .add_arc(current_head, EPS_IDX, resolved_value_idx, 0.0, next_head);
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
        // let current_head = self.make_bottleneck(&entity_value.raw_value, -weight_by_token)?;
        // The only part of the bottelneck that we need to retain is the
        // restart symbol
        let mut current_head = self.fst.start();
        let next_head = self.fst.add_state();
        self.fst.add_arc(current_head, RESTART_IDX, EPS_IDX, 0.0, next_head);
        self.fst.add_arc(current_head, EPS_IDX, EPS_IDX, 0.0, next_head);
        // self.fst.add_arc(current_head, EPS_IDX, EPS_IDX, 0.0, next_head);
        current_head = next_head;
        self.make_value_transducer(current_head, &entity_value, weight_by_token)?;
        Ok(())
    }

    /// Create a Parser from a Gazetteer, which represents an ordered list of entity values.
    /// This function adds the entity values from the gazetteer
    /// and performs several optimizations on the resulting FST. This is the recommended method
    /// to define a parser.
    pub fn from_gazetteer(gazetteer: &Gazetteer) -> GazetteerParserResult<Parser> {
        let mut parser = Parser::new()?;
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
                    // println!("VALUE: {:?}", value);
                    for res_value in self.word_to_value.get(&value).unwrap() {
                        // println!("RES_VALUE: {:?}", res_value);
                        input_fst.add_arc(current_head, value, *res_value, 0.0, next_head);
                    }
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
        // println!("INPUT FST NUM STATES {:?}", input_fst.num_states());
        // input_fst.write_file("input_fst.fst").unwrap();
        Ok((input_fst, tokens_ranges))
    }

    /// Decode the single shortest path
    fn decode_shortest_path(
        &self,
        shortest_path: &fst::Fst,
        tokens_range: &Vec<Range<usize>>,
        decoding_threshold: f32,
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
            .unwrap_or_else(|| Err(format_err!("Empty string path iterator")))?;
        // println!("PATH: {:?}", path);
        Ok(Parser::format_string_path(
            &path,
            &tokens_range,
            decoding_threshold,
        )?)
    }

    /// Format the shortest path as a vec of ParsedValue
    fn format_string_path(
        string_path: &StringPath,
        tokens_range: &Vec<Range<usize>>,
        threshold: f32,
    ) -> GazetteerParserResult<Vec<ParsedValue>> {
        // let mut input_iterator = whitespace_tokenizer(&string_path.istring);
        let mut parsed_values: Vec<ParsedValue> = vec![];
        // let mut input_value_until_now: Vec<String> = vec![];
        // let mut current_ranges: Vec<&Range<usize>> = vec![];
        // let mut advance_input = false;
        // let (_, mut current_input_token) = input_iterator
        //     .next()
        //     .ok_or_else(|| format_err!("Empty input string"))?;
        // let mut current_input_token_idx: usize = 0;
        // let mut last_output_skipped: bool = false;
        // let mut n_skips: usize = 0;
        let mut output_iterator = whitespace_tokenizer(&string_path.ostring).peekable();
        let mut input_iterator = whitespace_tokenizer(&string_path.istring).peekable();
        let mut input_ranges_iterator = tokens_range.iter();

        // let consumed_string = CONSUMED.to_string();
        // let skip_string = SKIP.to_string();

        'outer: loop {
            let mut n_consumed_tokens = 0;
            let mut n_skips = 0;
            let current_resolved_value = 'inner: loop {
                // First consume the skips and the current parsed value from the output
                // Assumption: the parser fst only outputs resolved values, the skip symbol, or the
                // consumed symbol
                match output_iterator.peek() {
                    Some((_, value)) => {
                        // println!("value {:?}", value);
                        // bail!("");
                        match value.as_ref() {
                            CONSUMED => {n_consumed_tokens += 1;}
                            SKIP => {n_skips += 1}
                            resolved => {break 'inner Some(resolved.to_string())}
                        }
                    }
                    None => { break 'inner None }
                };
                // println!("n_consumed_tokens {:?}", n_consumed_tokens);
                // println!("n_skips {:?}", n_skips);
                output_iterator.next();
            };

            match current_resolved_value {
                None => { break 'outer }
                Some(resolved_val) => {
                    // Get the input tokens corresponding to the current resolved value
                    // let consumed_input_tokens =
                    let mut input_value_until_now: Vec<String> = vec![];
                    let mut current_ranges: Vec<&Range<usize>> = vec![];
                    for _ in 0..n_consumed_tokens {
                        // println!("GOT IN FOR LOOP");
                        let token = match input_iterator.next() {
                            Some((_, value)) => { value }
                            None => bail!("Not enough input values")
                        };
                        let range = match input_ranges_iterator.next() {
                            Some(value) => { value }
                            None => bail!("Not enough input ranges")
                        };
                        input_value_until_now.push(token);
                        current_ranges.push(range);
                    }
                    if check_threshold(n_consumed_tokens, n_skips, threshold) {
                        let range_start = current_ranges.first().unwrap().start;
                        let range_end = current_ranges.last().unwrap().end;
                        parsed_values.push(ParsedValue {
                            raw_value: input_value_until_now.join(" "),
                            resolved_value: fst_unformat_resolved_value(&resolved_val),
                            range: range_start..range_end,
                        });
                    }
                }
            }
            // Going to the next output symbol
            output_iterator.next();
        }
        Ok(parsed_values)
        // // Get the input tokens corresponding to the current resolved value
        // // let consumed_input_tokens =
        // let mut input_value_until_now: Vec<String> = vec![];
        // let mut current_ranges: Vec<&Range<usize>> = vec![];
        // for _ in 0..n_consumed_tokens {
        //     // println!("GOT IN FOR LOOP");
        //     let token = match input_iterator.next() {
        //         Some((_, value)) => { value }
        //         None => bail!("Not enough input values")
        //     };
        //     let range = match input_ranges_iterator.next() {
        //         Some(value) => { value }
        //         None => bail!("Not enough input ranges")
        //     };
        //     input_value_until_now.push(token);
        //     current_ranges.push(range);
        // }
        // println!("INPUT VALUE UNTIL NOW {:?}", input_value_until_now);
        // // Add the parsed value if it meets the threshold condition
        // println!("THRESHOLD: n_consumed_tokens {:?}", n_consumed_tokens);
        // println!("N_SKIPS: n_skips {:?}", n_skips);
        // println!("CHECK_THRESHOLD RESULT: {:?}", check_threshold(n_consumed_tokens, n_skips, threshold));
        // if check_threshold(n_consumed_tokens, n_skips, threshold) {
        //     let range_start = current_ranges.first().unwrap().start;
        //     let range_end = current_ranges.last().unwrap().end;
        //     parsed_values.push(ParsedValue {
        //         raw_value: input_value_until_now.join(" "),
        //         resolved_value: fst_unformat_resolved_value(&current_resolved_value),
        //         range: range_start..range_end,
        //     });
        // }


        // 'outer: loop {
        //     // First consume the output symbols corresponding to the next parsed value
        //     let mut n_consumed_tokens = 0;
        //     let mut n_skips = 0;
        //     let mut finished = false;
        //     let mut current_resolved_value: String = "".to_string();
        //     // let mut skip_seen = false;
        //     'inner: loop {
        //         match output_iterator.peek() {
        //             Some((_, value)) => {
        //                 println!("CURRENT VALUE {:?}", value);
        //                 if n_consumed_tokens == 0 && value != SKIP {
        //                     // println!("VALUE {:?}", value);
        //                     // Consuming first token: setting the current resolved value
        //                     // if value == SKIP {
        //                     //     // println!("FOOOOO {:?}", n_consumed_tokens);
        //                     //     bail!("Skipping before any value is parsed")
        //                     // }
        //                     current_resolved_value = value.to_string();
        //                 }
        //                 // println!("current resolved value{:?}", current_resolved_value);
        //                 if (value != SKIP && value != current_resolved_value.as_str()) || (value == SKIP && n_consumed_tokens > 0) {
        //                     break 'inner;
        //                 }
        //                 if value == SKIP {
        //                     n_skips += 1;
        //                     // skip_seen = true;
        //                 } else {
        //                     n_consumed_tokens += 1;
        //                     // n_skips = 0
        //                 }
        //             }
        //             None => {
        //                 finished = true;
        //                 break 'inner;
        //             }
        //         }
        //         output_iterator.next();
        //         // drop(current_resolved_value);
        //         println!("n_consumed_tokens {:?}", n_consumed_tokens);
        //         println!("n_skips {:?}", n_skips);
        //     }
        //
        //     // Get the input tokens corresponding to the current resolved value
        //     // let consumed_input_tokens =
        //     let mut input_value_until_now: Vec<String> = vec![];
        //     let mut current_ranges: Vec<&Range<usize>> = vec![];
        //     for _ in 0..n_consumed_tokens {
        //         // println!("GOT IN FOR LOOP");
        //         let token = match input_iterator.next() {
        //             Some((_, value)) => { value }
        //             None => bail!("Not enough input values")
        //         };
        //         let range = match input_ranges_iterator.next() {
        //             Some(value) => { value }
        //             None => bail!("Not enough input ranges")
        //         };
        //         input_value_until_now.push(token);
        //         current_ranges.push(range);
        //     }
        //     println!("INPUT VALUE UNTIL NOW {:?}", input_value_until_now);
        //     // Add the parsed value if it meets the threshold condition
        //     println!("THRESHOLD: n_consumed_tokens {:?}", n_consumed_tokens);
        //     println!("N_SKIPS: n_skips {:?}", n_skips);
        //     println!("CHECK_THRESHOLD RESULT: {:?}", check_threshold(n_consumed_tokens, n_skips, threshold));
        //     if check_threshold(n_consumed_tokens, n_skips, threshold) {
        //         let range_start = current_ranges.first().unwrap().start;
        //         let range_end = current_ranges.last().unwrap().end;
        //         parsed_values.push(ParsedValue {
        //             raw_value: input_value_until_now.join(" "),
        //             resolved_value: fst_unformat_resolved_value(&current_resolved_value),
        //             range: range_start..range_end,
        //         });
        //     }
        //     if finished {
        //         break 'outer
        //     }
        // }
        // Ok(parsed_values)
        // let mut output_iterator = whitespace_tokenizer(&string_path.ostring).peekable();
        // let mut input_iterator = whitespace_tokenizer(&string_path.istring).peekable();
        // let mut start_new_resolved_value = true;
        // for (_, token) in whitespace_tokenizer(&string_path.ostring) {
        //     if last_output
        //     // if start_new_resolved_value {
        //     //     let current_resolved_value = token;
        //     //     let
        //     // }
        //
        // }

        //     if token == SKIP {
        //         n_skips += 1;
        //         continue;
        //     }
        //     if advance_input {
        //         if let Some((_, value)) = input_iterator.next() {
        //             current_input_token = value;
        //             current_input_token_idx += 1;
        //         }
        //     }
        //     if current_input_token != token {
        //         let range_start = current_ranges.first().unwrap().start;
        //         let range_end = current_ranges.last().unwrap().end;
        //         if check_threshold(input_value_until_now.len(), n_skips, threshold) {
        //             parsed_values.push(ParsedValue {
        //                 raw_value: input_value_until_now.join(" "),
        //                 resolved_value: fst_unformat_resolved_value(&token),
        //                 range: range_start..range_end,
        //             });
        //         }
        //         // Reinitialize accumulators
        //         n_skips = 0;
        //         input_value_until_now.clear();
        //         current_ranges.clear();
        //         advance_input = false;
        //     } else {
        //         input_value_until_now.push(token);
        //         current_ranges.push(
        //             tokens_range
        //                 .get(current_input_token_idx)
        //                 .ok_or_else(|| format_err!("Decoding went wrong"))?,
        //         );
        //         advance_input = true;
        //     }
        // }
        // Ok(parsed_values)
    }

    /// Parse the input string `input` and output a vec of `ParsedValue`. `decoding_threshold` is
    /// the minimum fraction of words to match for an entity to be parsed.
    pub fn run(
        &self,
        input: &str,
        decoding_threshold: f32,
    ) -> GazetteerParserResult<Vec<ParsedValue>> {
        let (input_fst, tokens_range) = self.build_input_fst(input)?;
        // Compose with the parser fst
        let composition = operations::compose(&input_fst, &self.fst);
        // composition.write_file("composition.fst").unwrap();
        // println!("COMPOSITION NUM STATES {:?}", composition.num_states());
        if composition.num_states() == 0 {
            return Ok(vec![]);
        }
        let shortest_path = composition.shortest_path(1, false, false);
        // println!("SHORTEST PATH NUM STATES {:?}", shortest_path.num_states());
        let parsing = self.decode_shortest_path(&shortest_path, &tokens_range, decoding_threshold)?;
        // println!("PARSING: {:?}", parsing);
        Ok(parsing)
    }

    fn get_parser_config(&self) -> ParserConfig {
        ParserConfig {
            fst_filename: "fst".to_string(),
            symbol_table_filename: "symbol_table".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Dump the parser to a folder
    pub fn dump<P: AsRef<Path>>(&self, folder_name: P) -> GazetteerParserResult<()> {
        fs::create_dir(folder_name.as_ref()).with_context(|_| {
            format!(
                "Cannot create resolver directory {:?}",
                folder_name.as_ref()
            )
        })?;
        let config = self.get_parser_config();
        let config_string = serde_json::to_string(&config)?;
        let mut buffer = fs::File::create(folder_name.as_ref().join(METADATA_FILENAME))?;
        buffer.write(config_string.as_bytes())?;
        self.fst
            .write_file(folder_name.as_ref().join(config.fst_filename))?;
        self.symbol_table.write_file(
            folder_name.as_ref().join(config.symbol_table_filename),
            false,
        )?;
        Ok(())
    }

    /// Load a resolver from a folder
    pub fn from_folder<P: AsRef<Path>>(folder_name: P) -> GazetteerParserResult<Parser> {
        let metadata_path = folder_name.as_ref().join(METADATA_FILENAME);
        let metadata_file = fs::File::open(&metadata_path)
            .with_context(|_| format!("Cannot open metadata file {:?}", metadata_path))?;
        let config: ParserConfig = serde_json::from_reader(metadata_file)?;
        let fst = fst::Fst::from_path(folder_name.as_ref().join(config.fst_filename))?;
        let symbol_table = SymbolTable::from_path(
            folder_name.as_ref().join(config.symbol_table_filename),
            true,
        )?;
        // FIXME
        Ok(Parser { fst, symbol_table, word_to_value: HashMap::new() })
    }
}

#[cfg(test)]
mod tests {
    extern crate tempfile;

    use self::tempfile::tempdir;
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use data::EntityValue;

    #[test]
    fn test_seralization_deserialization() {
        let tdir = tempdir().unwrap();
        let mut gazetteer = Gazetteer::new();
        gazetteer.add(EntityValue {
            resolved_value: "The Flying Stones".to_string(),
            raw_value: "the flying stones".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        let parser = Parser::from_gazetteer(&gazetteer).unwrap();
        parser.dump(tdir.as_ref().join("parser")).unwrap();
        let reloaded_parser = Parser::from_folder(tdir.as_ref().join("parser")).unwrap();
        tdir.close().unwrap();
        assert!(parser.fst.equals(&reloaded_parser.fst));
    }

    #[test]
    fn test_parser_first() {
        let mut gazetteer = Gazetteer::new();
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
        let parser = Parser::from_gazetteer(&gazetteer).unwrap();
        // parser.dump("test_new_parser").unwrap();
        let mut parsed = parser
            .run("je veux écouter les rolling stones", 0.0)
            .unwrap();
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

        parsed = parser
            .run("je veux ecouter les \t rolling stones", 0.0)
            .unwrap();
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
            .run("i want to listen to rolling stones and blink eight", 0.0)
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

        parsed = parser.run("joue moi quelque chose", 0.0).unwrap();
        assert_eq!(parsed, vec![]);
    }

    #[test]
    fn test_parser_with_ranking() {
        /* Weight is here a proxy for the ranking of an artist in a popularity
        index */

        let mut gazetteer = Gazetteer::new();
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
        let parser = Parser::from_gazetteer(&gazetteer).unwrap();

        /* When there is a tie in terms of number of token matched, match the most popular choice */
        let parsed = parser.run("je veux écouter the stones", 0.5).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "the stones".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
                range: 16..26,
            }]
        );
        let parsed = parser.run("je veux écouter brel", 0.5).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "brel".to_string(),
                resolved_value: "Jacques Brel".to_string(),
                range: 16..20,
            }]
        );

        // Resolve to the value with more words matching regardless of popularity
        let parsed = parser
            .run("je veux écouter the flying stones", 0.5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "the flying stones".to_string(),
                resolved_value: "The Flying Stones".to_string(),
                range: 16..33,
            }]
        );
        let parsed = parser.run("je veux écouter daniel brel", 0.5).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "daniel brel".to_string(),
                resolved_value: "Daniel Brel".to_string(),
                range: 16..27,
            }]
        );
        let parsed = parser.run("je veux écouter jacques", 0.5).unwrap();
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
        let mut gazetteer = Gazetteer::new();
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        let parser = Parser::from_gazetteer(&gazetteer).unwrap();

        let parsed = parser
            .run("the music I want to listen to is rolling on stones", 0.5)
            .unwrap();
        assert_eq!(parsed, vec![]);
    }

    #[test]
    fn test_parser_with_unicode_whitespace() {
        let mut gazetteer = Gazetteer::new();
        gazetteer.add(EntityValue {
            resolved_value: "Quand est-ce ?".to_string(),
            raw_value: "quand est -ce".to_string(),
        });
        let parser = Parser::from_gazetteer(&gazetteer).unwrap();
        let parsed = parser.run("non quand est survivre", 0.5).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                resolved_value: "Quand est-ce ?".to_string(),
                range: 4..13,
                raw_value: "quand est".to_string(),
            }]
        )
    }

    #[test]
    #[ignore]
    fn test_parser_with_mixed_ordered_entity() {
        let mut gazetteer = Gazetteer::new();
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        let parser = Parser::from_gazetteer(&gazetteer).unwrap();

        let parsed = parser.run("rolling the stones", 0.5).unwrap();
        assert_eq!(parsed, vec![]);
    }

    #[test]
    fn test_parser_with_threshold() {
        let mut gazetteer = Gazetteer::new();
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
        let parser = Parser::from_gazetteer(&gazetteer).unwrap();
        // parser.dump("test_new_parser").unwrap();
        let parsed = parser
            .run("je veux écouter les rolling stones", 0.5)
            .unwrap();
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

        let parsed = parser
            .run("je veux écouter les rolling stones", 0.3)
            .unwrap();
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

        let parsed = parser
            .run("je veux écouter les rolling stones", 0.6)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "rolling stones".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
                range: 20..34,
            }]
        );
    }

    #[test]
    fn real_world_gazetteer() {
        let gaz = Gazetteer::from_json("local_testing/artist_gazeteer_formatted.json", None).unwrap();
        let parser = Parser::from_gazetteer(&gaz).unwrap();
        parser.dump("./test_artist_parser").unwrap();
    }
}
