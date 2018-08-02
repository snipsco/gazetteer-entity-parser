use std::cmp::max;
// use std::collections::{HashMap, HashSet};
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
// use fnv::FnvHashMap as HashMap;
// use fnv::FnvHashSet as HashSet;
use constants::RESTART_IDX;
use constants::*;
use data::EntityValue;
use data::Gazetteer;
use errors::GazetteerParserResult;
use failure::ResultExt;
use serde_json;
use snips_fst::string_paths_iterator::StringPath;
use snips_fst::arc_iterator::ArcIterator;
use symbol_table::GazetteerParserSymbolTable;
use snips_fst::{fst, operations};
use std::fs;
use std::ops::Range;
use std::path::Path;
use utils::whitespace_tokenizer;
use utils::{check_threshold, fst_format_resolved_value, fst_unformat_resolved_value};
use serde::{Serialize};
use rmps::{Serializer, from_read};

// type HashMap<K, V> = std::collections::HashMap<K, V, FnvHasher>;

/// Struct representing the parser. The `symbol_table` attribute holds the symbol table used
/// to create the parsing FSTs. The Parser will match the longest possible contiguous substrings
/// of a query that match entity values. The order in which the values are added to the parser
/// matters: In case of ambiguity between two parsings, the Parser will output the value that was
/// added first (see Gazetteer).
pub struct Parser {
    symbol_table: GazetteerParserSymbolTable,
    token_to_count: HashMap<u32, u32>, // maps each token to its count in the dataset
    token_to_resolved_values: HashMap<u32, HashSet<u32>>,  // maps token to set of resolved values containing token
    resolved_value_to_tokens: HashMap<u32, (u32, Vec<u32>)>,  // maps resolved value to a tuple (rank, tokens)
    stop_words: HashSet<u32>,
    edge_cases: HashSet<u32>  // values composed only of stop words
}

#[derive(Serialize, Deserialize)]
struct ParserConfig {
    symbol_table_filename: String,
    version: String,
    token_to_resolved_values: String,
    resolved_value_to_tokens: String,
    token_to_count: String,
    stop_words: String,
    edge_cases: String
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
    /// Create an empty parser. Its symbol table contains the minimal symbols used during parsing.
    fn new() -> GazetteerParserResult<Parser> {
        // Add a symbol table with epsilon and skip symbols
        let mut symbol_table = GazetteerParserSymbolTable::new();
        let eps_idx = symbol_table.add_symbol(EPS, false)?;
        if eps_idx != EPS_IDX {
            return Err(format_err!("Wrong epsilon index: {}", eps_idx));
        }
        let skip_idx = symbol_table.add_symbol(SKIP, false)?;
        if skip_idx != SKIP_IDX {
            return Err(format_err!("Wrong skip index: {}", skip_idx));
        }
        let consumed_idx = symbol_table.add_symbol(CONSUMED, false)?;
        if consumed_idx != CONSUMED_IDX {
            return Err(format_err!("Wrong consumed index: {}", consumed_idx));
        }
        let restart_idx = symbol_table.add_symbol(RESTART, false)?;
        if restart_idx != RESTART_IDX {
            return Err(format_err!("Wrong restart index: {}", restart_idx));
        }
        Ok(Parser { symbol_table, token_to_resolved_values: HashMap::default(), resolved_value_to_tokens: HashMap::default(), token_to_count: HashMap::default(), stop_words: HashSet::default(), edge_cases: HashSet::default() })
    }

    /// Add a single entity value to the parser. This function is kept private to promote
    /// creating the parser with a higher level function (such as `from_gazetteer`) that
    /// performs additional global optimizations.
    fn add_value(&mut self, entity_value: &EntityValue, rank: u32) -> GazetteerParserResult<()> {
        // We force add the new resolved value: even if it already is present in the symbol table
        // we duplicate it to allow several raw values to map to it
        let res_value_idx = self.symbol_table.add_symbol(&fst_format_resolved_value(&entity_value.resolved_value), true)?;
        // if self.resolved_value_to_tokens.contains_key(&res_value_idx) {
        //     bail!("Cannot add value {:?} twice to the parser");
        // }
        for (_, token) in whitespace_tokenizer(&entity_value.raw_value) {
            let token_idx = self.symbol_table.add_symbol(&token, false)?;

            // Update token_to_resolved_values map
            self.token_to_resolved_values.entry(token_idx)
                .and_modify(|e| {e.insert(res_value_idx);})
                .or_insert( {
                    let mut h = HashSet::default();
                    h.insert(res_value_idx);
                    h
                });

            // Update token count
            self.token_to_count.entry(token_idx)
                .and_modify(|c| {*c += 1} )
                .or_insert(1);

            // Update resolved_value_to_tokens map
            self.resolved_value_to_tokens.entry(res_value_idx)
            .and_modify(|(_, v)| v.push(token_idx))
            .or_insert((rank, vec![token_idx]));
        }
        Ok(())
    }

    /// Update an internal set of stop words, with frequency higher than threshold_freq in the
    /// gazetteer, and corresponding set of edge cases, containing values only composed of stop
    /// words
    pub fn compute_stop_words(&mut self, threshold_freq: f32) -> GazetteerParserResult<()> {
        // Update the set of stop words
        let n_resolved_values = self.resolved_value_to_tokens.len() as f32;
        for (token, count) in &self.token_to_count {
            if *count as f32 / n_resolved_values as f32 > threshold_freq {
                self.stop_words.insert(*token);
            }
        }
        // Update the set of edge_cases. i.e. resolved value that only contain stop words
        'outer: for (res_val, (_, tokens)) in &self.resolved_value_to_tokens {
            for tok in tokens {
                if !(self.stop_words.contains(tok)) {
                    continue 'outer
                }
            }
            self.edge_cases.insert(*res_val);
        }
        Ok(())
    }

    /// Get the set of stop words
    pub fn get_stop_words(&self) -> GazetteerParserResult<HashSet<String>> {
        self.stop_words.iter().map(|idx| self.symbol_table.find_index(*idx)).collect()
    }

    /// Get the set of edge cases, containing only stop words
    pub fn get_edge_cases(&self) -> GazetteerParserResult<HashSet<String>> {
        Ok(self.edge_cases.iter().map(|idx| {
            let symbol: String = self.symbol_table.find_index(*idx).unwrap();
            fst_unformat_resolved_value(&symbol)
        }).collect())
    }

    /// Create a Parser from a Gazetteer, which represents an ordered list of entity values.
    /// This function adds the entity values from the gazetteer. This is the recommended method
    /// to define a parser.
    pub fn from_gazetteer(gazetteer: &Gazetteer) -> GazetteerParserResult<Parser> {
        let mut parser = Parser::new()?;
        for (rank, entity_value) in gazetteer.data.iter().enumerate() {
            parser.add_value(entity_value, rank as u32)?;
        }
        Ok(parser)
    }


    /// get the admissible tokens
    // fn filter_possible_resolutions(possible_resolved_values_counts: &HashSet<i32>, decoding_threshold: f32) -> GazetteerParserResult<HashSet<i32>

    /// get resolved value (DEBUG)
    #[inline(never)]
    fn get_tokens_from_resolved_value(&self, resolved_value: &u32) -> GazetteerParserResult<&(u32, Vec<u32>)> {
        Ok(self.resolved_value_to_tokens.get(resolved_value).ok_or_else(|| format_err!("Missing value {:?} from resolved_value_to_tokens", resolved_value))?)
    }

    /// get resolved values from token
    #[inline(never)]
    fn get_resolved_values_from_token(&self, token: &u32) -> GazetteerParserResult<&HashSet<u32>> {
        Ok(self.token_to_resolved_values.get(token).ok_or_else(|| format_err!("Missing value {:?} from token_to_resolved_values", token))?)
    }

    /// get number of raw tokens corresponding to a resolved value
    #[inline(never)]
    fn get_n_tokens(&self, resolved_value: &u32) -> GazetteerParserResult<usize> {
        let (_, tokens) = self.get_tokens_from_resolved_value(resolved_value)?;
        Ok(tokens.len())
    }

    /// Get counts of tokens for the possible resolutions
    #[inline(never)]
    fn get_resolutions_counts(&self, input: &str) -> GazetteerParserResult<HashMap<u32, (u32, u32)>> {
        let mut possible_resolved_values_counts: HashMap<u32, (u32, u32)> = HashMap::default();
        // let mut possible_edge_values: HashSet<u32> = self.edge_cases;
        let mut skipped_tokens: HashSet<u32> = HashSet::default();
        let mut rarest_skipped_token: u32 = 0;  // Used to filter the possible edge cases
        let mut n_res_val_with_rarest_skipped_token: u32 = self.resolved_value_to_tokens.len() as u32;  // Used to filter the possible edge cases
        for (_, token) in whitespace_tokenizer(input) {
            // single tokens should only appear once in the symbol table
            // DEBUG
            // println!("TOKEN: {:?}", token);
            // println!("STOP WORDS: {:?}", self.stop_words);
            match self.symbol_table.find_single_symbol(&token)? {
                Some(value) => {

                    if self.stop_words.contains(&value) {
                        // Stop word: we add only the resolved values from the edge case list
                        // which contain all of the stop words so far
                        // if skipped_tokens.is_empty() {
                        //     possible_edge_values = self.get_resolved_values_from_token(&value)?.intersection(&self.edge_cases)
                        // }
                        skipped_tokens.insert(value);
                        let n_res_val = self.get_resolved_values_from_token(&value)?.len() as u32;
                        if n_res_val < n_res_val_with_rarest_skipped_token {
                            n_res_val_with_rarest_skipped_token = n_res_val;
                            rarest_skipped_token = value;
                        }
                        // let
                        // let new_possible_edge_values: HashSet<u32> = HashSet::default();
                        // for val in possible_edge_values {
                        //     if
                        // }
                        //
                        // for res_val in self.get_resolved_values_from_token(&value)? {
                        //     possible_edge_values.
                        // }
                        // possible_edge_values.intersection(self.get_resolved_values_from_token(&value)?);
                        // skipped_tokens.insert(value);
                        // for res_value in self.get_resolved_values_from_token(&value)?.intersection(&self.edge_cases) {
                        //     possible_resolved_values_counts.entry(*res_value)
                        //     .and_modify(|(_, count)| *count += 1)
                        //     .or_insert((self.get_n_tokens(res_value)? as u32, 1));
                        // }
                    } else {
                        // Not a stop word: we consider all possible resolved values containing
                        // the token
                        for res_value in self.get_resolved_values_from_token(&value)? {
                            possible_resolved_values_counts.entry(*res_value)
                            .and_modify(|(_, count)| *count += 1)
                            .or_insert((self.get_n_tokens(res_value)? as u32, 1));
                        }
                    }
                }
                None => continue
            };
        }

        // DEBUG
        // println!("SKIPPED TOKENS: {:?}", skipped_tokens);

        // add back the counts for the tokens that we skipped, for the values that are not edge
        // cases
        for val in possible_resolved_values_counts.clone().keys() {
            let (_, tokens) = self.get_tokens_from_resolved_value(&val)?;
            for tok in tokens {
                if skipped_tokens.contains(&tok) {
                    possible_resolved_values_counts.entry(*val)
                        .and_modify(|(_, count)| *count += 1);
                }
            }
        }

        // for token in &skipped_tokens {
        //     let resolved_values = self.get_resolved_values_from_token(&token)?;
        //     for res_value in resolved_values {
        //         if !(self.edge_cases.contains(res_value)) {
        //             possible_resolved_values_counts.entry(*res_value)
        //             .and_modify(|(_, count)| *count += 1);
        //         }
        //     }
        // }

        // Finally, add the admissible edge cases, i.e those such as all their tokens are included
        // in the skipped tokens.
        // We gain some time by restricting to the values containing the rarest skipped token we
        // found
        // let mut ordered_skipped_tokens = skipped_tokens.iter().map(|tok| (*tok, self.get_resolved_values_from_token(tok).unwrap().len())).collect::<Vec<(u32, usize)>>();
        // ordered_skipped_tokens.sort_by_key(|k| k.1);
        // DEBUG
        // println!("ORDERED SKIPPED TOKENS {:?}", ordered_skipped_tokens);
        // let mut edge_case_values_to_add: HashSet<u32> = HashSet::default();

        // DEBUG
        // println!("SKIPPED TOKENS {:?}", skipped_tokens);
        // println!("RAREST SKIPPED TOKEN {:?}", self.symbol_table.find_index(rarest_skipped_token));
        // 'outer: for val in self.get_resolved_values_from_token(&rarest_skipped_token)? {
        //     let (_, tokens) = self.get_tokens_from_resolved_value(val)?;
        //     for tok in tokens {
        //         if !skipped_tokens.contains(tok) {
        //             continue 'outer
        //         }
        //     }
        //     // // DEBUG
        //     println!("ADDING POSSIBLE RESOLVED VALUE {:?}", self.symbol_table.find_index(*val));
        //     possible_resolved_values_counts.entry(*val)
        //         .or_insert((tokens.len() as u32, tokens.len() as u32));
        // }

        // possible_resolved_values_counts.entry(100)
        //     .or_insert((1 as u32, 1 as u32));
        // possible_resolved_values_counts.entry(101)
        //     .or_insert((1 as u32, 1 as u32));


        // for (idx, (tok, _)) in ordered_skipped_tokens.iter().enumerate() {
        //     if idx == 0 {
        //         // Our starting point is the set of resolved values which contain the rarest stop
        //         // word, and whose tokens are included in the skipped tokens
        //         edge_case_values_to_add = self.get_resolved_values_from_token(&tok)?.clone();
        //     } else {
        //         let mut new_edge_case_values_to_add: HashSet<u32> = HashSet::default();
        //         for val in edge_case_values_to_add.intersection(self.get_resolved_values_from_token(&tok)?) {
        //             new_edge_case_values_to_add.insert(*val);
        //         }
        //         edge_case_values_to_add = new_edge_case_values_to_add;
        //     }
        // }
        // for val in edge_case_values_to_add {
        //     let (_, tokens) = self.get_tokens_from_resolved_value(&val)?;
        //     possible_resolved_values_counts.entry(val)
        //         .or_insert((tokens.len() as u32, tokens.len() as u32));
        // }

        // DEBUG
        // println!("NUM POSSIBLE VALUES BEFORE THRESHOLD {:?}", possible_resolved_values_counts.len());
        Ok(possible_resolved_values_counts)
    }


    /// We do a first pass on the input to check what the possible resolved values are
    /// (taking threshold into account)
    #[inline(never)]
    fn get_possible_resolutions(&self, input: &str, decoding_threshold: f32) -> GazetteerParserResult<(HashSet<u32>, HashSet<u32>)> {
        let mut possible_resolved_values: HashSet<u32> = HashSet::default();
        let mut admissible_tokens: HashSet<u32> = HashSet::default();

        let possible_resolved_values_counts = self.get_resolutions_counts(input)?;

        // From the counts, we extract the possible resolved values
        for (res_val, (len, count)) in &possible_resolved_values_counts {
            if check_threshold(*count, max(0, *len as i32 - *count as i32) as u32, decoding_threshold) {
                possible_resolved_values.insert(*res_val);
                for tok in &self.resolved_value_to_tokens.get(&res_val).ok_or_else(|| format_err!("Missing value {:?} from resolved_value_to_tokens", res_val))?.1 {
                    admissible_tokens.insert(*tok);
                }
            }
        }
        Ok((possible_resolved_values, admissible_tokens))
    }


    /// Create an input fst from a string to be parsed. Outputs the input fst and a vec of ranges
    /// of the tokens composing it, as well as a set of possible resolution based on the tokens
    /// found in the sentence. This set is used to build the smallest possible parser fst in a
    /// subsequent step.
    #[inline(never)]
    fn build_input_fst(&self, input: &str, decoding_threshold: f32) -> GazetteerParserResult<(fst::Fst, Vec<Range<usize>>, HashSet<u32>)> {
        // build the input fst
        let mut input_fst = fst::Fst::new();
        let mut tokens_ranges: Vec<Range<usize>> = vec![];
        let mut current_head = input_fst.add_state();
        input_fst.set_start(current_head);
        let mut restart_to_be_inserted: bool = false;

        let (possible_resolved_values, admissible_tokens) = self.get_possible_resolutions(input, decoding_threshold)?;

        // Then we actually create the input FST
        for (token_range, token) in whitespace_tokenizer(input) {
            match self.symbol_table.find_single_symbol(&token)? {
                Some(value) => {
                    if restart_to_be_inserted {
                        let next_head = input_fst.add_state();
                        input_fst.add_arc(current_head, EPS_IDX as i32, RESTART_IDX as i32, 0.0, next_head);
                        current_head = next_head;
                        restart_to_be_inserted = false;
                    }
                    if admissible_tokens.contains(&value) {
                        let next_head = input_fst.add_state();
                        input_fst.add_arc(current_head, value as i32, value as i32, 0.0, next_head);
                        tokens_ranges.push(token_range);
                        current_head = next_head;
                    } else {
                        restart_to_be_inserted = true;
                    }
                }
                None => {
                    // if the word is not in the symbol table, there is no
                    // chance of matching it: we skip
                    // we also signal that we should restart the parsing
                    restart_to_be_inserted = true;
                }
            }
        }
        // Set final state
        input_fst.set_final(current_head, 0.0);
        input_fst.arc_sort(false);
        Ok((input_fst, tokens_ranges, possible_resolved_values))
    }

    /// Return the one shortest path as a StringPath
    fn get_1_shortest_path(&self, fst: &fst::Fst,) -> GazetteerParserResult<StringPath> {
        let shortest_path = fst.shortest_path(1, false, false);

        let mut current_head = shortest_path.start();
        let mut input_tokens = Vec::new();
        let mut output_tokens = Vec::new();
        let mut weight = 0.0;
        let path = loop {
            let arc_iterator = ArcIterator::new( &shortest_path, current_head );
            let mut num_arcs = 0;
            let mut next_head = -1;
            for arc in arc_iterator {
                num_arcs += 1;
                if num_arcs > 1 {
                    bail!("Shortest path fst is not linear")
                }
                let ilabel = arc.ilabel() as u32;
                if ilabel != EPS_IDX {
                    input_tokens.push(self.symbol_table.find_index(ilabel)?);
                }
                let olabel = arc.olabel() as u32;
                if olabel != EPS_IDX {
                    output_tokens.push(self.symbol_table.find_index(olabel)?);
                }
                weight += arc.weight();
                next_head = arc.nextstate();
            }
            if shortest_path.is_final(current_head) {
                if num_arcs > 0 {
                    bail!("Final state with outgoing arc!")
                } else {
                    weight += match shortest_path.final_weight(current_head) {
                        Some(value) => value,
                        None => 0.0
                    };
                    break StringPath {istring: input_tokens.join(" "), ostring: output_tokens.join(" "), weight};
                }
            }
            current_head = next_head;
        };

        Ok(path)
    }


    /// Format the shortest path as a vec of ParsedValue
    #[inline(never)]
    fn format_string_path(
        &self,
        string_path: &StringPath,
        tokens_range: &Vec<Range<usize>>,
        threshold: f32,
    ) -> GazetteerParserResult<Vec<ParsedValue>> {

        let mut parsed_values: Vec<ParsedValue> = vec![];
        let mut output_iterator = whitespace_tokenizer(&string_path.ostring);
        let mut input_iterator = whitespace_tokenizer(&string_path.istring);
        let mut input_ranges_iterator = tokens_range.iter();

        // Assumptions: the output values can be <skip>, <consumed>, <restart>, or tokens
        // The parsed value always comes after a certain number of consumed and skips symbols
        'outer: loop {
            // First we consume the output values corresponding to the same parsed value, and
            // compute the number of tokens consumed and skipped
            let mut n_consumed_tokens = 0;
            let mut n_skips = 0;

            let current_resolved_value = 'output: loop {
                match output_iterator.next() {
                    Some((_, value)) => {
                        match value.as_ref() {
                            CONSUMED => {n_consumed_tokens += 1;}
                            SKIP => {n_skips += 1;}
                            val => {break 'output val.to_string()}
                        }
                    }
                    None => {break 'outer}
                }
            };
            // Then we accumulate the tokens corresponing to this parsed value
            let mut input_value_until_now: Vec<String> = vec![];
            let mut current_ranges: Vec<&Range<usize>> = vec![];
            for _ in 0..n_consumed_tokens {
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
            // If the threshold condition is met, we append the parsed value to the results
            if check_threshold(n_consumed_tokens, n_skips, threshold) {
                let range_start = current_ranges.first().unwrap().start;
                let range_end = current_ranges.last().unwrap().end;
                parsed_values.push(ParsedValue {
                    raw_value: input_value_until_now.join(" "),
                    resolved_value: fst_unformat_resolved_value(&current_resolved_value),
                    range: range_start..range_end,
                });
            }
        }
        Ok(parsed_values)
    }

    /// Compute the weight of a resolved entity based on its rank. The formula is designed to
    /// verify two properties:
    /// 1) It must be an increasing function of the rank
    /// 2) for rank >= 0, the weight must be in [0, 1[. Because the weight associated to skipping
    /// a token in the parsing is set to 1.0, this condition ensures that the parsed value is
    /// always the one corresponding to the longest match in the input string, regardless of the
    /// rank, which is only here to break ties.
    fn compute_resolved_value_weight(rank: u32) -> GazetteerParserResult<f32> {
        Ok(1.0 - 1.0 / (1.0 + rank as f32)) // Adding 1 ensures 1 + rank is > 0
    }

    /// Build the parser FST for the current sentence. This FST can skip tokens and always
    /// outputs the parsed value with the longest match in the input string. In case of a tie,
    /// the parsed value is the one with the smallest rank in the gazetteer used to build the
    /// parser.
    #[inline(never)]
    fn build_parser_fst(&self, possible_resolved_values: HashSet<u32>) -> GazetteerParserResult<fst::Fst> {
        let mut resolver_fst = fst::Fst::new();
        // add a start state
        let start_state = resolver_fst.add_state();
        resolver_fst.set_start(start_state);
        // add a restart state. When the input fst contains a RESTART symbol, we force
        // the parser to come back to this state
        let restart_state = resolver_fst.add_state();
        resolver_fst.add_arc(start_state, RESTART_IDX as i32, EPS_IDX as i32, 0.0, restart_state);
        // We can also go back to the start state to match a new value without consuming a restart
        // token. This is useful to match things like "I like to listen to the rollings stones,
        // the beatles" without a transition word between the values. However, we weight this
        // transition to promote matching the longest possible substring from the input.
        // This weight should be at least two to compensate for the weight of skipping one token
        // and any rank-induced differences of weights between the competing parsings.
        // (see test_match_longest_substring for examples)
        resolver_fst.add_arc(start_state, EPS_IDX as i32, EPS_IDX as i32, 2.0, restart_state);
        for res_val in possible_resolved_values {
            let mut current_head = restart_state;
            let (rank, tokens) = self.resolved_value_to_tokens.get(&res_val).ok_or_else(|| format_err!("Error when building the pareser FST: could not find the resolved value {:?} in the possible_resolved_values hashmap", res_val))?;
            for tok in tokens {
                let next_head = resolver_fst.add_state();
                resolver_fst.add_arc(current_head, *tok as i32, CONSUMED_IDX as i32, 0.0, next_head);
                resolver_fst.add_arc(current_head, EPS_IDX as i32, SKIP_IDX as i32, 1.0, next_head);
                current_head = next_head;
            }
            let final_head = resolver_fst.add_state();
            resolver_fst.add_arc(current_head, EPS_IDX as i32, res_val as i32, 0.0, final_head);
            // we set the weight of the final state using the rank information
            resolver_fst.set_final(final_head, Self::compute_resolved_value_weight(*rank)?);
        }

        // Add the edge cases without eps skips
        for res_val in &self.edge_cases {
            let mut current_head = restart_state;
            let (rank, tokens) = self.resolved_value_to_tokens.get(&res_val).ok_or_else(|| format_err!("Error when building the pareser FST: could not find the resolved value {:?} in the possible_resolved_values hashmap", res_val))?;
            for tok in tokens {
                let next_head = resolver_fst.add_state();
                resolver_fst.add_arc(current_head, *tok as i32, CONSUMED_IDX as i32, 0.0, next_head);
                current_head = next_head;
            }
            let final_head = resolver_fst.add_state();
            resolver_fst.add_arc(current_head, EPS_IDX as i32, *res_val as i32, 0.0, final_head);
            // we set the weight of the final state using the rank information
            resolver_fst.set_final(final_head, Self::compute_resolved_value_weight(*rank)?);
        }

        // We do not optimize the fst (this actually saves some time)
        // We take the closure of the parser fst to allow parsing several values in the same
        // input sentence
        // DEBUG
        // resolver_fst.optimize();
        resolver_fst.closure_plus();
        resolver_fst.arc_sort(true);
        Ok(resolver_fst)
    }

    /// Parse the input string `input` and output a vec of `ParsedValue`. `decoding_threshold` is
    /// the minimum fraction of raw tokens to match for an entity to be parsed.
    pub fn run(
        &self,
        input: &str,
        decoding_threshold: f32,
    ) -> GazetteerParserResult<Vec<ParsedValue>> {

        let (input_fst, tokens_range, possible_resolved_values) = self.build_input_fst(input, decoding_threshold)?;
        // println!("NUM POSSIBLE RESOLVED VALUES: {:?}", possible_resolved_values.len());
        let resolver_fst = self.build_parser_fst(possible_resolved_values)?;
        let composition = operations::compose(&input_fst, &resolver_fst);
        if composition.num_states() == 0 {
            return Ok(vec![]);
        }
        let shortest_path = self.get_1_shortest_path(&composition)?;
        let parsing = self.format_string_path(&shortest_path, &tokens_range, decoding_threshold)?;
        Ok(parsing)
    }

    fn get_parser_config(&self) -> ParserConfig {
        ParserConfig {
            symbol_table_filename: SYMBOLTABLE_FILENAME.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            resolved_value_to_tokens: RESOLVED_VALUE_TO_TOKENS.to_string(),
            token_to_resolved_values: TOKEN_TO_RESOLVED_VALUES.to_string(),
            token_to_count: TOKEN_TO_COUNT.to_string(),
            stop_words: STOP_WORDS.to_string(),
            edge_cases: EDGE_CASES.to_string()
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
        let writer = fs::File::create(folder_name.as_ref().join(METADATA_FILENAME))?;
        serde_json::to_writer(writer, &config)?;

        self.symbol_table.write_file(
            folder_name.as_ref().join(config.symbol_table_filename)
        )?;

        let mut token_to_res_val_writer = fs::File::create(folder_name.as_ref().join(config.token_to_resolved_values))?;
        self.token_to_resolved_values.serialize(&mut Serializer::new(&mut token_to_res_val_writer))?;

        let mut resolved_value_to_tokens_writer =
        fs::File::create(folder_name.as_ref().join(config.resolved_value_to_tokens))?;
        self.resolved_value_to_tokens.serialize(&mut Serializer::new(&mut resolved_value_to_tokens_writer))?;

        let mut token_to_count_writer =
        fs::File::create(folder_name.as_ref().join(config.token_to_count))?;
        self.token_to_count.serialize(&mut Serializer::new(&mut token_to_count_writer))?;

        let mut stop_words_writer =
        fs::File::create(folder_name.as_ref().join(config.stop_words))?;
        self.stop_words.serialize(&mut Serializer::new(&mut stop_words_writer))?;

        let mut edge_cases_writer =
        fs::File::create(folder_name.as_ref().join(config.edge_cases))?;
        self.edge_cases.serialize(&mut Serializer::new(&mut edge_cases_writer))?;

        Ok(())
    }

    /// Load a resolver from a folder
    pub fn from_folder<P: AsRef<Path>>(folder_name: P) -> GazetteerParserResult<Parser> {
        let metadata_path = folder_name.as_ref().join(METADATA_FILENAME);
        let metadata_file = fs::File::open(&metadata_path)
            .with_context(|_| format!("Cannot open metadata file {:?}", metadata_path))?;
        let config: ParserConfig = serde_json::from_reader(metadata_file)?;

        let symbol_table = GazetteerParserSymbolTable::from_path(
            folder_name.as_ref().join(config.symbol_table_filename)
        )?;
        let token_to_res_val_path = folder_name.as_ref().join(config.token_to_resolved_values);
        let token_to_res_val_file = fs::File::open(&token_to_res_val_path)
            .with_context(|_| format!("Cannot open metadata file {:?}", token_to_res_val_path))?;
        let token_to_resolved_values = from_read(token_to_res_val_file)?;

        let res_val_to_tokens_path = folder_name.as_ref().join(config.resolved_value_to_tokens);
        let res_val_to_tokens_file = fs::File::open(&res_val_to_tokens_path)
            .with_context(|_| format!("Cannot open metadata file {:?}", res_val_to_tokens_path))?;
        let resolved_value_to_tokens = from_read(res_val_to_tokens_file)?;

        let token_to_count_path = folder_name.as_ref().join(config.token_to_count);
        let token_to_count_file = fs::File::open(&token_to_count_path)
            .with_context(|_| format!("Cannot open metadata file {:?}", token_to_count_path))?;
        let token_to_count = from_read(token_to_count_file)?;

        let stop_words_path = folder_name.as_ref().join(config.stop_words);
        let stop_words_file = fs::File::open(&stop_words_path)
            .with_context(|_| format!("Cannot open metadata file {:?}", stop_words_path))?;
        let stop_words = from_read(stop_words_file)?;

        let edge_cases_path = folder_name.as_ref().join(config.edge_cases);
        let edge_cases_file = fs::File::open(&edge_cases_path)
            .with_context(|_| format!("Cannot open metadata file {:?}", edge_cases_path))?;
        let edge_cases = from_read(edge_cases_file)?;

        Ok(Parser { symbol_table, token_to_resolved_values, resolved_value_to_tokens, token_to_count, stop_words, edge_cases })
    }
}

#[cfg(test)]
mod tests {
    extern crate tempfile;
    extern crate mio_httpc;

    use self::mio_httpc::CallBuilder;
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
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the stones".to_string(),
        });
        let mut parser = Parser::from_gazetteer(&gazetteer).unwrap();
        parser.compute_stop_words(0.51);

        parser.dump(tdir.as_ref().join("parser")).unwrap();
        let reloaded_parser = Parser::from_folder(tdir.as_ref().join("parser")).unwrap();
        tdir.close().unwrap();

        // compare resolved_value_to_tokens
        assert!(parser.resolved_value_to_tokens == reloaded_parser.resolved_value_to_tokens);
        // compare token_to_resolved_values
        assert!(parser.token_to_resolved_values == reloaded_parser.token_to_resolved_values);
        // Compare symbol tables
        assert!(parser.symbol_table == reloaded_parser.symbol_table);
        // Compare token counts
        assert!(parser.token_to_count == reloaded_parser.token_to_count);
        // Compare stop words
        assert!(parser.stop_words == reloaded_parser.stop_words);
        // Compare edge cases
        assert!(parser.edge_cases == reloaded_parser.edge_cases);
    }

    #[test]
    fn test_stop_words_and_edge_cases() {
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
            resolved_value: "The Stones".to_string(),
            raw_value: "the stones".to_string(),
        });
        let mut parser = Parser::from_gazetteer(&gazetteer).unwrap();
        parser.compute_stop_words(0.51);
        let mut gt_stop_words: HashSet<u32> = HashSet::default();
        gt_stop_words.insert(parser.symbol_table.find_single_symbol("the").unwrap().unwrap());
        gt_stop_words.insert(parser.symbol_table.find_single_symbol("stones").unwrap().unwrap());
        assert!(parser.stop_words == gt_stop_words);
        let mut gt_edge_cases: HashSet<u32> = HashSet::default();
        gt_edge_cases.insert(parser.symbol_table.find_single_symbol(&fst_format_resolved_value("The Stones")).unwrap().unwrap());
        assert!(parser.edge_cases == gt_edge_cases);
    }

    #[test]
    fn test_parser_base() {
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
    fn test_parser_multiple_raw_values() {
        let mut gazetteer = Gazetteer::new();
        gazetteer.add(EntityValue {
            resolved_value: "Blink-182".to_string(),
            raw_value: "blink one eight two".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Blink-182".to_string(),
            raw_value: "blink 182".to_string(),
        });
        let parser = Parser::from_gazetteer(&gazetteer).unwrap();
        let mut parsed = parser
            .run("let's listen to blink 182", 0.0)
            .unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    raw_value: "blink 182".to_string(),
                    resolved_value: "Blink-182".to_string(),
                    range: 16..25,
                }
            ]
        );

        parsed = parser
            .run("let's listen to blink", 0.5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    raw_value: "blink".to_string(),
                    resolved_value: "Blink-182".to_string(),
                    range: 16..21,
                }
            ]
        );

        parsed = parser
            .run("let's listen to blink one two", 0.5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    raw_value: "blink one two".to_string(),
                    resolved_value: "Blink-182".to_string(),
                    range: 16..29,
                }
            ]
        );
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
    fn test_parser_with_mixed_ordered_entity() {
        let mut gazetteer = Gazetteer::new();
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        let parser = Parser::from_gazetteer(&gazetteer).unwrap();

        let parsed = parser.run("rolling the stones", 0.5).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                resolved_value: "The Rolling Stones".to_string(),
                range: 8..18,
                raw_value: "the stones".to_string()}]
            );
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
    fn test_repeated_words() {
        let mut gazetteer = Gazetteer::new();
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        let parser = Parser::from_gazetteer(&gazetteer).unwrap();

        let parsed = parser
            .run("the the the", 0.5)
            .unwrap();
        assert_eq!(parsed, vec![]);

        let parsed = parser
            .run("the the the rolling stones stones stones stones", 1.0)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "the rolling stones".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
                range: 8..26,
            }]
        );
    }

    #[test]
    fn test_match_longest_substring() {
        let mut gazetteer = Gazetteer::new();
        gazetteer.add(EntityValue {
            resolved_value: "Black And White".to_string(),
            raw_value: "black and white".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Album".to_string(),
            raw_value: "album".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Black and White Album".to_string(),
            raw_value: "the black and white album".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "1 2 3 4".to_string(),
            raw_value: "one two three four".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "3 4 5 6".to_string(),
            raw_value: "three four five six".to_string(),
        });
        let parser = Parser::from_gazetteer(&gazetteer).unwrap();

        let parsed = parser
            .run("je veux écouter le black and white album", 0.5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "black and white album".to_string(),
                resolved_value: "The Black and White Album".to_string(),
                range: 19..40,
            }]
        );

        let parsed = parser
            .run("one two three four", 0.5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "one two three four".to_string(),
                resolved_value: "1 2 3 4".to_string(),
                range: 0..18,
            }]
        );

        // This last test is ambiguous and there may be several acceptable answers...
        let parsed = parser
            .run("one two three four five six", 0.5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "one two".to_string(),
                resolved_value: "1 2 3 4".to_string(),
                range: 0..7,
            },
            ParsedValue {
                raw_value: "three four five six".to_string(),
                resolved_value: "3 4 5 6".to_string(),
                range: 8..27,
            }]
        );

    }

    #[test]
    fn real_world_gazetteer() {

        // let (_, body) = CallBuilder::get().max_response(20000000).timeout_ms(60000).url("https://s3.amazonaws.com/snips/nlu-lm/test/gazetteer-entity-parser/artist_gazetteer_formatted.json").unwrap().exec().unwrap();
        // let data: Vec<EntityValue> = serde_json::from_reader(&*body).unwrap();
        // let gaz = Gazetteer{ data };
        // DEBUG
        let gaz = Gazetteer::from_json("local_testing/artist_gazetteer_formatted.json", None).unwrap();

        let mut parser = Parser::from_gazetteer(&gaz).unwrap();
        let fraction = 0.005;
        parser.compute_stop_words(fraction);
        println!("FRACTION {:?}", fraction);
        println!("ARTIST GAZETTEER, STOP WORDS {:?}", parser.stop_words.len());
        println!("ARTIST GAZETTEER, EDGE CASES {:?}", parser.edge_cases.len());

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
        let parsed = parser
            .run("je veux écouter bowie", 0.5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "bowie".to_string(),
                resolved_value: "David Bowie".to_string(),
                range: 16..21,
            }]
        );

        // let (_, body) = CallBuilder::get().max_response(20000000).timeout_ms(60000).url("https://s3.amazonaws.com/snips/nlu-lm/test/gazetteer-entity-parser/album_gazetteer_formatted.json").unwrap().exec().unwrap();
        // let data: Vec<EntityValue> = serde_json::from_reader(&*body).unwrap();
        // let gaz = Gazetteer{ data };
        // DEBUG
        let gaz = Gazetteer::from_json("local_testing/album_gazetteer_formatted.json", None).unwrap();

        let mut parser = Parser::from_gazetteer(&gaz).unwrap();
        let fraction = 0.01;
        parser.compute_stop_words(fraction);
        println!("FRACTION {:?}", fraction);
        println!("ALBUM GAZETTEER, STOP WORDS {:?}", parser.stop_words.len());
        println!("ALBUM GAZETTEER, EDGE CASES {:?}", parser.edge_cases.len());

        let parsed = parser
            .run("je veux écouter le black and white album", 0.6)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "black and white album".to_string(),
                resolved_value: "The Black and White Album".to_string(),
                range: 19..40,
            }]
        );
        let parsed = parser
            .run("je veux écouter dark side of the moon", 0.6)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "dark side of the moon".to_string(),
                resolved_value: "Dark Side of the Moon".to_string(),
                range: 16..37,
            }]
        );
    }
}
