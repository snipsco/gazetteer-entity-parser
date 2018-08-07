use std::cmp::Ordering;
use std::collections::hash_map::{Entry};
use fnv::FnvHashMap as HashMap;
use fnv::FnvHashSet as HashSet;
use std::collections::BinaryHeap;
use data::EntityValue;
use data::Gazetteer;
use constants::*;
use errors::GazetteerParserResult;
use failure::ResultExt;
use serde_json;
use symbol_table::GazetteerParserSymbolTable;
use std::fs;
use std::ops::Range;
use std::path::Path;
use utils::{whitespace_tokenizer, check_threshold};
use serde::{Serialize};
use rmps::{Serializer, from_read};


/// Struct representing the parser. The Parser will match the longest possible contiguous
/// substrings of a query that match partial entity values. The order in which the values are
/// added to the parser matters: In case of ambiguity between two parsings, the Parser will output
/// the value that was added first (see Gazetteer).
#[derive(PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Parser {
    tokens_symbol_table: GazetteerParserSymbolTable, // Symbol table for the raw tokens
    resolved_symbol_table: GazetteerParserSymbolTable, // Symbol table for the resvoled values
    // The latter differs from the first one in that it can contain the same resolved value
    // multiple times (to allow for multiple raw values corresponidng to the same resvoled value)
    token_to_count: HashMap<u32, u32>, // maps each token to its count in the dataset
    token_to_resolved_values: HashMap<u32, HashSet<u32>>,  // maps token to set of resolved values containing token
    resolved_value_to_tokens: HashMap<u32, (u32, Vec<u32>)>,  // maps resolved value to a tuple (rank, tokens)
    n_stop_words: usize,  // number of stop words to extract from the entity data
    additional_stop_words: Vec<String>, // external list of stop words
    stop_words: HashSet<u32>,
    edge_cases: HashSet<u32>,  // values composed only of stop words
    injected_values: HashSet<String>  // Keep track of values injected thus far
}

#[derive(Serialize, Deserialize)]
struct ParserConfig {
    version: String,
    parser_filename: String,
}

/// Struct holding a possible match that can be grown by iterating over the input tokens
#[derive(Debug, PartialEq, Eq, Clone)]
struct PossibleMatch {
    resolved_value: u32,
    range: Range<usize>,
    tokens_range: Range<usize>,
    raw_value_length: u32,
    n_consumed_tokens: u32,
    last_token_in_input: usize,
    last_token_in_resolution: usize,
    rank: u32
}

impl Ord for PossibleMatch {
    fn cmp(&self, other: &PossibleMatch) -> Ordering {
        if self.n_consumed_tokens < other.n_consumed_tokens {
            Ordering::Less
        } else if self.n_consumed_tokens > other.n_consumed_tokens {
            Ordering::Greater
        } else {
            if self.raw_value_length < other.raw_value_length {
                Ordering::Greater
            } else if self.raw_value_length > other.raw_value_length {
                Ordering::Less
            } else {
                // println!("Using rank to compare {:?} and {:?}", self, other);
                other.rank.cmp(&self.rank)
            }
        }
    }
}

impl PartialOrd for PossibleMatch {
    fn partial_cmp(&self, other: &PossibleMatch) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Struct holding an individual parsing result. The result of a run of the parser on a query
/// will be a vector of ParsedValue. The `range` attribute is the range of the characters
/// composing the raw value in the input query.
#[derive(Debug, PartialEq, Eq)]
pub struct ParsedValue {
    pub resolved_value: String,
    pub range: Range<usize>, // character-level
    pub raw_value: String,
}

impl Ord for ParsedValue {
    fn cmp(&self, other: &ParsedValue) -> Ordering {
        match self.partial_cmp(other) {
            Some(value) => value,
            None => panic!("Parsed value are not comaparable: {:?}, {:?}", self, other)
        }
    }
}

impl PartialOrd for ParsedValue {
    fn partial_cmp(&self, other: &ParsedValue) -> Option<Ordering> {
        if self.range.end < other.range.start {
            Some(Ordering::Less)
        } else if self.range.start > other.range.end {
            Some(Ordering::Greater)
        } else {
            None
        }
    }
}

impl Parser {
    /// Create an empty parser. Its symbol table contains the minimal symbols used during parsing.
    fn new() -> GazetteerParserResult<Parser> {
        Ok(Parser { tokens_symbol_table: GazetteerParserSymbolTable::new(),
            resolved_symbol_table: GazetteerParserSymbolTable::new(), token_to_resolved_values: HashMap::default(), resolved_value_to_tokens: HashMap::default(), token_to_count: HashMap::default(), stop_words: HashSet::default(), edge_cases: HashSet::default(), n_stop_words: 0, additional_stop_words: Vec::default(), injected_values: HashSet::default() })
    }

    /// Add a single entity value to the parser. This function is kept private to promote
    /// creating the parser with a higher level function (such as `from_gazetteer`) that
    /// performs additional global optimizations.
    fn add_value(&mut self, entity_value: &EntityValue, rank: u32) -> GazetteerParserResult<()> {
        // We force add the new resolved value: even if it already is present in the symbol table
        // we duplicate it to allow several raw values to map to it
        let res_value_idx = self.resolved_symbol_table.add_symbol(&entity_value.resolved_value, true)?;
        // if self.resolved_value_to_tokens.contains_key(&res_value_idx) {
        //     bail!("Cannot add value {:?} twice to the parser");
        // }
        for (_, token) in whitespace_tokenizer(&entity_value.raw_value) {
            let token_idx = self.tokens_symbol_table.add_symbol(&token, false)?;

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

    /// Update an internal set of stop words and corresponding edge cases.
    /// The set of stop words is made of the `n_stop_words` most frequent raw tokens in the
    /// gazetteer used to generate the parser. An optional `additional_stop_words` vector of
    /// strings can be added to the stop words. The edge cases are defined to the be the resolved
    /// values whose raw value is composed only of stop words. There are examined seperately
    /// during parsing, and will match if and only if they are present verbatim in the input
    /// string.
    pub fn set_stop_words(&mut self, n_stop_words: usize, additional_stop_words: Option<Vec<&str>>) -> GazetteerParserResult<()> {
        // Update the set of stop words with the most frequent words in the gazetteer
        // Reset stop words
        self.stop_words = HashSet::default();
        if n_stop_words > 0 {
            self.n_stop_words = n_stop_words;
            let mut smallest_count: u32 = <u32>::max_value();
            let mut tok_with_smallest_count: u32 = 0;
            let mut largest_counts: HashMap<u32, u32> = HashMap::default();
            for (token, count) in &self.token_to_count {
                if self.stop_words.len() < n_stop_words {
                    self.stop_words.insert(*token);
                    largest_counts.insert(*token, *count);
                    if count < &smallest_count {
                        smallest_count = *count;
                        tok_with_smallest_count = *token;
                    }
                } else {
                    if count > &smallest_count {
                        self.stop_words.remove(&tok_with_smallest_count);
                        largest_counts.remove(&tok_with_smallest_count);
                        self.stop_words.insert(*token);
                        largest_counts.insert(*token, *count);
                        smallest_count = *count;
                        tok_with_smallest_count = *token;
                        for (prev_tok, prev_count) in &largest_counts {
                            if prev_count < &smallest_count {
                                smallest_count = *prev_count;
                                tok_with_smallest_count = *prev_tok;
                            }
                        }
                    }
                }
            }
        }

        // add the words from the `additional_stop_words` vec (and potentially add them to
        // the symbol table)
        // Reset edhe cases
        self.edge_cases = HashSet::default();
        if let Some(additional_stop_words_vec) = additional_stop_words {
            self.additional_stop_words = additional_stop_words_vec.iter().map(|s| s.to_string()).collect();
            for tok_s in &additional_stop_words_vec {
                let tok_idx = self.tokens_symbol_table.add_symbol(tok_s, false)?;
                self.stop_words.insert(tok_idx);
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
        self.stop_words.iter().map(|idx| self.tokens_symbol_table.find_index(idx)).collect()
    }

    /// Get the set of edge cases, containing only stop words
    pub fn get_edge_cases(&self) -> GazetteerParserResult<HashSet<String>> {
        Ok(self.edge_cases.iter().map(|idx| {
            self.resolved_symbol_table.find_index(idx).unwrap()
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

    /// Add new values to an already trained Parser. This function is used for entity injection.
    /// It takes as arguments a vector of EntityValue's to inject, and a boolean indicating
    /// whether the new values should be prepended to the already existing values (`prepend=true`)
    /// or appended (`prepend=false`). Setting `from_vanilla` to true allows to remove all
    /// previously injected values before adding the new ones.
    pub fn inject_new_values(&mut self, new_values: &Vec<EntityValue>, prepend: bool, from_vanilla: bool) -> GazetteerParserResult<()> {

        if from_vanilla {
            // Remove the resolved values form the resolved_symbol_table
            // Remove the resolved value from the resolved_value_to_tokens map
            // remove the corresponding resolved value from the token_to_resolved_value map
            // if after removing, a token is left absent from all resolved entities, then remove
            // it from the tokens_count, tokens_to_resolved_value maps and from the
            // tokens_symbol_table
            let mut tokens_marked_for_removal: HashSet<u32> = HashSet::default();
            for val in &self.injected_values {
                if let Some(res_val_indices) = self.resolved_symbol_table.remove_symbol(&val)? {
                    for res_val in res_val_indices {
                        let (_, tokens) = self.get_tokens_from_resolved_value(&res_val)?.clone();
                        self.resolved_value_to_tokens.remove(&res_val);
                        for tok in tokens {
                            self.token_to_resolved_values.entry(tok)
                                .and_modify(|v| {v.remove(&res_val);});
                            // Check the remaining resolved values containing the token
                            if self.get_resolved_values_from_token(&tok)?.len() == 0 {
                                tokens_marked_for_removal.insert(tok);
                            }
                        }
                    }
                }
            }
            for tok_idx in tokens_marked_for_removal {
                let tok = self.tokens_symbol_table.find_index(&tok_idx)?;
                if let Some(tok_indices) = self.tokens_symbol_table.remove_symbol(&tok)? {
                    for idx in tok_indices {
                        self.token_to_resolved_values.remove(&idx);
                        self.token_to_count.remove(&idx);
                    }
                }
            }
        }

        if prepend {
            // update rank of previous values
            let n_new_values = new_values.len() as u32;
            for res_val in self.resolved_symbol_table.get_all_indices() {
                self.resolved_value_to_tokens.entry(*res_val)
                    .and_modify(|(rank, _)| *rank += n_new_values);
            }
        }

        let new_start_rank = match prepend {
            true => 0, // we inject new values from rank 0 to n_new_values - 1
            false => self.resolved_value_to_tokens.len()  // we inject new values from the current
            // last rank onwards
        } as u32;

        for (rank, entity_value) in new_values.iter().enumerate() {
            self.add_value(entity_value, new_start_rank + rank as u32)?;
            self.injected_values.insert(entity_value.resolved_value.clone());
        }

        // Update the stop words and edge cases
        let n_stop_words = self.n_stop_words.clone();
        let additional_stop_words = self.additional_stop_words.clone();
        self.set_stop_words(n_stop_words, Some(additional_stop_words.iter().map(|s| s.as_str()).collect()))?;

        Ok(())
    }

    /// get resolved value
    #[inline(never)]
    fn get_tokens_from_resolved_value(&self, resolved_value: &u32) -> GazetteerParserResult<&(u32, Vec<u32>)> {
        Ok(self.resolved_value_to_tokens.get(resolved_value).ok_or_else(|| format_err!("Missing value {:?} from resolved_value_to_tokens", resolved_value))?)
    }

    /// get resolved values from token
    #[inline(never)]
    fn get_resolved_values_from_token(&self, token: &u32) -> GazetteerParserResult<&HashSet<u32>> {
        Ok(self.token_to_resolved_values.get(token).ok_or_else(|| format_err!("Missing value {:?} from token_to_resolved_values", token))?)
    }

    /// Find all possible matches in a string.
    /// Returns a hashmap, indexed by resvolved values. The corresponding value is a vec of tuples
    /// each tuple is a possible match for the resvoled value, and is made of
    // (range of match, number of skips, index of last matched token in the resolved value)
    #[inline(never)]
    fn find_possible_matches(&self, input: &str, threshold: f32) -> GazetteerParserResult<BinaryHeap<PossibleMatch>> {
        let mut possible_matches: HashMap<u32, PossibleMatch> = HashMap::with_capacity_and_hasher(1000, Default::default());
        let mut matches_heap: BinaryHeap<PossibleMatch> = BinaryHeap::default();
        let mut skipped_tokens: HashMap<usize, (Range<usize>, u32)> = HashMap::default();
        'outer: for (token_idx, (range, token)) in whitespace_tokenizer(input).enumerate() {
            let range_start = range.start;
            let range_end = range.end;
            match self.tokens_symbol_table.find_single_symbol(&token)? {
                Some(value) => {

                    if !self.stop_words.contains(&value) {
                        for res_val in self.get_resolved_values_from_token(&value)? {
                            let entry = possible_matches.entry(*res_val);
                            match entry {
                                Entry::Occupied(mut entry) =>  self.update_previous_match(&mut *entry.get_mut(), res_val, token_idx, value, range_start, range_end, threshold, &mut matches_heap),
                                Entry::Vacant(entry) => {
                                    if let Some(new_possible_match) = self.insert_new_possible_match(res_val, value, range_start, range_end, token_idx, threshold, &skipped_tokens)? {
                                        entry.insert(new_possible_match);
                                    }
                                }
                            }
                        }
                    } else {
                        skipped_tokens.insert(token_idx, (range_start..range_end, value));
                        // Iterate over all edge cases and try to add or update corresponding
                        // PossibleMatch's
                        let res_vals_from_token = self.get_resolved_values_from_token(&value)?;
                        for res_val in &self.edge_cases {
                            if res_vals_from_token.contains(&res_val) {
                                let entry = possible_matches.entry(*res_val);
                                match entry {
                                    Entry::Occupied(mut entry) =>  self.update_previous_match(&mut *entry.get_mut(), res_val, token_idx, value, range_start, range_end, 1.0, &mut matches_heap),
                                    Entry::Vacant(entry) => {
                                        if let Some(new_possible_match) = self.insert_new_possible_match(res_val, value, range_start, range_end, token_idx, 1.0, &skipped_tokens)? {
                                            entry.insert(new_possible_match);
                                        }
                                    }
                                }
                            }
                        }
                        // Iterate over current possible matches containing the stop word and
                        // try to grow them (but do not initiate a new possible match)
                        for (res_val, mut possible_match) in &mut possible_matches {
                            if res_vals_from_token.contains(res_val) {
                                self.update_previous_match(&mut possible_match, res_val, token_idx, value, range_start, range_end, threshold, &mut matches_heap)
                            }
                        }
                    }
                },
                None => continue
            }
        }

        // Add to the heap the possible matches that remain
        for possible_match in possible_matches.values() {
            // We start by adding the PossibleMatch to the heap
            if possible_match.n_consumed_tokens > possible_match.raw_value_length {
                bail!("Consumed more tokens than are available: {:?}", possible_match)
            }
            // In case the resolved value is an edge case, we set the threshold to 1 for this
            // value
            let val_threshold = match self.edge_cases.contains(&possible_match.resolved_value) {
                false => threshold,
                true => 1.0
            };
            if check_threshold(possible_match.n_consumed_tokens, possible_match.raw_value_length - possible_match.n_consumed_tokens, val_threshold) {
                matches_heap.push(possible_match.clone());
            }
        }

        Ok(matches_heap)
    }


    #[inline(never)]
    fn update_previous_match(&self, possible_match: &mut PossibleMatch, res_val: &u32, token_idx: usize, value: u32, range_start: usize, range_end: usize, threshold: f32, ref mut matches_heap: &mut BinaryHeap<PossibleMatch>) {

        let (rank, otokens) = self.get_tokens_from_resolved_value(res_val).unwrap();
        {
            if possible_match.resolved_value == *res_val && token_idx == possible_match.last_token_in_input + 1 {
                // Grow the last Possible Match
                // Find the next token in the resolved value that matches the
                // input token
                for otoken_idx in possible_match.last_token_in_resolution + 1..otokens.len() {
                    let otok = otokens[otoken_idx];
                    if value == otok {
                        possible_match.range.end = range_end;
                        possible_match.n_consumed_tokens += 1;
                        possible_match.last_token_in_input = token_idx;
                        possible_match.last_token_in_resolution = otoken_idx;
                        possible_match.tokens_range.end += 1;
                        return
                    }
                }
            }
        }

        // the token belongs to a new resolved value, or the previous
        // PossibleMatch cannot be grown further. We start a new
        // PossibleMatch

        if possible_match.n_consumed_tokens > possible_match.raw_value_length {
            panic!("Consumed more tokens than are available: {:?}", possible_match)
        }
        // println!("CHECKING THRESHOLD FOR {:?}", possible_match);
        if check_threshold(possible_match.n_consumed_tokens, possible_match.raw_value_length - possible_match.n_consumed_tokens, threshold) {
            matches_heap.push(possible_match.clone());
        }
        // Then we initialize a new PossibleMatch with the same res val
        let last_token_in_resolution = otokens.iter().position(|e| *e == value).ok_or_else(|| format_err!("Tokens list should contain value but doesn't")).unwrap();
        *possible_match = PossibleMatch {
            resolved_value: *res_val,
            range: range_start..range_end,
            tokens_range: token_idx..(token_idx + 1),
            raw_value_length: otokens.len() as u32,
            last_token_in_input: token_idx,
            last_token_in_resolution,
            n_consumed_tokens: 1,
            rank: *rank
        };
    }

    /// when we insert a new possible match, we need to backtrack to check if the value did not
    /// start with some stop words
    #[inline(never)]
    fn insert_new_possible_match(&self, res_val: &u32, value: u32, range_start: usize, range_end: usize, token_idx: usize, threshold: f32, skipped_tokens: &HashMap<usize, (Range<usize>, u32)>) -> GazetteerParserResult<Option<PossibleMatch>> {
        let (rank, otokens) = self.get_tokens_from_resolved_value(res_val).unwrap();
        let last_token_in_resolution = otokens.iter().position(|e| *e == value).ok_or_else(|| format_err!("Tokens list should contain value but doesn't")).unwrap();

        let mut possible_match = PossibleMatch {
            resolved_value: *res_val,
            range: range_start..range_end,
            tokens_range: token_idx..(token_idx + 1),
            last_token_in_input: token_idx,
            last_token_in_resolution,
            n_consumed_tokens: 1,
            raw_value_length: otokens.len() as u32,
            rank: *rank
        };
        let mut n_skips = last_token_in_resolution as u32;

        // Bactrack to check if we left out from skipped words at the beginning
        'outer: for btok_idx in (0..token_idx).rev() {
            // println!("BACKTRACKING {:?} STEPS", btok_idx);
            if skipped_tokens.contains_key(&btok_idx) {
                let (skip_range, skip_tok) = skipped_tokens.get(&btok_idx).unwrap();
                match otokens.iter().position(|e| *e == *skip_tok) {
                    Some(idx) => {
                        if idx < possible_match.last_token_in_resolution {
                            possible_match.range.start = skip_range.start;
                            possible_match.tokens_range.start = btok_idx;
                            possible_match.n_consumed_tokens += 1;
                            n_skips -= 1;
                        } else {
                            break 'outer
                        }
                    }
                    None => break 'outer
                }
            } else {
                break 'outer
            }
        }

        // Conservative estimate of threshold condition for early stopping
        if possible_match.raw_value_length < n_skips {
            bail!("Skipped more tokens than are available: error")
        }
        if check_threshold(possible_match.raw_value_length - n_skips, n_skips, threshold) {
            Ok(Some(possible_match))
        } else {
            Ok(None)
        }

    }

    /// Parse the input string `input` and output a vec of `ParsedValue`. `decoding_threshold` is
    /// the minimum fraction of raw tokens to match for an entity to be parsed.
    #[inline(never)]
    pub fn run(
        &self,
        input: &str,
        decoding_threshold: f32,
    ) -> GazetteerParserResult<Vec<ParsedValue>> {

        let matches_heap = self.find_possible_matches(input, decoding_threshold)?;
        let parsing = self.parse_input(input, matches_heap)?;
        Ok(parsing)
    }

    #[inline(never)]
    fn parse_input(&self, input: &str, matches_heap: BinaryHeap<PossibleMatch>) -> GazetteerParserResult<Vec<ParsedValue>> {
        let mut taken_tokens: HashSet<usize> = HashSet::default();
        let n_total_tokens = whitespace_tokenizer(input).count();
        let mut parsing: BinaryHeap<ParsedValue> = BinaryHeap::default();

        'outer: for possible_match in matches_heap.into_sorted_vec().iter().rev() {

            let tokens_range_start = possible_match.tokens_range.start;
            let tokens_range_end = possible_match.tokens_range.end;
            for tok_idx in tokens_range_start..tokens_range_end {
                if taken_tokens.contains(&tok_idx) {
                    continue 'outer
                }
            }

            let range_start = possible_match.range.start;
            let range_end = possible_match.range.end;
            parsing.push(
                ParsedValue {
                    range: range_start..range_end,
                    raw_value: input.chars().skip(range_start).take(range_end - range_start).collect(),
                    resolved_value: self.resolved_symbol_table.find_index(&possible_match.resolved_value)?
                }
            );
            for idx in tokens_range_start..tokens_range_end {
                taken_tokens.insert(idx);
            }
            if taken_tokens.len() == n_total_tokens {
                break
            }
        }

        // Output ordered parsing
        Ok(parsing.into_sorted_vec())
    }

    fn get_parser_config() -> ParserConfig {
        ParserConfig {
            parser_filename: PARSER_FILE.to_string(),
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
        let config = Self::get_parser_config();
        let writer = fs::File::create(folder_name.as_ref().join(METADATA_FILENAME))?;
        serde_json::to_writer(writer, &config)?;

        let mut writer = fs::File::create(folder_name.as_ref().join(config.parser_filename))?;
        self.serialize(&mut Serializer::new(&mut writer))?;
        Ok(())
    }

    /// Load a resolver from a folder
    pub fn from_folder<P: AsRef<Path>>(folder_name: P) -> GazetteerParserResult<Parser> {
        let metadata_path = folder_name.as_ref().join(METADATA_FILENAME);
        let metadata_file = fs::File::open(&metadata_path)
            .with_context(|_| format!("Cannot open metadata file {:?}", metadata_path))?;
        let config: ParserConfig = serde_json::from_reader(metadata_file)?;

        let reader = fs::File::open(folder_name.as_ref().join(config.parser_filename))?;
        Ok(from_read(reader)?)
    }
}

#[cfg(test)]
mod tests {
    extern crate tempfile;
    extern crate mio_httpc;

    use std::time::{Instant};
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
        parser.set_stop_words(2, Some(vec!["hello"])).unwrap();

        parser.dump(tdir.as_ref().join("parser")).unwrap();
        let reloaded_parser = Parser::from_folder(tdir.as_ref().join("parser")).unwrap();
        tdir.close().unwrap();

        assert_eq!(parser, reloaded_parser)
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
            resolved_value: "The Stones Rolling".to_string(),
            raw_value: "the stones rolling".to_string(),
        });

        gazetteer.add(EntityValue {
            resolved_value: "The Stones".to_string(),
            raw_value: "the stones".to_string(),
        });
        let mut parser = Parser::from_gazetteer(&gazetteer).unwrap();
        parser.set_stop_words(2, Some(vec!["hello"])).unwrap();

        let mut gt_stop_words: HashSet<u32> = HashSet::default();
        gt_stop_words.insert(parser.tokens_symbol_table.find_single_symbol("the").unwrap().unwrap());
        gt_stop_words.insert(parser.tokens_symbol_table.find_single_symbol("stones").unwrap().unwrap());
        gt_stop_words.insert(parser.tokens_symbol_table.find_single_symbol("hello").unwrap().unwrap());
        assert!(parser.stop_words == gt_stop_words);
        let mut gt_edge_cases: HashSet<u32> = HashSet::default();
        gt_edge_cases.insert(parser.resolved_symbol_table.find_single_symbol("The Stones").unwrap().unwrap());
        assert!(parser.edge_cases == gt_edge_cases);

        // Value starting with a stop word
        let parsed = parser
            .run("je veux écouter les the rolling", 0.6)
            .unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    raw_value: "the rolling".to_string(),
                    resolved_value: "The Rolling Stones".to_string(),
                    range: 20..31,
                },
            ]
        );

        // Value starting with a stop word and ending with one
        let parsed = parser
            .run("je veux écouter les the rolling stones", 1.0)
            .unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    raw_value: "the rolling stones".to_string(),
                    resolved_value: "The Rolling Stones".to_string(),
                    range: 20..38,
                },
            ]
        );

        // Value starting with two stop words
        let parsed = parser
            .run("je veux écouter les the stones rolling", 1.0)
            .unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    raw_value: "the stones rolling".to_string(),
                    resolved_value: "The Stones Rolling".to_string(),
                    range: 20..38,
                },
            ]
        );

        // Edge case
        let parsed = parser
            .run("je veux écouter les the stones", 1.0)
            .unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    raw_value: "the stones".to_string(),
                    resolved_value: "The Stones".to_string(),
                    range: 20..30,
                },
            ]
        );


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
                resolved_value: "Quand est-ce ?".to_string(),
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

        // let parsed = parser
        //     .run("the the the", 0.5)
        //     .unwrap();
        // assert_eq!(parsed, vec![]);

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
    fn test_injection_ranking() {
        let mut gazetteer = Gazetteer::new();
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        let mut parser = Parser::from_gazetteer(&gazetteer).unwrap();

        let new_values = vec![
            EntityValue {
                resolved_value: "The Flying Stones".to_string(),
                raw_value: "the flying stones".to_string()
            }
        ];

        // Test with preprend set to false
        parser.inject_new_values(&new_values, false, false).unwrap();
        let parsed = parser
            .run("je veux écouter les flying stones", 0.6)
            .unwrap();

        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "flying stones".to_string(),
                resolved_value: "The Flying Stones".to_string(),
                range: 20..33,
            }]
        );
        let parsed = parser
            .run("je veux écouter the stones", 0.6)
            .unwrap();

        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "the stones".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
                range: 16..26,
            }]
        );

        // Test with preprend set to true
        parser.inject_new_values(&new_values, true, false).unwrap();
        let parsed = parser
            .run("je veux écouter les flying stones", 0.6)
            .unwrap();

        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "flying stones".to_string(),
                resolved_value: "The Flying Stones".to_string(),
                range: 20..33,
            }]
        );
        let parsed = parser
            .run("je veux écouter the stones", 0.6)
            .unwrap();

        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "the stones".to_string(),
                resolved_value: "The Flying Stones".to_string(),
                range: 16..26,
            }]
        );

    }


    #[test]
    fn test_injection_from_vanilla() {

        let mut gazetteer = Gazetteer::new();
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        let mut parser = Parser::from_gazetteer(&gazetteer).unwrap();

        let new_values_1 = vec![
            EntityValue {
                resolved_value: "The Flying Stones".to_string(),
                raw_value: "the flying stones".to_string()
            }
        ];

        parser.inject_new_values(&new_values_1, true, false).unwrap();

        // Test injection from vanilla
        let new_values_2 = vec![
            EntityValue {
                resolved_value: "Queens Of The Stone Age".to_string(),
                raw_value: "queens of the stone age".to_string()
            }
        ];

        let flying_idx = parser.tokens_symbol_table.find_single_symbol("flying").unwrap().unwrap();
        let stones_idx = parser.tokens_symbol_table.find_single_symbol("stones").unwrap().unwrap();
        let flying_stones_idx = parser.resolved_symbol_table.find_single_symbol("The Flying Stones").unwrap().unwrap();
        parser.inject_new_values(&new_values_2, true, true).unwrap();
        let parsed = parser
            .run("je veux écouter les flying stones", 0.6)
            .unwrap();
        assert_eq!(
            parsed,
            vec![]
        );

        let parsed = parser
            .run("je veux écouter queens the stone age", 0.6)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "queens the stone age".to_string(),
                resolved_value: "Queens Of The Stone Age".to_string(),
                range: 16..36,
            }]
        );

        assert_eq!(parser.resolved_symbol_table.find_symbol("The Flying Stones").unwrap(), None);
        assert_eq!(parser.tokens_symbol_table.find_symbol("flying").unwrap(), None);
        assert!(!parser.token_to_count.contains_key(&flying_idx));
        assert!(!parser.token_to_resolved_values.contains_key(&flying_idx));
        assert!(!parser.get_resolved_values_from_token(&stones_idx).unwrap().contains(&flying_stones_idx));
        assert!(!parser.resolved_value_to_tokens.contains_key(&flying_stones_idx));

    }

    #[test]
    fn test_injection_stop_words() {
        let mut gazetteer = Gazetteer::new();
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Stones".to_string(),
            raw_value: "the stones".to_string(),
        });
        let mut parser = Parser::from_gazetteer(&gazetteer).unwrap();
        parser.set_stop_words(2, Some(vec!["hello"])).unwrap();
        let mut parser_no_stop_words = Parser::from_gazetteer(&gazetteer).unwrap();

        let mut gt_stop_words: HashSet<u32> = HashSet::default();
        gt_stop_words.insert(parser.tokens_symbol_table.find_single_symbol("the").unwrap().unwrap());
        gt_stop_words.insert(parser.tokens_symbol_table.find_single_symbol("stones").unwrap().unwrap());
        gt_stop_words.insert(parser.tokens_symbol_table.find_single_symbol("hello").unwrap().unwrap());

        let mut gt_edge_cases: HashSet<u32> = HashSet::default();
        gt_edge_cases.insert(parser.resolved_symbol_table.find_single_symbol("The Stones").unwrap().unwrap());

        assert_eq!(parser.stop_words, gt_stop_words);
        assert_eq!(parser.edge_cases, gt_edge_cases);

        let new_values = vec![
            EntityValue {
                resolved_value: "Rolling".to_string(),
                raw_value: "rolling".to_string()
            },
            EntityValue {
                resolved_value: "Rolling Two".to_string(),
                raw_value: "rolling two".to_string()
            }
        ];

        parser.inject_new_values(&new_values, true, false).unwrap();

        gt_stop_words.remove(&parser.tokens_symbol_table.find_single_symbol("stones").unwrap().unwrap());
        gt_stop_words.insert(parser.tokens_symbol_table.find_single_symbol("rolling").unwrap().unwrap());

        gt_edge_cases.remove(&parser.resolved_symbol_table.find_single_symbol("The Stones").unwrap().unwrap());
        gt_edge_cases.insert(parser.resolved_symbol_table.find_single_symbol("Rolling").unwrap().unwrap());

        assert_eq!(parser.stop_words, gt_stop_words);
        assert_eq!(parser.edge_cases, gt_edge_cases);

        // No stop words case
        assert!(parser_no_stop_words.stop_words.is_empty());
        assert!(parser_no_stop_words.edge_cases.is_empty());

    }

    #[test]
    #[ignore]
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
    #[ignore]
    fn real_world_gazetteer_parser() {

        let (_, body) = CallBuilder::get().max_response(20000000).timeout_ms(60000).url("https://s3.amazonaws.com/snips/nlu-lm/test/gazetteer-entity-parser/artist_gazetteer_formatted.json").unwrap().exec().unwrap();
        let data: Vec<EntityValue> = serde_json::from_reader(&*body).unwrap();
        let gaz = Gazetteer{ data };
        // DEBUG
        // let gaz = Gazetteer::from_json("local_testing/artist_gazetteer_formatted.json", None).unwrap();

        let mut parser = Parser::from_gazetteer(&gaz).unwrap();
        let n_stop_words = 50;
        parser.set_stop_words(n_stop_words, None).unwrap();
        // parser.dump("test_artist_parser").unwrap();
        println!("N STOP WORDS {:?}", n_stop_words);
        println!("ARTIST GAZETTEER, STOP WORDS {:?}", parser.get_stop_words());
        println!("ARTIST GAZETTEER, EDGE CASES {:?}", parser.get_edge_cases());

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

        let (_, body) = CallBuilder::get().max_response(20000000).timeout_ms(60000).url("https://s3.amazonaws.com/snips/nlu-lm/test/gazetteer-entity-parser/album_gazetteer_formatted.json").unwrap().exec().unwrap();
        let data: Vec<EntityValue> = serde_json::from_reader(&*body).unwrap();
        let gaz = Gazetteer{ data };
        // DEBUG
        // let gaz = Gazetteer::from_json("local_testing/album_gazetteer_formatted.json", None).unwrap();

        let mut parser = Parser::from_gazetteer(&gaz).unwrap();
        // parser.dump("test_album_parser").unwrap();
        let n_stop_words = 50;
        parser.set_stop_words(n_stop_words, None).unwrap();
        println!("N STOP WORDS {:?}", n_stop_words);
        println!("ALBUM GAZETTEER, STOP WORDS {:?}", parser.get_stop_words());
        println!("ALBUM GAZETTEER, EDGE CASES {:?}", parser.get_edge_cases());
        // println!("FRACTION {:?}", fraction);
        // println!("ALBUM GAZETTEER, STOP WORDS {:?}", parser.stop_words.len());
        // println!("ALBUM GAZETTEER, EDGE CASES {:?}", parser.edge_cases.len());
        // println!("STOP WORDS {:?}", parser.get_stop_words().unwrap());
        // println!("EDGE CASES WORDS {:?}", parser.get_edge_cases().unwrap());
        //

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
        let parsed = parser
            .run("je veux écouter dark side of the moon", 0.5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![
            ParsedValue {
                raw_value: "je veux".to_string(),
                resolved_value: "Je veux du bonheur".to_string(),
                range: 0..7,
            },
            ParsedValue {
                raw_value: "dark side of the moon".to_string(),
                resolved_value: "Dark Side of the Moon".to_string(),
                range: 16..37,
            }]
        );
    }


    #[test]
    #[ignore]
    fn test_real_word_injection() {
        // Real-world artist gazetteer
        let (_, body) = CallBuilder::get().max_response(20000000).timeout_ms(60000).url("https://s3.amazonaws.com/snips/nlu-lm/test/gazetteer-entity-parser/artist_gazetteer_formatted.json").unwrap().exec().unwrap();
        let data: Vec<EntityValue> = serde_json::from_reader(&*body).unwrap();
        let album_gaz = Gazetteer{ data };
        // DEBUG
        // let album_gaz = Gazetteer::from_json("local_testing/artist_gazetteer_formatted.json", None).unwrap();

        let mut parser_for_test = Parser::from_gazetteer(&album_gaz).unwrap();
        parser_for_test.set_stop_words(50, None).unwrap();
        let mut parser = Parser::from_gazetteer(&album_gaz).unwrap();
        parser.set_stop_words(50, None).unwrap();

        // Get 10k values from the album gazetter to inject in the album parser
        let (_, body) = CallBuilder::get().max_response(20000000).timeout_ms(60000).url("https://s3.amazonaws.com/snips/nlu-lm/test/gazetteer-entity-parser/album_gazetteer_formatted.json").unwrap().exec().unwrap();
        let mut new_values: Vec<EntityValue> = serde_json::from_reader(&*body).unwrap();
        new_values.truncate(10000);
        // DEBUG
        // let file = fs::File::open("local_testing/album_gazetteer_formatted.json")
        //     .with_context(|_| format!("Cannot open gazetter file {:?}", "local_testing/artist_gazetteer_formatted.json")).unwrap();
        // let mut new_values: Vec<EntityValue> = serde_json::from_reader(file).unwrap();
        // new_values.truncate(10000);

        // Test injection
        let parsed = parser_for_test
            .run("je veux écouter hans knappertsbusch conducts", 0.7)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "hans knappertsbusch".to_string(),
                resolved_value: "Hans Knappertsbusch".to_string(),
                range: 16..35,
            }]
        );
        parser_for_test.inject_new_values(&new_values, true, false).unwrap();
        let parsed = parser_for_test
            .run("je veux écouter hans knappertsbusch conducts", 0.7)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "hans knappertsbusch conducts".to_string(),
                resolved_value: "Hans Knappertsbusch conducts".to_string(),
                range: 16..44,
            }]
        );

        let now = Instant::now();
        parser.inject_new_values(&new_values, true, false).unwrap();
        let total_time = now.elapsed().as_secs();
        // DEBUG
        // println!("TOTAL TIME {:?}", now.elapsed());
        assert!(total_time < 10);
    }
}
