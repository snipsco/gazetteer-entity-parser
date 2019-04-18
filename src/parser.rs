use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, BinaryHeap};
use std::fs;
use std::ops::Range;
use std::path::Path;

use failure::{format_err, ResultExt};
use fnv::{FnvHashMap as HashMap, FnvHashSet as HashSet};
use rmp_serde::{from_read, Serializer};
use serde::Serialize;
use serde_derive::*;
use serde_json;

use crate::constants::*;
use crate::data::EntityValue;
use crate::errors::*;
use crate::symbol_table::{ResolvedSymbolTable, TokenSymbolTable};
use crate::utils::{check_threshold, whitespace_tokenizer};

/// Struct representing the parser. The Parser will match the longest possible contiguous
/// substrings of a query that match partial entity values. The order in which the values are
/// added to the parser matters: In case of ambiguity between two parsings, the Parser will output
/// the value that was added first (see Gazetteer).
#[derive(PartialEq, Debug, Serialize, Deserialize, Default)]
pub struct Parser {
    // Symbol table for the raw tokens
    tokens_symbol_table: TokenSymbolTable,
    // Symbol table for the resolved values
    // The latter differs from the first one in that it can contain the same resolved value
    // multiple times (to allow for multiple raw values corresponding to the same resolved value)
    resolved_symbol_table: ResolvedSymbolTable,
    // maps token to set of resolved values containing token
    token_to_resolved_values: HashMap<u32, BTreeSet<u32>>,
    // maps resolved value to a tuple (rank, tokens)
    resolved_value_to_tokens: HashMap<u32, (u32, Vec<u32>)>,
    // number of stop words to extract from the entity data
    n_stop_words: usize,
    // external list of stop words
    additional_stop_words: Vec<String>,
    // set of all stop words
    stop_words: HashSet<u32>,
    // values composed only of stop words
    edge_cases: HashSet<u32>,
    // Keep track of values injected thus far
    injected_values: HashSet<String>,
    // Parsing threshold giving minimal fraction of tokens necessary to parse a value
    threshold: f32,
}

#[derive(Serialize, Deserialize)]
struct ParserConfig {
    version: String,
    parser_filename: String,
    threshold: f32,
    stop_words: HashSet<String>,
    edge_cases: HashSet<String>,
}

/// Struct holding a possible match that can be grown by iterating over the input tokens
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct PossibleMatch {
    resolved_value: u32,
    range: Range<usize>,
    tokens_range: Range<usize>,
    raw_value_length: u32,
    n_consumed_tokens: u32,
    last_token_in_input: usize,
    first_token_in_resolution: usize,
    last_token_in_resolution: usize,
    rank: u32,
}

impl PossibleMatch {
    fn check_threshold(&self, threshold: f32) -> bool {
        check_threshold(
            self.n_consumed_tokens,
            self.raw_value_length - self.n_consumed_tokens,
            threshold,
        )
    }
}

impl Ord for PossibleMatch {
    #[allow(clippy::if_same_then_else)]
    fn cmp(&self, other: &PossibleMatch) -> Ordering {
        if self.n_consumed_tokens < other.n_consumed_tokens {
            Ordering::Less
        } else if self.n_consumed_tokens > other.n_consumed_tokens {
            Ordering::Greater
        } else if self.raw_value_length < other.raw_value_length {
            Ordering::Greater
        } else if self.raw_value_length > other.raw_value_length {
            Ordering::Less
        } else {
            other.rank.cmp(&self.rank)
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
#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct ParsedValue {
    pub resolved_value: String,
    pub resolved_value_ids: Vec<String>,
    // character-level
    pub range: Range<usize>,
    pub raw_value: String,
    pub matched_value: String,
}

impl Ord for ParsedValue {
    fn cmp(&self, other: &ParsedValue) -> Ordering {
        match self.partial_cmp(other) {
            Some(value) => value,
            // The following should not happen: we need to make sure that we compare only
            // comparable ParsedValues wherever we use a heap of ParsedValue's (see e.g. the
            // `parse_input` method)
            None => panic!("Parsed values are not comparable: {:?}, {:?}", self, other),
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
    /// Add a single entity value, along with its rank, to the parser
    /// The ranks of the other entity values will not be changed
    pub fn add_value(&mut self, entity_value: EntityValue, rank: u32) {
        // We force add the new resolved value: even if it is already present in the symbol table
        // we duplicate it to allow several raw values to map to it
        let res_value_idx = self
            .resolved_symbol_table
            .add_symbol(entity_value.resolved_value, entity_value.resolved_value_id);

        for (_, token) in whitespace_tokenizer(&entity_value.raw_value) {
            let token_idx = self.tokens_symbol_table.add_symbol(token);

            // Update token_to_resolved_values map
            self.token_to_resolved_values
                .entry(token_idx)
                .and_modify(|e| {
                    e.insert(res_value_idx);
                })
                .or_insert_with(|| vec![res_value_idx].into_iter().collect());

            // Update resolved_value_to_tokens map
            self.resolved_value_to_tokens
                .entry(res_value_idx)
                .and_modify(|(_, v)| v.push(token_idx))
                .or_insert((rank, vec![token_idx]));
        }
    }

    /// Prepend a list of entity values to the parser and update the ranks accordingly
    pub fn prepend_values(&mut self, entity_values: Vec<EntityValue>) {
        // update rank of previous values
        self.shift_ranks(entity_values.len() as u32);

        for (rank, entity_value) in entity_values.into_iter().enumerate() {
            self.add_value(entity_value, rank as u32);
        }

        // Update the stop words and edge cases
        let n_stop_words = self.n_stop_words;
        let additional_stop_words = self.additional_stop_words.clone();
        self.set_stop_words(n_stop_words, Some(additional_stop_words));
    }

    /// Set the threshold (minimum fraction of tokens to match for an entity to be parsed).
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    /// Update an internal set of stop words and corresponding edge cases.
    /// The set of stop words is made of the `n_stop_words` most frequent raw tokens in the
    /// gazetteer used to generate the parser. An optional `additional_stop_words` vector of
    /// strings can be added to the stop words. The edge cases are defined to the be the resolved
    /// values whose raw value is composed only of stop words. There are examined seperately
    /// during parsing, and will match if and only if they are present verbatim in the input
    /// string.
    pub fn set_stop_words(
        &mut self,
        n_stop_words: usize,
        additional_stop_words: Option<Vec<String>>,
    ) {
        // Update the set of stop words with the most frequent words in the gazetteer
        let mut tokens_with_counts = self
            .token_to_resolved_values
            .iter()
            .map(|(idx, res_values)| (*idx, res_values.len()))
            .collect::<Vec<_>>();

        tokens_with_counts.sort_by_key(|&(_, count)| -(count as i32));
        self.n_stop_words = n_stop_words;
        self.stop_words = tokens_with_counts
            .into_iter()
            .take(n_stop_words)
            .map(|(idx, _)| idx)
            .collect();

        // add the words from the `additional_stop_words` vec (and potentially add them to
        // the symbol table and to tokens_to_resolved_value)
        if let Some(additional_stop_words_vec) = additional_stop_words {
            self.additional_stop_words = additional_stop_words_vec.clone();
            for tok_s in additional_stop_words_vec.into_iter() {
                let tok_idx = self.tokens_symbol_table.add_symbol(tok_s);
                self.stop_words.insert(tok_idx);
                self.token_to_resolved_values
                    .entry(tok_idx)
                    .or_insert_with(BTreeSet::new);
            }
        }

        // Update the set of edge_cases. i.e. resolved value that only contain stop words
        self.edge_cases = self
            .resolved_value_to_tokens
            .iter()
            .filter(|(_, (_, tokens))| tokens.iter().all(|token| self.stop_words.contains(token)))
            .map(|(res_val, _)| *res_val)
            .collect();
    }

    /// Get the set of stop words
    pub fn get_stop_words(&self) -> HashSet<String> {
        self.stop_words
            .iter()
            .flat_map(|idx| {
                self.tokens_symbol_table
                    .find_index(*idx)
                    .map(|sym| sym.to_string())
            })
            .collect()
    }

    /// Get the set of edge cases, containing only stop words
    pub fn get_edge_cases(&self) -> HashSet<String> {
        self.edge_cases
            .iter()
            .flat_map(|idx| {
                self.resolved_symbol_table
                    .find_index(*idx)
                    .map(|resolved_symbol| resolved_symbol.to_string())
            })
            .collect()
    }

    /// Parse the input string `input` and output a vec of `ParsedValue`.
    pub fn run(&self, input: &str) -> Result<Vec<ParsedValue>> {
        let matches_heap = self
            .find_possible_matches(input, self.threshold)
            .with_context(|_| format_err!("Error when finding possible matches"))?;
        Ok(self
            .parse_input(input, matches_heap)
            .with_context(|_| format_err!("Error when filtering possible matches"))?)
    }

    /// Add new values to an already trained Parser. This function is used for entity injection.
    /// It takes as arguments a vector of EntityValue's to inject, and a boolean indicating
    /// whether the new values should be prepended to the already existing values (`prepend=true`)
    /// or appended (`prepend=false`). Setting `from_vanilla` to true allows to remove all
    /// previously injected values before adding the new ones.
    pub fn inject_new_values(
        &mut self,
        new_values: Vec<EntityValue>,
        prepend: bool,
        from_vanilla: bool,
    ) -> Result<()> {
        if from_vanilla {
            // Remove the resolved values from the resolved_symbol_table
            // Remove the resolved values from the resolved_value_to_tokens map
            // remove the corresponding resolved value from the token_to_resolved_value map
            // if after removing, a token is left absent from all resolved entities, then remove
            // it from the tokens_to_resolved_value maps and from the tokens_symbol_table
            let mut tokens_marked_for_removal: HashSet<u32> = HashSet::default();
            for val in &self.injected_values {
                for res_val in self.resolved_symbol_table.remove_symbol(&val) {
                    let (_, tokens) = self
                        .get_tokens_from_resolved_value(res_val)
                        .with_context(|_| {
                            format_err!("Error when retrieving tokens of resolved value '{}'", val)
                        })?
                        .clone();
                    self.resolved_value_to_tokens.remove(&res_val);
                    for tok in tokens {
                        let remaining_values = self
                            .token_to_resolved_values
                            .get_mut(&tok)
                            .map(|v| {
                                v.remove(&res_val);
                                v
                            })
                            .ok_or_else(|| {
                                format_err!(
                                    "Cannot find token index {} in `token_to_resolved_values`",
                                    tok
                                )
                            })?;

                        // Check the remaining resolved values containing the token
                        if remaining_values.is_empty() {
                            tokens_marked_for_removal.insert(tok);
                        }
                    }
                }
            }
            for tok_idx in tokens_marked_for_removal {
                self.tokens_symbol_table.remove_index(tok_idx);
                self.token_to_resolved_values.remove(&tok_idx);
            }
        }

        if prepend {
            // update rank of previous values
            self.shift_ranks(new_values.len() as u32);
        }

        let new_start_rank = if prepend {
            // we inject new values from rank 0 to n_new_values - 1
            0
        } else {
            // we inject new values from the current last rank onwards
            self.resolved_value_to_tokens.len()
        } as u32;

        for (rank, entity_value) in new_values.into_iter().enumerate() {
            self.add_value(entity_value.clone(), new_start_rank + rank as u32);
            self.injected_values.insert(entity_value.resolved_value);
        }

        // Update the stop words and edge cases
        let n_stop_words = self.n_stop_words;
        let additional_stop_words = self.additional_stop_words.clone();
        self.set_stop_words(n_stop_words, Some(additional_stop_words));
        Ok(())
    }
}

impl Parser {
    /// Dump the parser to a folder
    pub fn dump<P: AsRef<Path>>(&self, folder_name: P) -> Result<()> {
        fs::create_dir(folder_name.as_ref())
            .with_context(|_| format_err!("Error when creating persisting directory"))?;

        let config = self.get_parser_config();

        let writer = fs::File::create(folder_name.as_ref().join(METADATA_FILENAME))
            .with_context(|_| format_err!("Error when creating metadata file"))?;

        serde_json::to_writer(writer, &config)
            .with_context(|_| format_err!("Error when serializing the parser's metadata"))?;

        let parser_path = folder_name.as_ref().join(config.parser_filename);
        let mut writer = fs::File::create(&parser_path)
            .with_context(|_| format_err!("Error when creating the parser file"))?;

        self.serialize(&mut Serializer::new(&mut writer))
            .with_context(|_| format_err!("Error when serializing the parser"))?;
        Ok(())
    }

    /// Load a parser from a folder
    pub fn from_folder<P: AsRef<Path>>(folder_name: P) -> Result<Parser> {
        let metadata_path = folder_name.as_ref().join(METADATA_FILENAME);
        let metadata_file = fs::File::open(&metadata_path)
            .with_context(|_| format_err!("Error when opening the metadata file"))?;

        let config: ParserConfig = serde_json::from_reader(metadata_file)
            .with_context(|_| format_err!("Error when deserializing the metadata"))?;

        let parser_path = folder_name.as_ref().join(config.parser_filename);
        let reader = fs::File::open(&parser_path)
            .with_context(|_| format_err!("Error when opening the parser file"))?;

        Ok(from_read(reader)
            .with_context(|_| format_err!("Error when deserializing the parser"))?)
    }
}

impl Parser {
    /// get resolved value
    fn get_tokens_from_resolved_value(&self, resolved_value: u32) -> Result<&(u32, Vec<u32>)> {
        self.resolved_value_to_tokens
            .get(&resolved_value)
            .ok_or_else(|| {
                format_err!(
                    "Cannot find resolved value index {} in `resolved_value_to_tokens`",
                    resolved_value
                )
            })
    }

    /// get resolved values from token
    fn get_resolved_values_from_token(&self, token: u32) -> Result<&BTreeSet<u32>> {
        self.token_to_resolved_values.get(&token).ok_or_else(|| {
            format_err!(
                "Cannot find token index {} in `token_to_resolved_values`",
                token
            )
        })
    }

    /// Shift the ranks of all the resolved values
    fn shift_ranks(&mut self, shift: u32) {
        for (rank, _) in self.resolved_value_to_tokens.values_mut() {
            *rank += shift
        }
    }

    /// get the underlying matched value associated to a `PossibleMatch`
    fn get_matched_value(&self, possible_match: &PossibleMatch) -> Result<String> {
        Ok(self
            .resolved_value_to_tokens
            .get(&possible_match.resolved_value)
            .ok_or_else(|| {
                format_err!(
                    "Missing key for resolved value {}",
                    possible_match.resolved_value
                )
            })?
            .1
            .iter()
            .flat_map(|token_idx| self.tokens_symbol_table.find_index(*token_idx))
            .map(|token_string| token_string)
            .collect::<Vec<_>>()
            .join(" "))
    }

    /// Find all possible matches in a string.
    /// Returns a hashmap, indexed by resolved values. The corresponding value is a vec of tuples
    /// each tuple is a possible match for the resolved value, and is made of
    /// (range of match, number of skips, index of last matched token in the resolved value)
    fn find_possible_matches(
        &self,
        input: &str,
        threshold: f32,
    ) -> Result<BinaryHeap<PossibleMatch>> {
        let mut possible_matches: HashMap<u32, PossibleMatch> =
            HashMap::with_capacity_and_hasher(1000, Default::default());
        let mut matches_heap: BinaryHeap<PossibleMatch> = BinaryHeap::default();
        let mut skipped_tokens: HashMap<usize, (Range<usize>, u32)> = HashMap::default();
        for (token_idx, (range, token)) in whitespace_tokenizer(input).enumerate() {
            if let Some(value) = self.tokens_symbol_table.find_symbol(&token) {
                let res_vals_from_token = self.get_resolved_values_from_token(*value)?;
                if res_vals_from_token.is_empty() {
                    continue;
                }
                if !self.stop_words.contains(&value) {
                    for res_val in res_vals_from_token {
                        self.update_or_insert_possible_match(
                            *value,
                            *res_val,
                            token_idx,
                            range.clone(),
                            &mut possible_matches,
                            &mut matches_heap,
                            &mut skipped_tokens,
                            threshold,
                        )?;
                    }
                } else {
                    skipped_tokens.insert(token_idx, (range.clone(), *value));
                    // Iterate over all edge cases and try to add or update corresponding
                    // PossibleMatch's. Using a threshold of 1.
                    for res_val in self.edge_cases.iter() {
                        if res_vals_from_token.contains(res_val) {
                            self.update_or_insert_possible_match(
                                *value,
                                *res_val,
                                token_idx,
                                range.clone(),
                                &mut possible_matches,
                                &mut matches_heap,
                                &mut skipped_tokens,
                                1.0,
                            )?;
                        }
                    }

                    // Iterate over current possible matches containing the stop word and
                    // try to grow them (but do not initiate a new possible match)
                    // Threshold depends on whether the res_val is an edge case or not
                    for (res_val, mut possible_match) in &mut possible_matches {
                        if !res_vals_from_token.contains(res_val)
                            || self.edge_cases.contains(res_val)
                        {
                            continue;
                        }
                        self.update_previous_match(
                            &mut possible_match,
                            token_idx,
                            *value,
                            range.clone(),
                            threshold,
                            &mut matches_heap,
                        )?;
                    }
                }
            }
        }

        // Add to the heap the possible matches that remain
        Ok(possible_matches
            .values()
            .filter(|possible_match| {
                if self.edge_cases.contains(&possible_match.resolved_value) {
                    possible_match.check_threshold(1.0)
                } else {
                    possible_match.check_threshold(threshold)
                }
            })
            .fold(matches_heap, |mut acc, possible_match| {
                acc.push(possible_match.clone());
                acc
            }))
    }

    #[allow(clippy::too_many_arguments)]
    fn update_or_insert_possible_match(
        &self,
        value: u32,
        res_val: u32,
        token_idx: usize,
        range: Range<usize>,
        possible_matches: &mut HashMap<u32, PossibleMatch>,
        mut matches_heap: &mut BinaryHeap<PossibleMatch>,
        skipped_tokens: &mut HashMap<usize, (Range<usize>, u32)>,
        threshold: f32,
    ) -> Result<()> {
        match possible_matches.entry(res_val) {
            Entry::Occupied(mut entry) => {
                self.update_previous_match(
                    entry.get_mut(),
                    token_idx,
                    value,
                    range,
                    threshold,
                    &mut matches_heap,
                )?;
            }
            Entry::Vacant(entry) => {
                if let Some(new_possible_match) = self.insert_new_possible_match(
                    res_val,
                    value,
                    range,
                    token_idx,
                    threshold,
                    &skipped_tokens,
                )? {
                    entry.insert(new_possible_match);
                }
            }
        }
        Ok(())
    }

    fn update_previous_match(
        &self,
        possible_match: &mut PossibleMatch,
        token_idx: usize,
        value: u32,
        range: Range<usize>,
        threshold: f32,
        matches_heap: &mut BinaryHeap<PossibleMatch>,
    ) -> Result<()> {
        let (rank, otokens) = self.get_tokens_from_resolved_value(possible_match.resolved_value)?;

        if token_idx == possible_match.last_token_in_input + 1 {
            // Grow the last Possible Match
            // Find the next token in the resolved value that matches the
            // input token
            for (otoken_idx, otok) in otokens
                .iter()
                .enumerate()
                .skip(possible_match.last_token_in_resolution + 1)
            {
                if value == *otok {
                    possible_match.range.end = range.end;
                    possible_match.n_consumed_tokens += 1;
                    possible_match.last_token_in_input = token_idx;
                    possible_match.last_token_in_resolution = otoken_idx;
                    possible_match.tokens_range.end += 1;
                    return Ok(());
                }
            }
        }

        // the token belongs to a new resolved value, or the previous
        // PossibleMatch cannot be grown further.
        // We start a new PossibleMatch.

        if possible_match.check_threshold(threshold) {
            matches_heap.push(possible_match.clone());
        }
        // Then we initialize a new PossibleMatch with the same res val
        let last_token_in_resolution =
            otokens.iter().position(|e| *e == value).ok_or_else(|| {
                format_err!("Missing token {} from list {:?}", value, otokens.clone())
            })?;

        *possible_match = PossibleMatch {
            resolved_value: possible_match.resolved_value,
            range,
            tokens_range: token_idx..(token_idx + 1),
            raw_value_length: otokens.len() as u32,
            last_token_in_input: token_idx,
            first_token_in_resolution: last_token_in_resolution,
            last_token_in_resolution,
            n_consumed_tokens: 1,
            rank: *rank,
        };
        Ok(())
    }

    /// when we insert a new possible match, we need to backtrack to check if the value did not
    /// start with some stop words
    fn insert_new_possible_match(
        &self,
        res_val: u32,
        value: u32,
        range: Range<usize>,
        token_idx: usize,
        threshold: f32,
        skipped_tokens: &HashMap<usize, (Range<usize>, u32)>,
    ) -> Result<Option<PossibleMatch>> {
        let (rank, otokens) = self.get_tokens_from_resolved_value(res_val)?;
        let last_token_in_resolution =
            otokens.iter().position(|e| *e == value).ok_or_else(|| {
                format_err!("Missing token {} from list {:?}", value, otokens.clone())
            })?;

        let mut possible_match = PossibleMatch {
            resolved_value: res_val,
            range,
            tokens_range: token_idx..(token_idx + 1),
            last_token_in_input: token_idx,
            first_token_in_resolution: last_token_in_resolution,
            last_token_in_resolution,
            n_consumed_tokens: 1,
            raw_value_length: otokens.len() as u32,
            rank: *rank,
        };
        let mut n_skips = last_token_in_resolution as u32;
        // Backtrack to check if we left out from skipped words at the beginning
        for btok_idx in (0..token_idx).rev() {
            if let Some((skip_range, skip_tok)) = skipped_tokens.get(&btok_idx) {
                match otokens.iter().position(|e| *e == *skip_tok) {
                    Some(idx) => {
                        if idx < possible_match.first_token_in_resolution {
                            possible_match.range.start = skip_range.start;
                            possible_match.tokens_range.start = btok_idx;
                            possible_match.n_consumed_tokens += 1;
                            possible_match.first_token_in_resolution -= 1;
                            n_skips -= 1;
                        } else {
                            break;
                        }
                    }
                    None => break,
                }
            } else {
                break;
            }
        }

        // Conservative estimate of threshold condition for early stopping
        // if we have already skipped more tokens than is permitted by the threshold condition,
        // there is no point in continuing
        if check_threshold(
            possible_match.raw_value_length - n_skips,
            n_skips,
            threshold,
        ) {
            Ok(Some(possible_match))
        } else {
            Ok(None)
        }
    }

    fn reduce_possible_match(
        input: &str,
        possible_match: PossibleMatch,
        overlapping_tokens: HashSet<usize>,
    ) -> Option<PossibleMatch> {
        let reduced_tokens = whitespace_tokenizer(input)
            .enumerate()
            .filter(|(token_idx, _)| {
                *token_idx >= possible_match.tokens_range.start
                    && *token_idx < possible_match.tokens_range.end
                    && !overlapping_tokens.contains(token_idx)
            })
            .collect::<Vec<_>>();

        match (reduced_tokens.first(), reduced_tokens.last()) {
            (
                Some((first_token_idx, (first_token_range, _))),
                Some((last_token_idx, (last_token_range, _))),
            ) => Some(PossibleMatch {
                resolved_value: possible_match.resolved_value,
                range: (first_token_range.start)..(last_token_range.end),
                tokens_range: *first_token_idx..(last_token_idx + 1),
                raw_value_length: possible_match.raw_value_length,
                n_consumed_tokens: *last_token_idx as u32 - *first_token_idx as u32 + 1,
                last_token_in_input: 0, // we are not going to need this one
                last_token_in_resolution: 0, // we are not going to need this one
                first_token_in_resolution: 0, // we are not going to need this one
                rank: possible_match.rank,
            }),
            _ => None,
        }
    }

    fn parse_input(
        &self,
        input: &str,
        mut matches_heap: BinaryHeap<PossibleMatch>,
    ) -> Result<Vec<ParsedValue>> {
        let mut taken_tokens: HashSet<usize> = HashSet::default();
        let n_total_tokens = whitespace_tokenizer(input).count();
        let mut parsing: BinaryHeap<ParsedValue> = BinaryHeap::default();

        while !matches_heap.is_empty() && taken_tokens.len() < n_total_tokens {
            let possible_match = matches_heap.pop().unwrap();

            let tokens_range_start = possible_match.tokens_range.start;
            let tokens_range_end = possible_match.tokens_range.end;

            let overlapping_tokens: HashSet<_> = taken_tokens
                .iter()
                .filter(|idx| {
                    possible_match.tokens_range.start <= **idx
                        && possible_match.tokens_range.end > **idx
                })
                .cloned()
                .collect();

            if !overlapping_tokens.is_empty() {
                let opt_reduced_possible_match =
                    Self::reduce_possible_match(input, possible_match, overlapping_tokens);
                if let Some(reduced_possible_match) = opt_reduced_possible_match {
                    let threshold = if self
                        .edge_cases
                        .contains(&reduced_possible_match.resolved_value)
                    {
                        1.0
                    } else {
                        self.threshold
                    };
                    if reduced_possible_match.check_threshold(threshold) {
                        matches_heap.push(reduced_possible_match.clone());
                    }
                }
                continue;
            }
            let resolved_symbol = self
                .resolved_symbol_table
                .find_index(possible_match.resolved_value)
                .ok_or_else(|| {
                    format_err!(
                        "Missing key for resolved value {}",
                        possible_match.resolved_value
                    )
                })?;
            let associated_ids = self
                .resolved_symbol_table
                .get_associated_ids(resolved_symbol)
                .into_iter()
                .map(|id| id.to_string())
                .collect();

            parsing.push(ParsedValue {
                range: possible_match.range.clone(),
                raw_value: input
                    .chars()
                    .skip(possible_match.range.start)
                    .take(possible_match.range.len())
                    .collect(),
                resolved_value: resolved_symbol.to_string(),
                resolved_value_ids: associated_ids,
                matched_value: self.get_matched_value(&possible_match)?,
            });
            for idx in tokens_range_start..tokens_range_end {
                taken_tokens.insert(idx);
            }
        }

        // Output ordered parsing
        Ok(parsing.into_sorted_vec())
    }

    fn get_parser_config(&self) -> ParserConfig {
        ParserConfig {
            parser_filename: PARSER_FILE.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            threshold: self.threshold,
            stop_words: self.get_stop_words(),
            edge_cases: self.get_edge_cases(),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate mio_httpc;
    extern crate tempfile;

    use std::time::Instant;

    use failure::ResultExt;

    use crate::data::EntityValue;
    use crate::data::Gazetteer;
    use crate::parser_builder::ParserBuilder;

    #[allow(unused_imports)]
    use super::*;

    use self::mio_httpc::CallBuilder;
    use self::tempfile::tempdir;

    #[test]
    fn test_serialization_deserialization() {
        let tdir = tempdir().unwrap();
        let mut gazetteer = Gazetteer::default();
        gazetteer.add(EntityValue {
            resolved_value: "The Flying Stones".to_string(),
            raw_value: "the flying stones".to_string(),
            resolved_value_id: None,
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
            resolved_value_id: None,
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the stones".to_string(),
            resolved_value_id: None,
        });
        let parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .n_stop_words(2)
            .additional_stop_words(vec!["hello".to_string()])
            .build()
            .unwrap();
        parser.dump(tdir.as_ref().join("parser")).unwrap();
        let reloaded_parser = Parser::from_folder(tdir.as_ref().join("parser")).unwrap();

        assert_eq!(parser, reloaded_parser);

        // check content of metadata
        let metadata_path = tdir.as_ref().join("parser").join(METADATA_FILENAME);
        let metadata_file = fs::File::open(&metadata_path)
            .with_context(|_| format!("Cannot open metadata file {:?}", metadata_path))
            .unwrap();
        let config: ParserConfig = serde_json::from_reader(metadata_file).unwrap();

        assert_eq!(config.threshold, 0.5);
        let expected_stop_words =
            vec!["the".to_string(), "stones".to_string(), "hello".to_string()]
                .into_iter()
                .collect();

        assert_eq!(config.stop_words, expected_stop_words);
        let expected_edge_cases = vec!["The Rolling Stones".to_string()].into_iter().collect();
        assert_eq!(config.edge_cases, expected_edge_cases);

        tdir.close().unwrap();
    }

    #[test]
    fn test_stop_words_and_edge_cases() {
        let mut gazetteer = Gazetteer::default();
        gazetteer.add(EntityValue {
            resolved_value: "The Flying Stones".to_string(),
            raw_value: "the flying stones".to_string(),
            resolved_value_id: None,
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
            resolved_value_id: None,
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Stones Rolling".to_string(),
            raw_value: "the stones rolling".to_string(),
            resolved_value_id: None,
        });

        gazetteer.add(EntityValue {
            resolved_value: "The Stones".to_string(),
            raw_value: "the stones".to_string(),
            resolved_value_id: None,
        });

        let mut parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .n_stop_words(2)
            .additional_stop_words(vec!["hello".to_string()])
            .build()
            .unwrap();

        let expected_stop_words: HashSet<_> = vec!["the", "stones", "hello"]
            .into_iter()
            .map(|sym| *parser.tokens_symbol_table.find_symbol(sym).unwrap())
            .collect();
        assert_eq!(expected_stop_words, parser.stop_words);
        let mut expected_edge_cases: HashSet<u32> = HashSet::default();
        expected_edge_cases.insert(parser.resolved_symbol_table.find_symbol("The Stones")[0]);
        assert_eq!(expected_edge_cases, parser.edge_cases);

        // Value starting with a stop word
        parser.set_threshold(0.6);
        let parsed = parser.run("je veux écouter les the rolling").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "the rolling".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
                resolved_value_ids: vec![],
                matched_value: "the rolling stones".to_string(),
                range: 20..31,
            }]
        );

        // Value starting with a stop word and ending with one
        parser.set_threshold(1.0);
        let parsed = parser
            .run("je veux écouter les the rolling stones")
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "the rolling stones".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
                resolved_value_ids: vec![],
                matched_value: "the rolling stones".to_string(),
                range: 20..38,
            }]
        );

        // Value starting with two stop words
        parser.set_threshold(1.0);
        let parsed = parser
            .run("je veux écouter les the stones rolling")
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "the stones rolling".to_string(),
                resolved_value: "The Stones Rolling".to_string(),
                resolved_value_ids: vec![],
                matched_value: "the stones rolling".to_string(),
                range: 20..38,
            }]
        );

        // Edge case
        parser.set_threshold(1.0);
        let parsed = parser.run("je veux écouter les the stones").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "the stones".to_string(),
                resolved_value: "The Stones".to_string(),
                resolved_value_ids: vec![],
                matched_value: "the stones".to_string(),
                range: 20..30,
            }]
        );

        // Edge case should not match if not present in full
        parser.set_threshold(0.5);
        let parsed = parser.run("je veux écouter les the").unwrap();
        assert_eq!(parsed, vec![]);

        // Sentence containing an additional stop word which is absent from the gazetteer
        let parsed = parser
            .run("hello I want to listen to the rolling stones")
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "the rolling stones".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
                resolved_value_ids: vec![],
                matched_value: "the rolling stones".to_string(),
                range: 26..44,
            }]
        );

        // Multiple stop words at the beginning of a value
        let parsed = parser
            .run("hello I want to listen to the the rolling stones")
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "the rolling stones".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
                resolved_value_ids: vec![],
                matched_value: "the rolling stones".to_string(),
                range: 30..48,
            }]
        );
    }

    #[test]
    fn test_parser_base() {
        let mut gazetteer = Gazetteer::default();
        gazetteer.add(EntityValue {
            resolved_value: "The Flying Stones".to_string(),
            resolved_value_id: None,
            raw_value: "the flying stones".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            resolved_value_id: None,
            raw_value: "the rolling stones".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Blink-182".to_string(),
            resolved_value_id: None,
            raw_value: "blink one eight two".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Je Suis Animal".to_string(),
            resolved_value_id: None,
            raw_value: "je suis animal".to_string(),
        });

        let parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.0)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let mut parsed = parser.run("je veux écouter les rolling stones").unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    raw_value: "je".to_string(),
                    resolved_value: "Je Suis Animal".to_string(),
                    resolved_value_ids: vec![],
                    matched_value: "je suis animal".to_string(),
                    range: 0..2,
                },
                ParsedValue {
                    raw_value: "rolling stones".to_string(),
                    resolved_value: "The Rolling Stones".to_string(),
                    resolved_value_ids: vec![],
                    matched_value: "the rolling stones".to_string(),
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
                    resolved_value_ids: vec![],
                    matched_value: "je suis animal".to_string(),
                    range: 0..2,
                },
                ParsedValue {
                    raw_value: "rolling stones".to_string(),
                    resolved_value: "The Rolling Stones".to_string(),
                    resolved_value_ids: vec![],
                    matched_value: "the rolling stones".to_string(),
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
                    resolved_value_ids: vec![],
                    matched_value: "the rolling stones".to_string(),
                    range: 20..34,
                },
                ParsedValue {
                    raw_value: "blink eight".to_string(),
                    resolved_value: "Blink-182".to_string(),
                    resolved_value_ids: vec![],
                    matched_value: "blink one eight two".to_string(),
                    range: 39..50,
                },
            ]
        );

        parsed = parser.run("joue moi quelque chose").unwrap();
        assert_eq!(parsed, vec![]);
    }

    #[test]
    fn test_parser_multiple_raw_values() {
        let mut gazetteer = Gazetteer::default();
        gazetteer.add(EntityValue {
            resolved_value: "Blink-182".to_string(),
            resolved_value_id: None,
            raw_value: "blink one eight two".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Blink-182".to_string(),
            resolved_value_id: None,
            raw_value: "blink 182".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Blink-182".to_string(),
            resolved_value_id: None,
            raw_value: "blink".to_string(),
        });

        let mut parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.0)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let mut parsed = parser.run("let's listen to blink 182").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "blink 182".to_string(),
                resolved_value: "Blink-182".to_string(),
                resolved_value_ids: vec![],
                matched_value: "blink 182".to_string(),
                range: 16..25,
            }]
        );

        parser.set_threshold(0.5);
        parsed = parser.run("let's listen to blink").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "blink".to_string(),
                resolved_value: "Blink-182".to_string(),
                resolved_value_ids: vec![],
                matched_value: "blink".to_string(),
                range: 16..21,
            }]
        );

        parser.set_threshold(0.5);
        parsed = parser.run("let's listen to one eight two").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "one eight two".to_string(),
                resolved_value: "Blink-182".to_string(),
                resolved_value_ids: vec![],
                matched_value: "blink one eight two".to_string(),
                range: 16..29,
            }]
        );
    }

    #[test]
    fn test_parser_with_resolved_value_ids() {
        let mut gazetteer = Gazetteer::default();
        gazetteer.add(EntityValue {
            resolved_value: "Blink-182".to_string(),
            resolved_value_id: Some("42".to_string()),
            raw_value: "blink one eight two".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Blink-182".to_string(),
            resolved_value_id: Some("43".to_string()),
            raw_value: "blink 182".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Blink".to_string(),
            resolved_value_id: Some("44".to_string()),
            raw_value: "blink".to_string(),
        });

        let mut parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.0)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let mut parsed = parser.run("let's listen to blink 182").unwrap();
        assert_eq!(
            vec![ParsedValue {
                raw_value: "blink 182".to_string(),
                resolved_value: "Blink-182".to_string(),
                resolved_value_ids: vec!["42".to_string(), "43".to_string()],
                matched_value: "blink 182".to_string(),
                range: 16..25,
            }],
            parsed
        );

        parser.set_threshold(1.0);
        parsed = parser.run("let's listen to blink").unwrap();
        assert_eq!(
            vec![ParsedValue {
                raw_value: "blink".to_string(),
                resolved_value: "Blink".to_string(),
                resolved_value_ids: vec!["44".to_string()],
                matched_value: "blink".to_string(),
                range: 16..21,
            }],
            parsed
        );

        parser.set_threshold(3.0 / 4.0);
        parsed = parser.run("let's listen to one eight two").unwrap();
        assert_eq!(
            vec![ParsedValue {
                raw_value: "one eight two".to_string(),
                resolved_value: "Blink-182".to_string(),
                resolved_value_ids: vec!["42".to_string(), "43".to_string()],
                matched_value: "blink one eight two".to_string(),
                range: 16..29,
            }],
            parsed
        );
    }

    #[test]
    fn test_parser_with_ranking() {
        let mut gazetteer = Gazetteer::default();
        gazetteer.add(EntityValue {
            resolved_value: "Jacques Brel".to_string(),
            resolved_value_id: None,
            raw_value: "jacques brel".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            resolved_value_id: None,
            raw_value: "the rolling stones".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Flying Stones".to_string(),
            resolved_value_id: None,
            raw_value: "the flying stones".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Daniel Brel".to_string(),
            resolved_value_id: None,
            raw_value: "daniel brel".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Jacques".to_string(),
            resolved_value_id: None,
            raw_value: "jacques".to_string(),
        });
        let parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        // When there is a tie in terms of number of token matched, match the most popular choice
        let parsed = parser.run("je veux écouter the stones").unwrap();
        assert_eq!(
            vec![ParsedValue {
                raw_value: "the stones".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
                resolved_value_ids: vec![],
                matched_value: "the rolling stones".to_string(),
                range: 16..26,
            }],
            parsed
        );

        let parsed = parser.run("je veux écouter brel").unwrap();
        assert_eq!(
            vec![ParsedValue {
                raw_value: "brel".to_string(),
                resolved_value: "Jacques Brel".to_string(),
                resolved_value_ids: vec![],
                matched_value: "jacques brel".to_string(),
                range: 16..20,
            }],
            parsed
        );

        // Resolve to the value with more words matching regardless of popularity
        let parsed = parser.run("je veux écouter the flying stones").unwrap();
        assert_eq!(
            vec![ParsedValue {
                raw_value: "the flying stones".to_string(),
                resolved_value: "The Flying Stones".to_string(),
                resolved_value_ids: vec![],
                matched_value: "the flying stones".to_string(),
                range: 16..33,
            }],
            parsed
        );

        let parsed = parser.run("je veux écouter daniel brel").unwrap();
        assert_eq!(
            vec![ParsedValue {
                raw_value: "daniel brel".to_string(),
                resolved_value: "Daniel Brel".to_string(),
                resolved_value_ids: vec![],
                matched_value: "daniel brel".to_string(),
                range: 16..27,
            }],
            parsed
        );

        let parsed = parser.run("je veux écouter jacques").unwrap();
        assert_eq!(
            vec![ParsedValue {
                raw_value: "jacques".to_string(),
                resolved_value: "Jacques".to_string(),
                resolved_value_ids: vec![],
                matched_value: "jacques".to_string(),
                range: 16..23,
            }],
            parsed
        );
    }

    #[test]
    fn test_preprend_values() {
        let mut gazetteer = Gazetteer::default();
        gazetteer.add(EntityValue {
            resolved_value: "Jacques Brel".to_string(),
            resolved_value_id: None,
            raw_value: "jacques brel".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            resolved_value_id: None,
            raw_value: "the rolling stones".to_string(),
        });
        let mut parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let input = "je veux écouter brel";

        let parsed = parser.run(input).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "brel".to_string(),
                resolved_value: "Jacques Brel".to_string(),
                resolved_value_ids: vec![],
                matched_value: "jacques brel".to_string(),
                range: 16..20,
            }]
        );

        let values_to_prepend = vec![
            EntityValue {
                resolved_value: "Daniel Brel".to_string(),
                resolved_value_id: None,
                raw_value: "daniel brel".to_string(),
            },
            EntityValue {
                resolved_value: "Eric Brel".to_string(),
                resolved_value_id: None,
                raw_value: "eric brel".to_string(),
            },
        ];

        parser.prepend_values(values_to_prepend);

        let parsed = parser.run(input).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "brel".to_string(),
                resolved_value: "Daniel Brel".to_string(),
                resolved_value_ids: vec![],
                matched_value: "daniel brel".to_string(),
                range: 16..20,
            }]
        );
    }

    #[test]
    fn test_parser_with_restart() {
        let mut gazetteer = Gazetteer::default();
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            resolved_value_id: None,
            raw_value: "the rolling stones".to_string(),
        });

        let parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let parsed = parser
            .run("the music I want to listen to is rolling on stones")
            .unwrap();
        assert_eq!(parsed, vec![]);
    }

    #[test]
    fn test_parser_with_unicode_whitespace() {
        let mut gazetteer = Gazetteer::default();
        gazetteer.add(EntityValue {
            resolved_value: "Quand est-ce ?".to_string(),
            resolved_value_id: None,
            raw_value: "quand est -ce".to_string(),
        });
        let parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let parsed = parser.run("non quand est survivre").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                resolved_value: "Quand est-ce ?".to_string(),
                resolved_value_ids: vec![],
                range: 4..13,
                raw_value: "quand est".to_string(),
                matched_value: "quand est -ce".to_string(),
            }]
        )
    }

    #[test]
    fn test_parser_with_mixed_ordered_entity() {
        let mut gazetteer = Gazetteer::default();
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            resolved_value_id: None,
            raw_value: "the rolling stones".to_string(),
        });

        let parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let parsed = parser.run("rolling the stones").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                resolved_value: "The Rolling Stones".to_string(),
                resolved_value_ids: vec![],
                range: 8..18,
                raw_value: "the stones".to_string(),
                matched_value: "the rolling stones".to_string(),
            }]
        );
    }

    #[test]
    fn test_parser_with_threshold() {
        let mut gazetteer = Gazetteer::default();
        gazetteer.add(EntityValue {
            resolved_value: "The Flying Stones".to_string(),
            resolved_value_id: None,
            raw_value: "the flying stones".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            resolved_value_id: None,
            raw_value: "the rolling stones".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Blink-182".to_string(),
            resolved_value_id: None,
            raw_value: "blink one eight two".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Je Suis Animal".to_string(),
            resolved_value_id: None,
            raw_value: "je suis animal".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Les Enfoirés".to_string(),
            resolved_value_id: None,
            raw_value: "les enfoirés".to_string(),
        });

        let mut parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let parsed = parser.run("je veux écouter les rolling stones").unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    resolved_value: "Les Enfoirés".to_string(),
                    resolved_value_ids: vec![],
                    range: 16..19,
                    raw_value: "les".to_string(),
                    matched_value: "les enfoirés".to_string(),
                },
                ParsedValue {
                    raw_value: "rolling stones".to_string(),
                    resolved_value: "The Rolling Stones".to_string(),
                    resolved_value_ids: vec![],
                    matched_value: "the rolling stones".to_string(),
                    range: 20..34,
                },
            ]
        );

        parser.set_threshold(0.3);
        let parsed = parser.run("je veux écouter les rolling stones").unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    raw_value: "je".to_string(),
                    resolved_value: "Je Suis Animal".to_string(),
                    resolved_value_ids: vec![],
                    matched_value: "je suis animal".to_string(),
                    range: 0..2,
                },
                ParsedValue {
                    resolved_value: "Les Enfoirés".to_string(),
                    resolved_value_ids: vec![],
                    matched_value: "les enfoirés".to_string(),
                    range: 16..19,
                    raw_value: "les".to_string(),
                },
                ParsedValue {
                    raw_value: "rolling stones".to_string(),
                    resolved_value: "The Rolling Stones".to_string(),
                    resolved_value_ids: vec![],
                    matched_value: "the rolling stones".to_string(),
                    range: 20..34,
                },
            ]
        );

        parser.set_threshold(0.6);
        let parsed = parser.run("je veux écouter les rolling stones").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "rolling stones".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
                resolved_value_ids: vec![],
                matched_value: "the rolling stones".to_string(),
                range: 20..34,
            }]
        );
    }

    #[test]
    fn test_repeated_words() {
        let mut gazetteer = Gazetteer::default();
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            resolved_value_id: None,
            raw_value: "the rolling stones".to_string(),
        });

        let mut parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let parsed = parser.run("the the the").unwrap();
        assert_eq!(parsed, vec![]);

        parser.set_threshold(1.0);
        let parsed = parser
            .run("the the the rolling stones stones stones stones")
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "the rolling stones".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
                resolved_value_ids: vec![],
                matched_value: "the rolling stones".to_string(),
                range: 8..26,
            }]
        );
    }

    #[test]
    fn test_injection_ranking() {
        let mut gazetteer = Gazetteer::default();
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            resolved_value_id: None,
            raw_value: "the rolling stones".to_string(),
        });

        let mut parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.6)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let new_values = vec![EntityValue {
            resolved_value: "The Flying Stones".to_string(),
            resolved_value_id: None,
            raw_value: "the flying stones".to_string(),
        }];

        // Test with preprend set to false
        parser
            .inject_new_values(new_values.clone(), false, false)
            .unwrap();
        let parsed = parser.run("je veux écouter les flying stones").unwrap();

        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "flying stones".to_string(),
                resolved_value: "The Flying Stones".to_string(),
                resolved_value_ids: vec![],
                matched_value: "the flying stones".to_string(),
                range: 20..33,
            }]
        );

        let parsed = parser.run("je veux écouter the stones").unwrap();

        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "the stones".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
                resolved_value_ids: vec![],
                matched_value: "the rolling stones".to_string(),
                range: 16..26,
            }]
        );

        // Test with preprend set to true
        parser.inject_new_values(new_values, true, false).unwrap();

        let parsed = parser.run("je veux écouter les flying stones").unwrap();

        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "flying stones".to_string(),
                resolved_value: "The Flying Stones".to_string(),
                resolved_value_ids: vec![],
                matched_value: "the flying stones".to_string(),
                range: 20..33,
            }]
        );

        let parsed = parser.run("je veux écouter the stones").unwrap();

        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "the stones".to_string(),
                resolved_value: "The Flying Stones".to_string(),
                resolved_value_ids: vec![],
                matched_value: "the flying stones".to_string(),
                range: 16..26,
            }]
        );
    }

    #[test]
    fn test_injection_from_vanilla() {
        let mut gazetteer = Gazetteer::default();
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            resolved_value_id: None,
            raw_value: "the rolling stones".to_string(),
        });

        let mut parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.6)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let new_values_1 = vec![EntityValue {
            resolved_value: "The Flying Stones".to_string(),
            resolved_value_id: None,
            raw_value: "the flying stones".to_string(),
        }];

        parser.inject_new_values(new_values_1, true, false).unwrap();

        // Test injection from vanilla
        let new_values_2 = vec![EntityValue {
            resolved_value: "Queens Of The Stone Age".to_string(),
            resolved_value_id: None,
            raw_value: "queens of the stone age".to_string(),
        }];

        let flying_idx = *parser.tokens_symbol_table.find_symbol("flying").unwrap();
        let stones_idx = *parser.tokens_symbol_table.find_symbol("stones").unwrap();
        let flying_stones_idx = *parser
            .resolved_symbol_table
            .find_symbol("The Flying Stones")
            .first()
            .unwrap();
        parser.inject_new_values(new_values_2, true, true).unwrap();

        let parsed = parser.run("je veux écouter les flying stones").unwrap();
        assert_eq!(parsed, vec![]);

        let parsed = parser.run("je veux écouter queens the stone age").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "queens the stone age".to_string(),
                resolved_value: "Queens Of The Stone Age".to_string(),
                resolved_value_ids: vec![],
                matched_value: "queens of the stone age".to_string(),
                range: 16..36,
            }]
        );

        assert!(parser
            .resolved_symbol_table
            .find_symbol("The Flying Stones")
            .is_empty());
        assert!(parser.tokens_symbol_table.find_symbol("flying").is_none());
        assert!(!parser.token_to_resolved_values.contains_key(&flying_idx));
        assert!(!parser.token_to_resolved_values[&stones_idx].contains(&flying_stones_idx));
        assert!(!parser
            .resolved_value_to_tokens
            .contains_key(&flying_stones_idx));
    }

    #[test]
    fn test_injection_stop_words() {
        let mut gazetteer = Gazetteer::default();
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            resolved_value_id: None,
            raw_value: "the rolling stones".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Stones".to_string(),
            resolved_value_id: None,
            raw_value: "the stones".to_string(),
        });

        let mut parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.0)
            .gazetteer(gazetteer.clone())
            .n_stop_words(2)
            .additional_stop_words(vec!["hello".to_string()])
            .build()
            .unwrap();

        let parser_no_stop_words = ParserBuilder::default()
            .minimum_tokens_ratio(0.0)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let mut expected_stop_words = vec!["the", "stones", "hello"]
            .into_iter()
            .map(|sym| *parser.tokens_symbol_table.find_symbol(sym).unwrap())
            .collect();

        let mut expected_edge_cases: HashSet<u32> = HashSet::default();
        expected_edge_cases.insert(parser.resolved_symbol_table.find_symbol("The Stones")[0]);

        assert_eq!(parser.stop_words, expected_stop_words);
        assert_eq!(parser.edge_cases, expected_edge_cases);

        let new_values = vec![
            EntityValue {
                resolved_value: "Rolling".to_string(),
                resolved_value_id: None,
                raw_value: "rolling".to_string(),
            },
            EntityValue {
                resolved_value: "Rolling Two".to_string(),
                resolved_value_id: None,
                raw_value: "rolling two".to_string(),
            },
        ];

        parser.inject_new_values(new_values, true, false).unwrap();

        expected_stop_words.remove(parser.tokens_symbol_table.find_symbol("stones").unwrap());
        expected_stop_words.insert(*parser.tokens_symbol_table.find_symbol("rolling").unwrap());

        expected_edge_cases.remove(&parser.resolved_symbol_table.find_symbol("The Stones")[0]);
        expected_edge_cases.insert(parser.resolved_symbol_table.find_symbol("Rolling")[0]);

        assert_eq!(expected_stop_words, parser.stop_words);
        assert_eq!(expected_edge_cases, parser.edge_cases);

        // No stop words case
        assert!(parser_no_stop_words.stop_words.is_empty());
        assert!(parser_no_stop_words.edge_cases.is_empty());
    }

    #[test]
    fn test_match_longest_substring() {
        let mut gazetteer = Gazetteer::default();
        gazetteer.add(EntityValue {
            resolved_value: "Black And White".to_string(),
            resolved_value_id: None,
            raw_value: "black and white".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Album".to_string(),
            resolved_value_id: None,
            raw_value: "album".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Black and White Album".to_string(),
            resolved_value_id: None,
            raw_value: "the black and white album".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "1 2 3 4".to_string(),
            resolved_value_id: None,
            raw_value: "one two three four".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "3 4 5 6".to_string(),
            resolved_value_id: None,
            raw_value: "three four five six".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "6 7".to_string(),
            resolved_value_id: None,
            raw_value: "six seven".to_string(),
        });

        let parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let parsed = parser
            .run("je veux écouter le black and white album")
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "black and white album".to_string(),
                resolved_value: "The Black and White Album".to_string(),
                resolved_value_ids: vec![],
                matched_value: "the black and white album".to_string(),
                range: 19..40,
            }]
        );

        let parsed = parser.run("one two three four").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "one two three four".to_string(),
                resolved_value: "1 2 3 4".to_string(),
                resolved_value_ids: vec![],
                matched_value: "one two three four".to_string(),
                range: 0..18,
            }]
        );

        // This test is ambiguous and there may be several acceptable answers...
        let parsed = parser.run("zero one two three four five six").unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    raw_value: "one two three four".to_string(),
                    resolved_value: "1 2 3 4".to_string(),
                    resolved_value_ids: vec![],
                    matched_value: "one two three four".to_string(),
                    range: 5..23,
                },
                ParsedValue {
                    raw_value: "five six".to_string(),
                    resolved_value: "3 4 5 6".to_string(),
                    resolved_value_ids: vec![],
                    matched_value: "three four five six".to_string(),
                    range: 24..32,
                },
            ]
        );

        let parsed = parser
            .run("zero one two three four five six seven")
            .unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    raw_value: "one two three four".to_string(),
                    resolved_value: "1 2 3 4".to_string(),
                    resolved_value_ids: vec![],
                    matched_value: "one two three four".to_string(),
                    range: 5..23,
                },
                ParsedValue {
                    raw_value: "six seven".to_string(),
                    resolved_value: "6 7".to_string(),
                    resolved_value_ids: vec![],
                    matched_value: "six seven".to_string(),
                    range: 29..38,
                },
            ]
        );
    }

    #[test]
    #[ignore]
    fn real_world_gazetteer_parser() {
        let (_, body) = CallBuilder::get().max_response(20_000_000).timeout_ms(60_000).url("https://s3.amazonaws.com/snips/nlu-lm/test/gazetteer-entity-parser/artist_gazetteer_formatted.json").unwrap().exec().unwrap();
        let data: Vec<EntityValue> = serde_json::from_reader(&*body).unwrap();
        let gaz = Gazetteer { data };

        let n_stop_words = 50;
        let mut parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.6)
            .gazetteer(gaz)
            .n_stop_words(n_stop_words)
            .build()
            .unwrap();

        let parsed = parser.run("je veux écouter les rolling stones").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "rolling stones".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
                resolved_value_ids: vec![],
                matched_value: "the rolling stones".to_string(),
                range: 20..34,
            }]
        );

        parser.set_threshold(0.5);
        let parsed = parser.run("je veux écouter bowie").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "bowie".to_string(),
                resolved_value: "David Bowie".to_string(),
                resolved_value_ids: vec![],
                matched_value: "david bowie".to_string(),
                range: 16..21,
            }]
        );

        let (_, body) = CallBuilder::get().max_response(20_000_000).timeout_ms(60_000).url("https://s3.amazonaws.com/snips/nlu-lm/test/gazetteer-entity-parser/album_gazetteer_formatted.json").unwrap().exec().unwrap();
        let data: Vec<EntityValue> = serde_json::from_reader(&*body).unwrap();
        let gaz = Gazetteer { data };

        let n_stop_words = 50;
        let mut parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.6)
            .gazetteer(gaz)
            .n_stop_words(n_stop_words)
            .build()
            .unwrap();

        let parsed = parser
            .run("je veux écouter le black and white album")
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "black and white album".to_string(),
                resolved_value: "The Black and White Album".to_string(),
                resolved_value_ids: vec![],
                matched_value: "black and white album".to_string(),
                range: 19..40,
            }]
        );

        let parsed = parser
            .run("je veux écouter dark side of the moon")
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "dark side of the moon".to_string(),
                resolved_value: "Dark Side of the Moon".to_string(),
                resolved_value_ids: vec![],
                matched_value: "dark side of the moon".to_string(),
                range: 16..37,
            }]
        );

        parser.set_threshold(0.5);
        let parsed = parser
            .run("je veux écouter dark side of the moon")
            .unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    raw_value: "je veux".to_string(),
                    resolved_value: "Je veux du bonheur".to_string(),
                    resolved_value_ids: vec![],
                    matched_value: "je veux du bonheur".to_string(),
                    range: 0..7,
                },
                ParsedValue {
                    raw_value: "dark side of the moon".to_string(),
                    resolved_value: "Dark Side of the Moon".to_string(),
                    resolved_value_ids: vec![],
                    matched_value: "dark side of the moon".to_string(),
                    range: 16..37,
                },
            ]
        );
    }

    #[test]
    #[ignore]
    fn test_real_word_injection() {
        // Real-world artist gazetteer
        let (_, body) = CallBuilder::get().max_response(20_000_000).timeout_ms(100_000).url("https://s3.amazonaws.com/snips/nlu-lm/test/gazetteer-entity-parser/artist_gazetteer_formatted.json").unwrap().exec().unwrap();
        let data: Vec<EntityValue> = serde_json::from_reader(&*body).unwrap();
        let album_gaz = Gazetteer { data };

        let mut parser_for_test = ParserBuilder::default()
            .minimum_tokens_ratio(0.6)
            .gazetteer(album_gaz.clone())
            .n_stop_words(50)
            .build()
            .unwrap();

        let mut parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.6)
            .gazetteer(album_gaz)
            .n_stop_words(50)
            .build()
            .unwrap();

        // Get 10k values from the album gazetter to inject in the album parser
        let (_, body) = CallBuilder::get().max_response(20_000_000).timeout_ms(100_000).url("https://s3.amazonaws.com/snips/nlu-lm/test/gazetteer-entity-parser/album_gazetteer_formatted.json").unwrap().exec().unwrap();
        let mut new_values: Vec<EntityValue> = serde_json::from_reader(&*body).unwrap();
        new_values.truncate(10000);

        // Test injection
        parser_for_test.set_threshold(0.7);
        let parsed = parser_for_test
            .run("je veux écouter hans knappertsbusch conducts")
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "hans knappertsbusch".to_string(),
                resolved_value: "Hans Knappertsbusch".to_string(),
                resolved_value_ids: vec![],
                matched_value: "hans knappertsbusch".to_string(),
                range: 16..35,
            }]
        );
        parser_for_test
            .inject_new_values(new_values.clone(), true, false)
            .unwrap();
        parser_for_test.set_threshold(0.7);
        let parsed = parser_for_test
            .run("je veux écouter hans knappertsbusch conducts")
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "hans knappertsbusch conducts".to_string(),
                resolved_value: "Hans Knappertsbusch conducts".to_string(),
                resolved_value_ids: vec![],
                matched_value: "hans knappertsbusch conducts".to_string(),
                range: 16..44,
            }]
        );

        let now = Instant::now();
        parser.inject_new_values(new_values, true, false).unwrap();
        let total_time = now.elapsed().as_secs();
        assert!(total_time < 10);
    }
}
