use constants::*;
use data::EntityValue;
use errors::*;
use fnv::FnvHashMap as HashMap;
use fnv::FnvHashSet as HashSet;
use rmps::{from_read, Serializer};
use serde::Serialize;
use serde_json;
use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::BinaryHeap;
use std::fs;
use std::ops::Range;
use std::path::Path;
use std::result::Result;
use symbol_table::GazetteerParserSymbolTable;
use utils::{check_threshold, whitespace_tokenizer};

/// Struct representing the parser. The Parser will match the longest possible contiguous
/// substrings of a query that match partial entity values. The order in which the values are
/// added to the parser matters: In case of ambiguity between two parsings, the Parser will output
/// the value that was added first (see Gazetteer).
#[derive(PartialEq, Debug, Serialize, Deserialize, Default)]
pub struct Parser {
    // Symbol table for the raw tokens
    tokens_symbol_table: GazetteerParserSymbolTable,
    // Symbol table for the resolved values
    // The latter differs from the first one in that it can contain the same resolved value
    // multiple times (to allow for multiple raw values corresponidng to the same resolved value)
    resolved_symbol_table: GazetteerParserSymbolTable,
    // maps each token to its count in the dataset
    token_to_count: HashMap<u32, u32>,
    // maps token to set of resolved values containing token
    token_to_resolved_values: HashMap<u32, HashSet<u32>>,
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
    last_token_in_resolution: usize,
    rank: u32,
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
    /// Add a single entity value to the parser
    pub(crate) fn add_value(
        &mut self,
        entity_value: EntityValue,
        rank: u32,
    ) -> Result<(), AddValueError> {
        // We force add the new resolved value: even if it already is present in the symbol table
        // we duplicate it to allow several raw values to map to it

        let EntityValue {
            raw_value,
            resolved_value,
        } = entity_value;
        let res_value_idx = self
            .resolved_symbol_table
            .add_symbol(resolved_value, true)
            .map_err(|cause| match cause.clone() {
                SymbolTableAddSymbolError::MissingKeyError { key: k }
                | SymbolTableAddSymbolError::DuplicateSymbolError {
                    0: DuplicateSymbolError { symbol: k },
                } => AddValueError {
                    kind: AddValueErrorKind::ResolvedValue,
                    value: k,
                    cause,
                },
            })?;

        for (_, token) in whitespace_tokenizer(&raw_value) {
            let token_idx = self
                .tokens_symbol_table
                .add_symbol(token, false)
                .map_err(|cause| AddValueError {
                    kind: AddValueErrorKind::RawValue,
                    value: raw_value.clone(),
                    cause,
                })?;
            // Update token_to_resolved_values map
            self.token_to_resolved_values
                .entry(token_idx)
                .and_modify(|e| {
                    e.insert(res_value_idx);
                })
                .or_insert({
                    let mut h = HashSet::default();
                    h.insert(res_value_idx);
                    h
                });

            // Update token count
            self.token_to_count
                .entry(token_idx)
                .and_modify(|c| *c += 1)
                .or_insert(1);

            // Update resolved_value_to_tokens map
            self.resolved_value_to_tokens
                .entry(res_value_idx)
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
    pub fn set_stop_words(
        &mut self,
        n_stop_words: usize,
        additional_stop_words: Option<Vec<String>>,
    ) -> Result<(), SetStopWordsError> {
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
        // the symbol table and to tokens_to_resolved_value)
        if let Some(additional_stop_words_vec) = additional_stop_words {
            self.additional_stop_words = additional_stop_words_vec.clone();
            for tok_s in additional_stop_words_vec.into_iter() {
                let tok_idx = self
                    .tokens_symbol_table
                    .add_symbol(tok_s, false)
                    .map_err(|cause| SetStopWordsError { cause })?;
                self.stop_words.insert(tok_idx);
                self.token_to_resolved_values
                    .entry(tok_idx)
                    .or_insert(HashSet::default());
            }
        }

        // Update the set of edge_cases. i.e. resolved value that only contain stop words

        // Reset edge cases
        self.edge_cases = HashSet::default();

        'outer: for (res_val, (_, tokens)) in &self.resolved_value_to_tokens {
            for tok in tokens {
                if !(self.stop_words.contains(tok)) {
                    continue 'outer;
                }
            }
            self.edge_cases.insert(*res_val);
        }

        Ok(())
    }

    /// Get the set of stop words
    pub fn get_stop_words(&self) -> Result<HashSet<String>, GetStopWordsError> {
        self.stop_words
            .iter()
            .map(|idx| {
                self.tokens_symbol_table
                    .find_index(idx)
                    .map_err(|cause| GetStopWordsError { cause })
            })
            .collect()
    }

    /// Get the set of edge cases, containing only stop words
    pub fn get_edge_cases(&self) -> Result<HashSet<String>, GetEdgeCasesError> {
        self.edge_cases
            .iter()
            .map(|idx| {
                self.resolved_symbol_table
                    .find_index(idx)
                    .map_err(|cause| GetEdgeCasesError { cause })
            })
            .collect()
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
    ) -> Result<(), InjectionError> {
        if from_vanilla {
            // Remove the resolved values form the resolved_symbol_table
            // Remove the resolved value from the resolved_value_to_tokens map
            // remove the corresponding resolved value from the token_to_resolved_value map
            // if after removing, a token is left absent from all resolved entities, then remove
            // it from the tokens_count, tokens_to_resolved_value maps and from the
            // tokens_symbol_table
            let mut tokens_marked_for_removal: HashSet<u32> = HashSet::default();
            for val in &self.injected_values {
                if let Some(res_val_indices) = self.resolved_symbol_table.remove_symbol(&val) {
                    for res_val in res_val_indices {
                        let (_, tokens) = self
                            .get_tokens_from_resolved_value(&res_val)
                            .map_err(|cause| InjectionError {
                                cause: InjectionRootError::TokensFromResolvedValueError(cause),
                            })?
                            .clone();
                        self.resolved_value_to_tokens.remove(&res_val);
                        for tok in tokens {
                            self.token_to_resolved_values.entry(tok).and_modify(|v| {
                                v.remove(&res_val);
                            });
                            // Check the remaining resolved values containing the token
                            if self
                                .get_resolved_values_from_token(&tok)
                                .map_err(|cause| InjectionError {
                                    cause: InjectionRootError::ResolvedValuesFromTokenError(cause),
                                })?
                                .is_empty()
                            {
                                tokens_marked_for_removal.insert(tok);
                            }
                        }
                    }
                }
            }
            for tok_idx in tokens_marked_for_removal {
                let tok = self
                    .tokens_symbol_table
                    .find_index(&tok_idx)
                    .map_err(|cause| InjectionError {
                        cause: InjectionRootError::SymbolTableFindIndexError(cause),
                    })?;
                if let Some(tok_indices) = self.tokens_symbol_table.remove_symbol(&tok) {
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
                self.resolved_value_to_tokens
                    .entry(*res_val)
                    .and_modify(|(rank, _)| *rank += n_new_values);
            }
        }

        let new_start_rank = match prepend {
            true => 0, // we inject new values from rank 0 to n_new_values - 1
            false => self.resolved_value_to_tokens.len(), // we inject new values from the current
                        // last rank onwards
        } as u32;

        for (rank, entity_value) in new_values.into_iter().enumerate() {
            self.add_value(entity_value.clone(), new_start_rank + rank as u32)
                .map_err(|cause| InjectionError {
                    cause: InjectionRootError::AddValueError(cause),
                })?;
            self.injected_values.insert(entity_value.resolved_value);
        }

        // Update the stop words and edge cases
        let n_stop_words = self.n_stop_words;
        let additional_stop_words = self.additional_stop_words.clone();
        self.set_stop_words(n_stop_words, Some(additional_stop_words))
            .map_err(|cause| InjectionError {
                cause: InjectionRootError::SetStopWordsError(cause),
            })?;

        Ok(())
    }

    /// get resolved value
    fn get_tokens_from_resolved_value(
        &self,
        resolved_value: &u32,
    ) -> Result<&(u32, Vec<u32>), TokensFromResolvedValueError> {
        Ok(self
            .resolved_value_to_tokens
            .get(resolved_value)
            .ok_or_else(|| TokensFromResolvedValueError::MissingKeyError {
                key: *resolved_value,
            })?)
    }

    /// get resolved values from token
    fn get_resolved_values_from_token(
        &self,
        token: &u32,
    ) -> Result<&HashSet<u32>, ResolvedValuesFromTokenError> {
        Ok(self
            .token_to_resolved_values
            .get(token)
            .ok_or_else(|| ResolvedValuesFromTokenError::MissingKeyError { key: *token })?)
    }

    /// Find all possible matches in a string.
    /// Returns a hashmap, indexed by resolved values. The corresponding value is a vec of tuples
    /// each tuple is a possible match for the resolved value, and is made of
    // (range of match, number of skips, index of last matched token in the resolved value)
    fn find_possible_matches(
        &self,
        input: &str,
        threshold: f32,
    ) -> Result<BinaryHeap<PossibleMatch>, FindPossibleMatchError> {
        let mut possible_matches: HashMap<u32, PossibleMatch> =
            HashMap::with_capacity_and_hasher(1000, Default::default());
        let mut matches_heap: BinaryHeap<PossibleMatch> = BinaryHeap::default();
        let mut skipped_tokens: HashMap<usize, (Range<usize>, u32)> = HashMap::default();
        'outer: for (token_idx, (range, token)) in whitespace_tokenizer(input).enumerate() {
            let range_start = range.start;
            let range_end = range.end;
            match self
                .tokens_symbol_table
                .find_single_symbol(&token)
                .map_err(|cause| FindPossibleMatchError {
                    cause: FindPossibleMatchRootError::SymbolTableFindSingleSymbolError(cause),
                })? {
                Some(value) => {
                    if !self.stop_words.contains(&value) {
                        for res_val in self.get_resolved_values_from_token(&value).map_err(
                            |cause| FindPossibleMatchError {
                                cause: FindPossibleMatchRootError::ResolvedValuesFromTokenError(
                                    cause,
                                ),
                            },
                        )? {
                            let entry = possible_matches.entry(*res_val);
                            match entry {
                                Entry::Occupied(mut entry) => {
                                    self.update_previous_match(
                                        &mut *entry.get_mut(),
                                        res_val,
                                        token_idx,
                                        value,
                                        range_start,
                                        range_end,
                                        threshold,
                                        &mut matches_heap,
                                    ).map_err(|cause| FindPossibleMatchError { cause })?
                                }
                                Entry::Vacant(entry) => {
                                    if let Some(new_possible_match) =
                                        self.insert_new_possible_match(
                                            res_val,
                                            value,
                                            range_start,
                                            range_end,
                                            token_idx,
                                            threshold,
                                            &skipped_tokens,
                                        ).map_err(|cause| FindPossibleMatchError { cause })?
                                    {
                                        entry.insert(new_possible_match);
                                    }
                                }
                            }
                        }
                    } else {
                        skipped_tokens.insert(token_idx, (range_start..range_end, value));
                        // Iterate over all edge cases and try to add or update corresponding
                        // PossibleMatch's. Using a threshold of 1.
                        let res_vals_from_token = self
                            .get_resolved_values_from_token(&value)
                            .map_err(|cause| FindPossibleMatchError {
                                cause: FindPossibleMatchRootError::ResolvedValuesFromTokenError(
                                    cause,
                                ),
                            })?;
                        for res_val in &self.edge_cases {
                            if res_vals_from_token.contains(&res_val) {
                                let entry = possible_matches.entry(*res_val);
                                match entry {
                                    Entry::Occupied(mut entry) => {
                                        self.update_previous_match(
                                            &mut *entry.get_mut(),
                                            res_val,
                                            token_idx,
                                            value,
                                            range_start,
                                            range_end,
                                            1.0,
                                            &mut matches_heap,
                                        ).map_err(|cause| FindPossibleMatchError { cause })?
                                    }
                                    Entry::Vacant(entry) => {
                                        if let Some(new_possible_match) =
                                            self.insert_new_possible_match(
                                                res_val,
                                                value,
                                                range_start,
                                                range_end,
                                                token_idx,
                                                1.0,
                                                &skipped_tokens,
                                            ).map_err(|cause| FindPossibleMatchError { cause })?
                                        {
                                            entry.insert(new_possible_match);
                                        }
                                    }
                                }
                            }
                        }
                        // Iterate over current possible matches containing the stop word and
                        // try to grow them (but do not initiate a new possible match)
                        // Threshold depends on whether the res_val is a edge case of not
                        for (res_val, mut possible_match) in &mut possible_matches {
                            if res_vals_from_token.contains(res_val) {
                                let val_threshold = match self
                                    .edge_cases
                                    .contains(&possible_match.resolved_value)
                                {
                                    false => threshold,
                                    true => 1.0,
                                };
                                self.update_previous_match(
                                    &mut possible_match,
                                    res_val,
                                    token_idx,
                                    value,
                                    range_start,
                                    range_end,
                                    val_threshold,
                                    &mut matches_heap,
                                ).map_err(|cause| FindPossibleMatchError { cause })?;
                            }
                        }
                    }
                }
                None => continue,
            }
        }

        // Add to the heap the possible matches that remain
        for possible_match in possible_matches.values() {
            // We start by adding the PossibleMatch to the heap
            if possible_match.n_consumed_tokens > possible_match.raw_value_length {
                return Err(PossibleMatchRootError::PossibleMatchConsumedError {
                    possible_match: possible_match.clone(),
                }).map_err(|cause| FindPossibleMatchError {
                    cause: FindPossibleMatchRootError::PossibleMatchRootError(cause),
                });
            }
            // In case the resolved value is an edge case, we set the threshold to 1 for this
            // value
            let val_threshold = match self.edge_cases.contains(&possible_match.resolved_value) {
                false => threshold,
                true => 1.0,
            };
            if check_threshold(
                possible_match.n_consumed_tokens,
                possible_match.raw_value_length - possible_match.n_consumed_tokens,
                val_threshold,
            ) {
                matches_heap.push(possible_match.clone());
            }
        }
        Ok(matches_heap)
    }

    fn update_previous_match(
        &self,
        possible_match: &mut PossibleMatch,
        res_val: &u32,
        token_idx: usize,
        value: u32,
        range_start: usize,
        range_end: usize,
        threshold: f32,
        ref mut matches_heap: &mut BinaryHeap<PossibleMatch>,
    ) -> Result<(), FindPossibleMatchRootError> {
        let (rank, otokens) = self.get_tokens_from_resolved_value(res_val).unwrap();
        {
            if possible_match.resolved_value == *res_val
                && token_idx == possible_match.last_token_in_input + 1
            {
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
                        return Ok(());
                    }
                }
            }
        }

        // the token belongs to a new resolved value, or the previous
        // PossibleMatch cannot be grown further. We start a new
        // PossibleMatch

        if possible_match.n_consumed_tokens > possible_match.raw_value_length {
            return Err(PossibleMatchRootError::PossibleMatchConsumedError {
                possible_match: possible_match.clone(),
            }).map_err(|cause| FindPossibleMatchRootError::PossibleMatchRootError(cause));
        }

        if check_threshold(
            possible_match.n_consumed_tokens,
            possible_match.raw_value_length - possible_match.n_consumed_tokens,
            threshold,
        ) {
            matches_heap.push(possible_match.clone());
        }
        // Then we initialize a new PossibleMatch with the same res val
        let last_token_in_resolution = otokens.iter().position(|e| *e == value).ok_or_else(|| {
            FindPossibleMatchRootError::MissingTokenFromList {
                token_list: otokens.clone(),
                value,
            }
        })?;
        *possible_match = PossibleMatch {
            resolved_value: *res_val,
            range: range_start..range_end,
            tokens_range: token_idx..(token_idx + 1),
            raw_value_length: otokens.len() as u32,
            last_token_in_input: token_idx,
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
        res_val: &u32,
        value: u32,
        range_start: usize,
        range_end: usize,
        token_idx: usize,
        threshold: f32,
        skipped_tokens: &HashMap<usize, (Range<usize>, u32)>,
    ) -> Result<Option<PossibleMatch>, FindPossibleMatchRootError> {
        let (rank, otokens) = self.get_tokens_from_resolved_value(res_val).unwrap();
        let last_token_in_resolution = otokens.iter().position(|e| *e == value).ok_or_else(|| {
            FindPossibleMatchRootError::MissingTokenFromList {
                token_list: otokens.clone(),
                value,
            }
        })?;

        let mut possible_match = PossibleMatch {
            resolved_value: *res_val,
            range: range_start..range_end,
            tokens_range: token_idx..(token_idx + 1),
            last_token_in_input: token_idx,
            last_token_in_resolution,
            n_consumed_tokens: 1,
            raw_value_length: otokens.len() as u32,
            rank: *rank,
        };
        let mut n_skips = last_token_in_resolution as u32;

        // Bactrack to check if we left out from skipped words at the beginning
        'outer: for btok_idx in (0..token_idx).rev() {
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
                            break 'outer;
                        }
                    }
                    None => break 'outer,
                }
            } else {
                break 'outer;
            }
        }

        // Conservative estimate of threshold condition for early stopping
        // if we have already skipped more tokens than is permitted by the threshold condition,
        // there is no point in continuing
        if possible_match.raw_value_length < n_skips {
            return Err(FindPossibleMatchRootError::PossibleMatchRootError(
                PossibleMatchRootError::PossibleMatchSkippedError {
                    possible_match: possible_match,
                },
            ));
        }
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

    /// Parse the input string `input` and output a vec of `ParsedValue`.
    pub fn run(&self, input: &str) -> Result<Vec<ParsedValue>, RunError> {
        let matches_heap = self
            .find_possible_matches(input, self.threshold)
            .map_err(|cause| RunError {
                input: input.to_string(),
                cause: RunRootError::FindPossibleMatchError(cause),
            })?;
        let parsing = self
            .parse_input(input, matches_heap)
            .map_err(|cause| RunError {
                input: input.to_string(),
                cause: RunRootError::ParseInputError(cause),
            })?;
        Ok(parsing)
    }

    /// Set the threshold (minimum fraction of tokens to match for an entity to be parsed).
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    fn reduce_possible_match(
        input: &str,
        possible_match: PossibleMatch,
        overlapping_tokens: HashSet<usize>,
    ) -> Option<PossibleMatch> {
        let mut reduced_possible_match: Option<PossibleMatch> = None;
        for (token_idx, (token_range, _)) in whitespace_tokenizer(input).enumerate() {
            if possible_match.tokens_range.start < token_idx
                && possible_match.tokens_range.end > token_idx
                && !overlapping_tokens.contains(&token_idx)
            {
                if let Some(ref mut current_possible_match) = reduced_possible_match {
                    current_possible_match.range.end = token_range.end;
                    current_possible_match.tokens_range.end = token_idx + 1;
                    current_possible_match.n_consumed_tokens += 1;
                    continue;
                }

                reduced_possible_match = Some(PossibleMatch {
                    resolved_value: possible_match.resolved_value,
                    range: token_range,
                    tokens_range: token_idx..token_idx + 1,
                    raw_value_length: possible_match.resolved_value,
                    n_consumed_tokens: 1,
                    last_token_in_input: 0, // we are not going to need this one
                    last_token_in_resolution: 0, // we are not going to need this one
                    rank: possible_match.rank,
                })
            }
        }
        reduced_possible_match
    }

    fn parse_input(
        &self,
        input: &str,
        mut matches_heap: BinaryHeap<PossibleMatch>,
    ) -> Result<Vec<ParsedValue>, ParseInputError> {
        let mut taken_tokens: HashSet<usize> = HashSet::default();
        let n_total_tokens = whitespace_tokenizer(input).count();
        let mut parsing: BinaryHeap<ParsedValue> = BinaryHeap::default();

        'outer: while !matches_heap.is_empty() && taken_tokens.len() < n_total_tokens {
            let possible_match = matches_heap.pop().unwrap();

            let tokens_range_start = possible_match.tokens_range.start;
            let tokens_range_end = possible_match.tokens_range.end;

            let overlapping_tokens = {
                let mut rep: HashSet<usize> = HashSet::default();
                for t in tokens_range_start..tokens_range_end {
                    if taken_tokens.contains(&t) {
                        rep.insert(t);
                    }
                }
                rep
            };

            if !overlapping_tokens.is_empty() {
                let reduced_possible_matches =
                    Self::reduce_possible_match(input, possible_match, overlapping_tokens);
                if let Some(reduced_possible_match) = reduced_possible_matches {
                    let val_threshold = match self
                        .edge_cases
                        .contains(&reduced_possible_match.resolved_value)
                    {
                        false => self.threshold,
                        true => 1.0,
                    };
                    if check_threshold(
                        reduced_possible_match.n_consumed_tokens,
                        reduced_possible_match.raw_value_length
                            - reduced_possible_match.n_consumed_tokens,
                        val_threshold,
                    ) {
                        matches_heap.push(reduced_possible_match.clone());
                    }
                }
                continue 'outer;
            }

            let range_start = possible_match.range.start;
            let range_end = possible_match.range.end;
            parsing.push(ParsedValue {
                range: range_start..range_end,
                raw_value: input
                    .chars()
                    .skip(range_start)
                    .take(range_end - range_start)
                    .collect(),
                resolved_value: self
                    .resolved_symbol_table
                    .find_index(&possible_match.resolved_value)
                    .map_err(|cause| ParseInputError { cause })?,
            });
            for idx in tokens_range_start..tokens_range_end {
                taken_tokens.insert(idx);
            }
        }

        // Output ordered parsing
        Ok(parsing.into_sorted_vec())
    }

    fn get_parser_config(&self) -> Result<ParserConfig, GetParserConfigError> {
        Ok(ParserConfig {
            parser_filename: PARSER_FILE.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            threshold: self.threshold,
            stop_words: self
                .get_stop_words()
                .map_err(|cause| GetParserConfigError {
                    cause: GetParserConfigRootError::GetStopWordsError(cause),
                })?,
            edge_cases: self
                .get_edge_cases()
                .map_err(|cause| GetParserConfigError {
                    cause: GetParserConfigRootError::GetEdgeCasesError(cause),
                })?,
        })
    }

    /// Dump the parser to a folder
    pub fn dump<P: AsRef<Path>>(&self, folder_name: P) -> Result<(), DumpError> {
        fs::create_dir(folder_name.as_ref())
            .map_err(|cause| SerializationError::Io {
                path: folder_name.as_ref().to_path_buf(),
                cause,
            })
            .map_err(|cause| DumpError {
                cause: DumpRootError::SerializationError(cause),
            })?;

        let config = self.get_parser_config().map_err(|cause| DumpError {
            cause: DumpRootError::GetParserConfigError(cause),
        })?;

        let writer = fs::File::create(folder_name.as_ref().join(METADATA_FILENAME))
            .map_err(|cause| SerializationError::Io {
                path: folder_name.as_ref().join(METADATA_FILENAME).to_path_buf(),
                cause: cause,
            })
            .map_err(|cause| DumpError {
                cause: DumpRootError::SerializationError(cause),
            })?;

        serde_json::to_writer(writer, &config)
            .map_err(|cause| SerializationError::InvalidConfigFormat {
                path: folder_name.as_ref().join(METADATA_FILENAME),
                cause: cause,
            })
            .map_err(|cause| DumpError {
                cause: DumpRootError::SerializationError(cause),
            })?;

        let parser_path = folder_name.as_ref().join(config.parser_filename);
        let mut writer = fs::File::create(&parser_path)
            .map_err(|cause| SerializationError::Io {
                path: parser_path.clone(),
                cause: cause,
            })
            .map_err(|cause| DumpError {
                cause: DumpRootError::SerializationError(cause),
            })?;

        self.serialize(&mut Serializer::new(&mut writer))
            .map_err(|cause| SerializationError::ParserSerializationError {
                path: parser_path,
                cause: cause,
            })
            .map_err(|cause| DumpError {
                cause: DumpRootError::SerializationError(cause),
            })?;
        Ok(())
    }

    /// Load a resolver from a folder
    pub fn from_folder<P: AsRef<Path>>(folder_name: P) -> Result<Parser, LoadError> {
        let metadata_path = folder_name.as_ref().join(METADATA_FILENAME);
        let metadata_file = fs::File::open(&metadata_path).map_err(|cause| LoadError {
            cause: DeserializationError::Io {
                path: metadata_path.clone(),
                cause,
            },
        })?;

        let config: ParserConfig =
            serde_json::from_reader(metadata_file).map_err(|cause| LoadError {
                cause: DeserializationError::ReadConfigError {
                    path: metadata_path,
                    cause: cause,
                },
            })?;

        let parser_path = folder_name.as_ref().join(config.parser_filename);
        let reader = fs::File::open(&parser_path).map_err(|cause| LoadError {
            cause: DeserializationError::Io {
                path: parser_path.clone(),
                cause: cause,
            },
        })?;

        Ok(from_read(reader).map_err(|cause| LoadError {
            cause: DeserializationError::ParserDeserializationError {
                path: parser_path,
                cause: cause,
            },
        })?)
    }
}

#[cfg(test)]
mod tests {
    extern crate mio_httpc;
    extern crate tempfile;

    use self::mio_httpc::CallBuilder;
    use self::tempfile::tempdir;
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use data::EntityValue;
    use data::Gazetteer;
    use failure::ResultExt;
    use parser_builder::ParserBuilder;
    use std::time::Instant;

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
        let parser = ParserBuilder::new(gazetteer, 0.5)
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
        let mut gt_stop_words: HashSet<String> = HashSet::default();
        gt_stop_words.insert("the".to_string());
        gt_stop_words.insert("stones".to_string());
        gt_stop_words.insert("hello".to_string());
        assert_eq!(config.stop_words, gt_stop_words);
        let mut gt_edge_cases: HashSet<String> = HashSet::default();
        gt_edge_cases.insert("The Rolling Stones".to_string());
        assert_eq!(config.edge_cases, gt_edge_cases);

        tdir.close().unwrap();
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

        let mut parser = ParserBuilder::new(gazetteer, 0.5)
            .n_stop_words(2)
            .additional_stop_words(vec!["hello".to_string()])
            .build()
            .unwrap();

        let mut gt_stop_words: HashSet<u32> = HashSet::default();
        gt_stop_words.insert(
            parser
                .tokens_symbol_table
                .find_single_symbol("the")
                .unwrap()
                .unwrap(),
        );
        gt_stop_words.insert(
            parser
                .tokens_symbol_table
                .find_single_symbol("stones")
                .unwrap()
                .unwrap(),
        );
        gt_stop_words.insert(
            parser
                .tokens_symbol_table
                .find_single_symbol("hello")
                .unwrap()
                .unwrap(),
        );
        assert!(parser.stop_words == gt_stop_words);
        let mut gt_edge_cases: HashSet<u32> = HashSet::default();
        gt_edge_cases.insert(
            parser
                .resolved_symbol_table
                .find_single_symbol("The Stones")
                .unwrap()
                .unwrap(),
        );
        assert!(parser.edge_cases == gt_edge_cases);

        // Value starting with a stop word
        parser.set_threshold(0.6);
        let parsed = parser.run("je veux écouter les the rolling").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "the rolling".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
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
                range: 26..44,
            }]
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

        let parser = ParserBuilder::new(gazetteer, 0.0).build().unwrap();

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

        let mut parser = ParserBuilder::new(gazetteer, 0.0).build().unwrap();

        let mut parsed = parser.run("let's listen to blink 182").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "blink 182".to_string(),
                resolved_value: "Blink-182".to_string(),
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
                range: 16..21,
            }]
        );

        parser.set_threshold(0.5);
        parsed = parser.run("let's listen to blink one two").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "blink one two".to_string(),
                resolved_value: "Blink-182".to_string(),
                range: 16..29,
            }]
        );
    }

    #[test]
    fn test_parser_with_ranking() {
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
        let parser = ParserBuilder::new(gazetteer, 0.5).build().unwrap();

        // When there is a tie in terms of number of token matched, match the most popular choice
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
        let mut gazetteer = Gazetteer::new();
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });

        let parser = ParserBuilder::new(gazetteer, 0.5).build().unwrap();

        let parsed = parser
            .run("the music I want to listen to is rolling on stones")
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
        let parser = ParserBuilder::new(gazetteer, 0.5).build().unwrap();

        let parsed = parser.run("non quand est survivre").unwrap();
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

        let parser = ParserBuilder::new(gazetteer, 0.5).build().unwrap();

        let parsed = parser.run("rolling the stones").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                resolved_value: "The Rolling Stones".to_string(),
                range: 8..18,
                raw_value: "the stones".to_string(),
            }]
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

        let mut parser = ParserBuilder::new(gazetteer, 0.5).build().unwrap();

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

        parser.set_threshold(0.3);
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

        parser.set_threshold(0.6);
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

    #[test]
    fn test_repeated_words() {
        let mut gazetteer = Gazetteer::new();
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });

        let mut parser = ParserBuilder::new(gazetteer, 0.5).build().unwrap();

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

        let mut parser = ParserBuilder::new(gazetteer, 0.6).build().unwrap();

        let new_values = vec![EntityValue {
            resolved_value: "The Flying Stones".to_string(),
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
                range: 20..33,
            }]
        );

        let parsed = parser.run("je veux écouter the stones").unwrap();

        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "the stones".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
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
                range: 20..33,
            }]
        );

        let parsed = parser.run("je veux écouter the stones").unwrap();

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

        let mut parser = ParserBuilder::new(gazetteer, 0.6).build().unwrap();

        let new_values_1 = vec![EntityValue {
            resolved_value: "The Flying Stones".to_string(),
            raw_value: "the flying stones".to_string(),
        }];

        parser.inject_new_values(new_values_1, true, false).unwrap();

        // Test injection from vanilla
        let new_values_2 = vec![EntityValue {
            resolved_value: "Queens Of The Stone Age".to_string(),
            raw_value: "queens of the stone age".to_string(),
        }];

        let flying_idx = parser
            .tokens_symbol_table
            .find_single_symbol("flying")
            .unwrap()
            .unwrap();
        let stones_idx = parser
            .tokens_symbol_table
            .find_single_symbol("stones")
            .unwrap()
            .unwrap();
        let flying_stones_idx = parser
            .resolved_symbol_table
            .find_single_symbol("The Flying Stones")
            .unwrap()
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
                range: 16..36,
            }]
        );

        assert_eq!(
            parser
                .resolved_symbol_table
                .find_symbol("The Flying Stones"),
            None
        );
        assert_eq!(parser.tokens_symbol_table.find_symbol("flying"), None);
        assert!(!parser.token_to_count.contains_key(&flying_idx));
        assert!(!parser.token_to_resolved_values.contains_key(&flying_idx));
        assert!(
            !parser
                .get_resolved_values_from_token(&stones_idx)
                .unwrap()
                .contains(&flying_stones_idx)
        );
        assert!(
            !parser
                .resolved_value_to_tokens
                .contains_key(&flying_stones_idx)
        );
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

        let mut parser = ParserBuilder::new(gazetteer.clone(), 0.0)
            .n_stop_words(2)
            .additional_stop_words(vec!["hello".to_string()])
            .build()
            .unwrap();

        let parser_no_stop_words = ParserBuilder::new(gazetteer, 0.0).build().unwrap();

        let mut gt_stop_words: HashSet<u32> = HashSet::default();
        gt_stop_words.insert(
            parser
                .tokens_symbol_table
                .find_single_symbol("the")
                .unwrap()
                .unwrap(),
        );
        gt_stop_words.insert(
            parser
                .tokens_symbol_table
                .find_single_symbol("stones")
                .unwrap()
                .unwrap(),
        );
        gt_stop_words.insert(
            parser
                .tokens_symbol_table
                .find_single_symbol("hello")
                .unwrap()
                .unwrap(),
        );

        let mut gt_edge_cases: HashSet<u32> = HashSet::default();
        gt_edge_cases.insert(
            parser
                .resolved_symbol_table
                .find_single_symbol("The Stones")
                .unwrap()
                .unwrap(),
        );

        assert_eq!(parser.stop_words, gt_stop_words);
        assert_eq!(parser.edge_cases, gt_edge_cases);

        let new_values = vec![
            EntityValue {
                resolved_value: "Rolling".to_string(),
                raw_value: "rolling".to_string(),
            },
            EntityValue {
                resolved_value: "Rolling Two".to_string(),
                raw_value: "rolling two".to_string(),
            },
        ];

        parser.inject_new_values(new_values, true, false).unwrap();

        gt_stop_words.remove(
            &parser
                .tokens_symbol_table
                .find_single_symbol("stones")
                .unwrap()
                .unwrap(),
        );
        gt_stop_words.insert(
            parser
                .tokens_symbol_table
                .find_single_symbol("rolling")
                .unwrap()
                .unwrap(),
        );

        gt_edge_cases.remove(
            &parser
                .resolved_symbol_table
                .find_single_symbol("The Stones")
                .unwrap()
                .unwrap(),
        );
        gt_edge_cases.insert(
            parser
                .resolved_symbol_table
                .find_single_symbol("Rolling")
                .unwrap()
                .unwrap(),
        );

        assert_eq!(parser.stop_words, gt_stop_words);
        assert_eq!(parser.edge_cases, gt_edge_cases);

        // No stop words case
        assert!(parser_no_stop_words.stop_words.is_empty());
        assert!(parser_no_stop_words.edge_cases.is_empty());
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
        gazetteer.add(EntityValue {
            resolved_value: "6 7".to_string(),
            raw_value: "six seven".to_string(),
        });

        let parser = ParserBuilder::new(gazetteer, 0.5).build().unwrap();

        let parsed = parser
            .run("je veux écouter le black and white album")
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "black and white album".to_string(),
                resolved_value: "The Black and White Album".to_string(),
                range: 19..40,
            }]
        );

        let parsed = parser.run("one two three four").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "one two three four".to_string(),
                resolved_value: "1 2 3 4".to_string(),
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
                    range: 5..23,
                },
                ParsedValue {
                    raw_value: "five six".to_string(),
                    resolved_value: "3 4 5 6".to_string(),
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
                    range: 5..23,
                },
                ParsedValue {
                    raw_value: "six seven".to_string(),
                    resolved_value: "6 7".to_string(),
                    range: 29..38,
                },
            ]
        );
    }

    #[test]
    fn real_world_gazetteer_parser() {
        let (_, body) = CallBuilder::get().max_response(20000000).timeout_ms(60000).url("https://s3.amazonaws.com/snips/nlu-lm/test/gazetteer-entity-parser/artist_gazetteer_formatted.json").unwrap().exec().unwrap();
        let data: Vec<EntityValue> = serde_json::from_reader(&*body).unwrap();
        let gaz = Gazetteer { data };

        let n_stop_words = 50;
        let mut parser = ParserBuilder::new(gaz, 0.6)
            .n_stop_words(n_stop_words)
            .build()
            .unwrap();

        let parsed = parser.run("je veux écouter les rolling stones").unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                raw_value: "rolling stones".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
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
                range: 16..21,
            }]
        );

        let (_, body) = CallBuilder::get().max_response(20000000).timeout_ms(60000).url("https://s3.amazonaws.com/snips/nlu-lm/test/gazetteer-entity-parser/album_gazetteer_formatted.json").unwrap().exec().unwrap();
        let data: Vec<EntityValue> = serde_json::from_reader(&*body).unwrap();
        let gaz = Gazetteer { data };

        let n_stop_words = 50;
        let mut parser = ParserBuilder::new(gaz, 0.6)
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
                    range: 0..7,
                },
                ParsedValue {
                    raw_value: "dark side of the moon".to_string(),
                    resolved_value: "Dark Side of the Moon".to_string(),
                    range: 16..37,
                },
            ]
        );
    }

    #[test]
    fn test_real_word_injection() {
        // Real-world artist gazetteer
        let (_, body) = CallBuilder::get().max_response(20000000).timeout_ms(100000).url("https://s3.amazonaws.com/snips/nlu-lm/test/gazetteer-entity-parser/artist_gazetteer_formatted.json").unwrap().exec().unwrap();
        let data: Vec<EntityValue> = serde_json::from_reader(&*body).unwrap();
        let album_gaz = Gazetteer { data };

        let mut parser_for_test = ParserBuilder::new(album_gaz.clone(), 0.6)
            .n_stop_words(50)
            .build()
            .unwrap();

        let mut parser = ParserBuilder::new(album_gaz, 0.6)
            .n_stop_words(50)
            .build()
            .unwrap();

        // Get 10k values from the album gazetter to inject in the album parser
        let (_, body) = CallBuilder::get().max_response(20000000).timeout_ms(100000).url("https://s3.amazonaws.com/snips/nlu-lm/test/gazetteer-entity-parser/album_gazetteer_formatted.json").unwrap().exec().unwrap();
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
                range: 16..44,
            }]
        );

        let now = Instant::now();
        parser.inject_new_values(new_values, true, false).unwrap();
        let total_time = now.elapsed().as_secs();
        assert!(total_time < 10);
    }
}
