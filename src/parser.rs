use crate::constants::*;
use crate::data::EntityValue;
use crate::errors::*;
use crate::symbol_table::{ResolvedSymbolTable, TokenSymbolTable};
use crate::utils::{check_threshold, whitespace_tokenizer};
use failure::{format_err, ResultExt};
use fnv::{FnvHashMap as HashMap, FnvHashSet as HashSet};
use rmp_serde::{from_read, Serializer};
use serde::{Deserialize, Serialize};
use serde_json;
use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, BinaryHeap};
use std::fs;
use std::ops::Range;
use std::path::Path;

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
    // License information associated to the parser's data
    #[serde(default)]
    license_info: Option<LicenseInfo>,
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct LicenseInfo {
    pub filename: String,
    pub content: String,
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
    // List of tuples (resolved_value_idx, rank)
    alternative_resolved_values: Vec<(u32, u32)>,
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
#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct ParsedValue {
    pub resolved_value: ResolvedValue,
    pub alternatives: Vec<ResolvedValue>,
    // character-level
    pub range: Range<usize>,
    pub matched_value: String,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct ResolvedValue {
    pub resolved: String,
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
        if self.range.end <= other.range.start {
            Some(Ordering::Less)
        } else if self.range.start >= other.range.end {
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
            .add_symbol(entity_value.resolved_value);

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
        for res_val in self.resolved_symbol_table.get_all_indices() {
            self.resolved_value_to_tokens
                .entry(*res_val)
                .and_modify(|(rank, _)| *rank += entity_values.len() as u32);
        }
        for (rank, entity_value) in entity_values.into_iter().enumerate() {
            self.add_value(entity_value.clone(), rank as u32);
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
            .map(|(idx, res_values)| (idx.clone(), res_values.len()))
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
                    .or_insert_with(|| BTreeSet::new());
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

    /// Set the license info
    pub fn set_license_info<T: Into<Option<LicenseInfo>>>(&mut self, license_info: T) {
        self.license_info = license_info.into();
    }

    /// Get the set of stop words
    pub fn get_stop_words(&self) -> HashSet<String> {
        self.stop_words
            .iter()
            .flat_map(|idx| self.tokens_symbol_table.find_index(idx).cloned())
            .collect()
    }

    /// Get the set of edge cases, containing only stop words
    pub fn get_edge_cases(&self) -> HashSet<String> {
        self.edge_cases
            .iter()
            .flat_map(|idx| self.resolved_symbol_table.find_index(idx).cloned())
            .collect()
    }

    /// Parse the input string `input` and output a vec of `ParsedValue`.
    /// The `max_alternatives` defines how many alternative resolved values must be returned in
    /// addition to the top one.
    pub fn run(&self, input: &str, max_alternatives: usize) -> Result<Vec<ParsedValue>> {
        let matches_heap = self
            .find_possible_matches(input, self.threshold, max_alternatives)
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
                        .get_tokens_from_resolved_value(&res_val)
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
                self.tokens_symbol_table.remove_index(&tok_idx);
                self.token_to_resolved_values.remove(&tok_idx);
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
            // we inject new values from rank 0 to n_new_values - 1
            true => 0,
            // we inject new values from the current last rank onwards
            false => self.resolved_value_to_tokens.len() as u32,
        };

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

        if let Some(license_info) = &self.license_info {
            let license_path = folder_name.as_ref().join(&license_info.filename);
            fs::write(license_path, &license_info.content)
                .with_context(|_| format_err!("Error when writing the license"))?
        }

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
    fn get_tokens_from_resolved_value(&self, resolved_value: &u32) -> Result<&(u32, Vec<u32>)> {
        self.resolved_value_to_tokens
            .get(resolved_value)
            .ok_or_else(|| {
                format_err!(
                    "Cannot find resolved value index {} in `resolved_value_to_tokens`",
                    resolved_value
                )
            })
    }

    /// get resolved values from token
    fn get_resolved_values_from_token(&self, token: &u32) -> Result<&BTreeSet<u32>> {
        self.token_to_resolved_values.get(token).ok_or_else(|| {
            format_err!(
                "Cannot find token index {} in `token_to_resolved_values`",
                token
            )
        })
    }

    /// get the resolved values from a possible match
    fn get_resolved_value(&self, resolved_value_index: u32) -> Result<ResolvedValue> {
        let resolved = self
            .resolved_symbol_table
            .find_index(&resolved_value_index)
            .cloned()
            .ok_or_else(|| {
                format_err!("Missing key for resolved value {}", resolved_value_index)
            })?;
        let matched_value = self
            .resolved_value_to_tokens
            .get(&resolved_value_index)
            .ok_or_else(|| format_err!("Missing key for resolved value {}", resolved_value_index))?
            .1
            .iter()
            .flat_map(|token_idx| self.tokens_symbol_table.find_index(token_idx))
            .map(|token_string| token_string.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(ResolvedValue {
            resolved,
            raw_value: matched_value,
        })
    }

    /// Find and return all possible matches in a string, ordered by the order defined on
    /// `PossibleMatch`, from max to min.
    fn find_possible_matches(
        &self,
        input: &str,
        threshold: f32,
        max_alternatives: usize,
    ) -> Result<BinaryHeap<PossibleMatch>> {
        let mut partial_matches: HashMap<u32, PossibleMatch> = HashMap::default();
        let mut final_matches: Vec<PossibleMatch> = vec![];
        let mut skipped_tokens: HashMap<usize, (Range<usize>, u32)> = HashMap::default();
        for (token_idx, (range, token)) in whitespace_tokenizer(input).enumerate() {
            if let Some(value) = self.tokens_symbol_table.find_symbol(&token) {
                let res_vals_from_token = self.get_resolved_values_from_token(&value)?;
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
                            &mut partial_matches,
                            &mut final_matches,
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
                                &mut partial_matches,
                                &mut final_matches,
                                &mut skipped_tokens,
                                1.0,
                            )?;
                        }
                    }

                    // Iterate over current possible matches containing the stop word and
                    // try to grow them (but do not initiate a new possible match)
                    // Threshold depends on whether the res_val is an edge case or not
                    for (res_val, mut possible_match) in &mut partial_matches {
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
                            &mut final_matches,
                        )?;
                    }
                }
            }
        }

        // Add to the heap the possible matches that remain
        let final_matches = partial_matches
            .values()
            .filter(|possible_match| {
                if self.edge_cases.contains(&possible_match.resolved_value) {
                    possible_match.check_threshold(1.0)
                } else {
                    possible_match.check_threshold(threshold)
                }
            })
            .fold(final_matches, |mut acc, possible_match| {
                acc.push(possible_match.clone());
                acc
            });

        // Group possible matches that matched the same underlying range
        Ok(group_matches(final_matches, max_alternatives))
    }

    fn update_or_insert_possible_match(
        &self,
        value: u32,
        res_val: u32,
        token_idx: usize,
        range: Range<usize>,
        partial_matches: &mut HashMap<u32, PossibleMatch>,
        mut final_matches: &mut Vec<PossibleMatch>,
        skipped_tokens: &mut HashMap<usize, (Range<usize>, u32)>,
        threshold: f32,
    ) -> Result<()> {
        match partial_matches.entry(res_val) {
            Entry::Occupied(mut entry) => {
                self.update_previous_match(
                    entry.get_mut(),
                    token_idx,
                    value,
                    range,
                    threshold,
                    &mut final_matches,
                )?;
            }
            Entry::Vacant(entry) => {
                self.insert_new_possible_match(
                    res_val,
                    value,
                    range,
                    token_idx,
                    threshold,
                    &skipped_tokens,
                )?
                .map(|new_possible_match| {
                    entry.insert(new_possible_match);
                });
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
        ref mut final_matches: &mut Vec<PossibleMatch>,
    ) -> Result<()> {
        let (rank, otokens) =
            self.get_tokens_from_resolved_value(&possible_match.resolved_value)?;

        if token_idx == possible_match.last_token_in_input + 1 {
            // Grow the last Possible Match
            // Find the next token in the resolved value that matches the
            // input token
            for otoken_idx in (possible_match.last_token_in_resolution + 1)..otokens.len() {
                let otok = otokens[otoken_idx];
                if value == otok {
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
            final_matches.push(possible_match.clone());
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
            alternative_resolved_values: vec![],
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
        let (rank, otokens) = self.get_tokens_from_resolved_value(&res_val)?;
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
            alternative_resolved_values: vec![],
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
                alternative_resolved_values: possible_match.alternative_resolved_values,
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
                .map(|idx| *idx)
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

            parsing.push(ParsedValue {
                range: possible_match.range.clone(),
                matched_value: input
                    .chars()
                    .skip(possible_match.range.start)
                    .take(possible_match.range.len())
                    .collect(),
                resolved_value: self.get_resolved_value(possible_match.resolved_value)?,
                alternatives: possible_match
                    .alternative_resolved_values
                    .iter()
                    .map(|(idx, _)| Ok(self.get_resolved_value(*idx)?))
                    .collect::<Result<Vec<_>>>()?,
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

fn group_matches(
    final_matches: Vec<PossibleMatch>,
    max_alternatives: usize,
) -> BinaryHeap<PossibleMatch> {
    final_matches
        .iter()
        .fold(
            HashMap::<Range<usize>, BinaryHeap<&PossibleMatch>>::default(),
            |mut grouped_matches, final_match| {
                grouped_matches
                    .entry(final_match.range.clone())
                    .and_modify(|entry| entry.push(final_match))
                    .or_insert_with(|| {
                        let mut alternatives = BinaryHeap::new();
                        alternatives.push(final_match);
                        alternatives
                    });
                grouped_matches
            },
        )
        .into_iter()
        .map(|(_, mut matches)| {
            let mut best_match = matches.pop().unwrap().clone();
            while !matches.is_empty()
                && best_match.alternative_resolved_values.len() < max_alternatives
            {
                let m = matches.pop().unwrap();
                // Only add alternative with the same matching ratio
                if m.raw_value_length > best_match.raw_value_length {
                    break;
                }
                best_match
                    .alternative_resolved_values
                    .push((m.resolved_value, m.rank));
            }
            best_match
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::EntityValue;
    use crate::data::Gazetteer;
    use crate::gazetteer;
    use crate::parser_builder::ParserBuilder;
    use failure::ResultExt;
    use tempfile::tempdir;

    fn get_license_info() -> LicenseInfo {
        let license_content = "Some content here".to_string();
        let license_filename = "LICENSE".to_string();
        let license_info = LicenseInfo {
            filename: license_filename,
            content: license_content,
        };
        license_info
    }

    #[test]
    fn test_serialization_deserialization() {
        let tdir = tempdir().unwrap();
        let gazetteer = gazetteer!(
            ("the flying stones", "The Flying Stones"),
            ("the rolling stones", "The Rolling Stones"),
            ("the stones", "The Rolling Stones"),
        );

        let license_info = get_license_info();

        let parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .n_stop_words(2)
            .additional_stop_words(vec!["hello".to_string()])
            .license_info(license_info)
            .build()
            .unwrap();

        let serialized_parser_path = tdir.as_ref().join("parser");
        let license_path = serialized_parser_path.join("LICENSE");
        parser.dump(serialized_parser_path).unwrap();

        assert!(license_path.exists());

        let expected_content = "Some content here".to_string();
        let content = fs::read_to_string(license_path).unwrap();
        assert_eq!(content, expected_content);

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
        let gazetteer = gazetteer!(
            ("the flying stones", "The Flying Stones"),
            ("the rolling stones", "The Rolling Stones"),
            ("the stones rolling", "The Stones Rolling"),
            ("the stones", "The Stones"),
        );

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
        let parsed = parser.run("je veux écouter les the rolling", 5).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "the rolling".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "The Rolling Stones".to_string(),
                    raw_value: "the rolling stones".to_string()
                },
                alternatives: vec![ResolvedValue {
                    resolved: "The Stones Rolling".to_string(),
                    raw_value: "the stones rolling".to_string()
                }],
                range: 20..31,
            }]
        );

        // Value starting with a stop word and ending with one
        parser.set_threshold(1.0);
        let parsed = parser
            .run("je veux écouter les the rolling stones", 5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "the rolling stones".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "The Rolling Stones".to_string(),
                    raw_value: "the rolling stones".to_string(),
                },
                alternatives: vec![],
                range: 20..38,
            }]
        );

        // Value starting with two stop words
        parser.set_threshold(1.0);
        let parsed = parser
            .run("je veux écouter les the stones rolling", 5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "the stones rolling".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "The Stones Rolling".to_string(),
                    raw_value: "the stones rolling".to_string(),
                },
                alternatives: vec![],
                range: 20..38,
            }]
        );

        // Edge case
        parser.set_threshold(1.0);
        let parsed = parser.run("je veux écouter les the stones", 5).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "the stones".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "The Stones".to_string(),
                    raw_value: "the stones".to_string(),
                },
                alternatives: vec![],
                range: 20..30,
            }]
        );

        // Edge case should not match if not present in full
        parser.set_threshold(0.5);
        let parsed = parser.run("je veux écouter les the", 5).unwrap();
        assert_eq!(parsed, vec![]);

        // Sentence containing an additional stop word which is absent from the gazetteer
        let parsed = parser
            .run("hello I want to listen to the rolling stones", 5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "the rolling stones".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "The Rolling Stones".to_string(),
                    raw_value: "the rolling stones".to_string(),
                },
                alternatives: vec![],
                range: 26..44,
            }]
        );

        // Multiple stop words at the beginning of a value
        let parsed = parser
            .run("hello I want to listen to the the rolling stones", 5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "the rolling stones".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "The Rolling Stones".to_string(),
                    raw_value: "the rolling stones".to_string(),
                },
                alternatives: vec![],
                range: 30..48,
            }]
        );
    }

    #[test]
    fn test_parser_base() {
        let gazetteer = gazetteer!(
            ("the flying stones", "The Flying Stones"),
            ("the rolling stones", "The Rolling Stones"),
            ("blink one eight two", "Blink-182"),
            ("je suis animal", "Je Suis Animal"),
        );

        let parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.0)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let mut parsed = parser
            .run("je veux écouter les rolling stones", 5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    matched_value: "je".to_string(),
                    resolved_value: ResolvedValue {
                        resolved: "Je Suis Animal".to_string(),
                        raw_value: "je suis animal".to_string(),
                    },
                    alternatives: vec![],
                    range: 0..2,
                },
                ParsedValue {
                    matched_value: "rolling stones".to_string(),
                    resolved_value: ResolvedValue {
                        resolved: "The Rolling Stones".to_string(),
                        raw_value: "the rolling stones".to_string(),
                    },
                    alternatives: vec![],
                    range: 20..34,
                },
            ]
        );

        parsed = parser
            .run("je veux ecouter les \t rolling stones", 5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    matched_value: "je".to_string(),
                    resolved_value: ResolvedValue {
                        resolved: "Je Suis Animal".to_string(),
                        raw_value: "je suis animal".to_string(),
                    },
                    alternatives: vec![],
                    range: 0..2,
                },
                ParsedValue {
                    matched_value: "rolling stones".to_string(),
                    resolved_value: ResolvedValue {
                        resolved: "The Rolling Stones".to_string(),
                        raw_value: "the rolling stones".to_string(),
                    },
                    alternatives: vec![],
                    range: 22..36,
                },
            ]
        );

        parsed = parser
            .run("i want to listen to rolling stones and blink eight", 5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    matched_value: "rolling stones".to_string(),
                    resolved_value: ResolvedValue {
                        resolved: "The Rolling Stones".to_string(),
                        raw_value: "the rolling stones".to_string(),
                    },
                    alternatives: vec![],
                    range: 20..34,
                },
                ParsedValue {
                    matched_value: "blink eight".to_string(),
                    resolved_value: ResolvedValue {
                        resolved: "Blink-182".to_string(),
                        raw_value: "blink one eight two".to_string(),
                    },
                    alternatives: vec![],
                    range: 39..50,
                },
            ]
        );

        parsed = parser.run("joue moi quelque chose", 5).unwrap();
        assert_eq!(parsed, vec![]);
    }

    #[test]
    fn test_parser_multiple_raw_values() {
        let gazetteer = gazetteer!(
            ("blink one eight two", "Blink-182"),
            ("blink 182", "Blink-182"),
            ("blink", "Blink-182"),
        );

        let mut parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.0)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let mut parsed = parser.run("let's listen to blink 182", 5).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "blink 182".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "Blink-182".to_string(),
                    raw_value: "blink 182".to_string(),
                },
                alternatives: vec![],
                range: 16..25,
            }]
        );

        parser.set_threshold(0.5);
        parsed = parser.run("let's listen to blink", 5).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "blink".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "Blink-182".to_string(),
                    raw_value: "blink".to_string(),
                },
                alternatives: vec![],
                range: 16..21,
            }]
        );

        parser.set_threshold(0.5);
        parsed = parser.run("let's listen to one eight two", 5).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "one eight two".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "Blink-182".to_string(),
                    raw_value: "blink one eight two".to_string(),
                },
                alternatives: vec![],
                range: 16..29,
            }]
        );
    }

    #[test]
    fn test_parser_with_ranking() {
        let gazetteer = gazetteer!(
            ("jacques brel", "Jacques Brel"),
            ("the rolling stones", "The Rolling Stones"),
            ("the flying stones", "The Flying Stones"),
            ("daniel brel", "Daniel Brel"),
            ("jacques", "Jacques"),
        );

        let parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        // When there is a tie in terms of number of token matched, match the most popular choice
        let parsed = parser.run("je veux écouter the stones", 5).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "the stones".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "The Rolling Stones".to_string(),
                    raw_value: "the rolling stones".to_string(),
                },
                alternatives: vec![ResolvedValue {
                    resolved: "The Flying Stones".to_string(),
                    raw_value: "the flying stones".to_string(),
                }],
                range: 16..26,
            }]
        );

        // Resolve to the value with more words matching regardless of popularity
        let parsed = parser.run("je veux écouter the flying stones", 5).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "the flying stones".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "The Flying Stones".to_string(),
                    raw_value: "the flying stones".to_string(),
                },
                alternatives: vec![],
                range: 16..33,
            }]
        );

        // Resolve to the value with the best matching ratio
        let parsed = parser.run("je veux écouter jacques", 5).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "jacques".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "Jacques".to_string(),
                    raw_value: "jacques".to_string(),
                },
                alternatives: vec![],
                range: 16..23,
            }]
        );
    }

    #[test]
    fn test_preprend_values() {
        let gazetteer = gazetteer!(
            ("jacques brel", "Jacques Brel"),
            ("the rolling stones", "The Rolling Stones"),
        );
        let mut parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let input = "je veux écouter brel";

        let parsed = parser.run(input, 5).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "brel".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "Jacques Brel".to_string(),
                    raw_value: "jacques brel".to_string(),
                },
                alternatives: vec![],
                range: 16..20,
            }]
        );

        let values_to_prepend = vec![
            EntityValue {
                resolved_value: "Daniel Brel".to_string(),
                raw_value: "daniel brel".to_string(),
            },
            EntityValue {
                resolved_value: "Eric Brel".to_string(),
                raw_value: "eric brel".to_string(),
            },
        ];

        parser.prepend_values(values_to_prepend);

        let parsed = parser.run(input, 5).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "brel".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "Daniel Brel".to_string(),
                    raw_value: "daniel brel".to_string(),
                },
                alternatives: vec![
                    ResolvedValue {
                        resolved: "Eric Brel".to_string(),
                        raw_value: "eric brel".to_string(),
                    },
                    ResolvedValue {
                        resolved: "Jacques Brel".to_string(),
                        raw_value: "jacques brel".to_string(),
                    }
                ],
                range: 16..20,
            }]
        );
    }

    #[test]
    fn test_parser_with_restart() {
        let gazetteer = gazetteer!(("the rolling stones", "The Rolling Stones"),);
        let parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let parsed = parser
            .run("the music I want to listen to is rolling on stones", 5)
            .unwrap();
        assert_eq!(parsed, vec![]);
    }

    #[test]
    fn test_parser_with_unicode_whitespace() {
        let gazetteer = gazetteer!(("quand est -ce", "Quand est-ce ?"),);
        let parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let parsed = parser.run("non quand est survivre", 5).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                resolved_value: ResolvedValue {
                    resolved: "Quand est-ce ?".to_string(),
                    raw_value: "quand est -ce".to_string(),
                },
                range: 4..13,
                matched_value: "quand est".to_string(),
                alternatives: vec![],
            }]
        )
    }

    #[test]
    fn test_parser_with_mixed_ordered_entity() {
        let gazetteer = gazetteer!(("the rolling stones", "The Rolling Stones"),);
        let parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let parsed = parser.run("rolling the stones", 5).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                resolved_value: ResolvedValue {
                    resolved: "The Rolling Stones".to_string(),
                    raw_value: "the rolling stones".to_string(),
                },
                range: 8..18,
                matched_value: "the stones".to_string(),
                alternatives: vec![],
            }]
        );
    }

    #[test]
    fn test_parser_with_threshold() {
        let gazetteer = gazetteer!(
            ("the flying stones", "The Flying Stones"),
            ("the rolling stones", "The Rolling Stones"),
            ("blink one eight two", "Blink-182"),
            ("je suis animal", "Je Suis Animal"),
            ("les enfoirés", "Les Enfoirés"),
        );

        let mut parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let parsed = parser
            .run("je veux écouter les rolling stones", 5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    resolved_value: ResolvedValue {
                        resolved: "Les Enfoirés".to_string(),
                        raw_value: "les enfoirés".to_string(),
                    },
                    range: 16..19,
                    matched_value: "les".to_string(),
                    alternatives: vec![],
                },
                ParsedValue {
                    matched_value: "rolling stones".to_string(),
                    resolved_value: ResolvedValue {
                        resolved: "The Rolling Stones".to_string(),
                        raw_value: "the rolling stones".to_string(),
                    },
                    alternatives: vec![],
                    range: 20..34,
                },
            ]
        );

        parser.set_threshold(0.3);
        let parsed = parser
            .run("je veux écouter les rolling stones", 5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    matched_value: "je".to_string(),
                    resolved_value: ResolvedValue {
                        resolved: "Je Suis Animal".to_string(),
                        raw_value: "je suis animal".to_string(),
                    },
                    alternatives: vec![],
                    range: 0..2,
                },
                ParsedValue {
                    resolved_value: ResolvedValue {
                        resolved: "Les Enfoirés".to_string(),
                        raw_value: "les enfoirés".to_string(),
                    },
                    alternatives: vec![],
                    range: 16..19,
                    matched_value: "les".to_string(),
                },
                ParsedValue {
                    matched_value: "rolling stones".to_string(),
                    resolved_value: ResolvedValue {
                        resolved: "The Rolling Stones".to_string(),
                        raw_value: "the rolling stones".to_string(),
                    },
                    alternatives: vec![],
                    range: 20..34,
                },
            ]
        );

        parser.set_threshold(0.6);
        let parsed = parser
            .run("je veux écouter les rolling stones", 5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "rolling stones".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "The Rolling Stones".to_string(),
                    raw_value: "the rolling stones".to_string(),
                },
                alternatives: vec![],
                range: 20..34,
            }]
        );
    }

    #[test]
    fn test_repeated_words() {
        let gazetteer = gazetteer!(("the rolling stones", "The Rolling Stones"),);
        let mut parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let parsed = parser.run("the the the", 5).unwrap();
        assert_eq!(parsed, vec![]);

        parser.set_threshold(1.0);
        let parsed = parser
            .run("the the the rolling stones stones stones stones", 5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "the rolling stones".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "The Rolling Stones".to_string(),
                    raw_value: "the rolling stones".to_string(),
                },
                alternatives: vec![],
                range: 8..26,
            }]
        );
    }

    #[test]
    fn test_injection_ranking() {
        let gazetteer = gazetteer!(("the rolling stones", "The Rolling Stones"),);
        let mut parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.6)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let new_values = vec![EntityValue {
            resolved_value: "The Flying Stones".to_string(),
            raw_value: "the flying stones".to_string(),
        }];

        // Test with preprend set to false
        parser
            .inject_new_values(new_values.clone(), false, false)
            .unwrap();
        let parsed = parser.run("je veux écouter les flying stones", 5).unwrap();

        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "flying stones".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "The Flying Stones".to_string(),
                    raw_value: "the flying stones".to_string(),
                },
                alternatives: vec![],
                range: 20..33,
            }]
        );

        let parsed = parser.run("je veux écouter the stones", 5).unwrap();

        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "the stones".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "The Rolling Stones".to_string(),
                    raw_value: "the rolling stones".to_string(),
                },
                alternatives: vec![ResolvedValue {
                    resolved: "The Flying Stones".to_string(),
                    raw_value: "the flying stones".to_string(),
                }],
                range: 16..26,
            }]
        );

        // Test with preprend set to true
        parser.inject_new_values(new_values, true, true).unwrap();

        let parsed = parser.run("je veux écouter les flying stones", 5).unwrap();

        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "flying stones".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "The Flying Stones".to_string(),
                    raw_value: "the flying stones".to_string(),
                },
                alternatives: vec![],
                range: 20..33,
            }]
        );

        let parsed = parser.run("je veux écouter the stones", 5).unwrap();

        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "the stones".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "The Flying Stones".to_string(),
                    raw_value: "the flying stones".to_string(),
                },
                alternatives: vec![ResolvedValue {
                    resolved: "The Rolling Stones".to_string(),
                    raw_value: "the rolling stones".to_string(),
                }],
                range: 16..26,
            }]
        );
    }

    #[test]
    fn test_injection_from_vanilla() {
        let gazetteer = gazetteer!(("the rolling stones", "The Rolling Stones"),);
        let mut parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.6)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

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

        let flying_idx = *parser.tokens_symbol_table.find_symbol("flying").unwrap();
        let stones_idx = *parser.tokens_symbol_table.find_symbol("stones").unwrap();
        let flying_stones_idx = *parser
            .resolved_symbol_table
            .find_symbol("The Flying Stones")
            .first()
            .unwrap();
        parser.inject_new_values(new_values_2, true, true).unwrap();

        let parsed = parser.run("je veux écouter les flying stones", 5).unwrap();
        assert_eq!(parsed, vec![]);

        let parsed = parser
            .run("je veux écouter queens the stone age", 5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "queens the stone age".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "Queens Of The Stone Age".to_string(),
                    raw_value: "queens of the stone age".to_string(),
                },
                alternatives: vec![],
                range: 16..36,
            }]
        );

        assert!(parser
            .resolved_symbol_table
            .find_symbol("The Flying Stones")
            .is_empty());
        assert!(parser.tokens_symbol_table.find_symbol("flying").is_none());
        assert!(!parser.token_to_resolved_values.contains_key(&flying_idx));
        assert!(!parser
            .token_to_resolved_values
            .get(&stones_idx)
            .unwrap()
            .contains(&flying_stones_idx));
        assert!(!parser
            .resolved_value_to_tokens
            .contains_key(&flying_stones_idx));
    }

    #[test]
    fn test_injection_stop_words() {
        let gazetteer = gazetteer!(
            ("the rolling stones", "The Rolling Stones"),
            ("the stones", "The Stones"),
        );
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
                raw_value: "rolling".to_string(),
            },
            EntityValue {
                resolved_value: "Rolling Two".to_string(),
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
        let gazetteer = gazetteer!(
            ("black and white", "Black And White"),
            ("album", "Album"),
            ("the black and white album", "The Black and White Album"),
            ("one two three four", "1 2 3 4"),
            ("three four five", "3 4 5"),
            ("five six", "5 6"),
        );

        let parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.7)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let parsed = parser
            .run("je veux écouter le black and white album", 5)
            .unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "black and white album".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "The Black and White Album".to_string(),
                    raw_value: "the black and white album".to_string(),
                },
                alternatives: vec![],
                range: 19..40,
            }]
        );

        let parsed = parser.run("zero one two three four five", 5).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "one two three four".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "1 2 3 4".to_string(),
                    raw_value: "one two three four".to_string(),
                },
                alternatives: vec![],
                range: 5..23,
            },]
        );

        let parsed = parser.run("zero one two three four five six", 5).unwrap();
        assert_eq!(
            parsed,
            vec![
                ParsedValue {
                    matched_value: "one two three four".to_string(),
                    resolved_value: ResolvedValue {
                        resolved: "1 2 3 4".to_string(),
                        raw_value: "one two three four".to_string(),
                    },
                    alternatives: vec![],
                    range: 5..23,
                },
                ParsedValue {
                    matched_value: "five six".to_string(),
                    resolved_value: ResolvedValue {
                        resolved: "5 6".to_string(),
                        raw_value: "five six".to_string(),
                    },
                    alternatives: vec![],
                    range: 24..32,
                },
            ]
        );
    }

    #[test]
    fn test_alternative_matches() {
        let gazetteer = gazetteer!(
            ("space invader", "Space Invader"),
            ("invader on mars", "Invader on Mars"),
            ("invader attack", "Invader Attack"),
        );

        let parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let parsed = parser.run("I want to play to invader", 5).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "invader".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "Space Invader".to_string(),
                    raw_value: "space invader".to_string(),
                },
                alternatives: vec![ResolvedValue {
                    resolved: "Invader Attack".to_string(),
                    raw_value: "invader attack".to_string(),
                }],
                range: 18..25,
            }]
        );
    }

    #[test]
    fn test_max_alternative_matches() {
        let gazetteer = gazetteer!(
            ("space invader", "Space Invader"),
            ("invader war", "Invader War"),
            ("invader attack", "Invader Attack"),
            ("invader life", "Invader Life"),
        );

        let parser = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer)
            .build()
            .unwrap();

        let parsed = parser.run("I want to play to invader", 2).unwrap();
        assert_eq!(
            parsed,
            vec![ParsedValue {
                matched_value: "invader".to_string(),
                resolved_value: ResolvedValue {
                    resolved: "Space Invader".to_string(),
                    raw_value: "space invader".to_string(),
                },
                alternatives: vec![
                    ResolvedValue {
                        resolved: "Invader War".to_string(),
                        raw_value: "invader war".to_string(),
                    },
                    ResolvedValue {
                        resolved: "Invader Attack".to_string(),
                        raw_value: "invader attack".to_string(),
                    }
                ],
                range: 18..25,
            }]
        );
    }
}
