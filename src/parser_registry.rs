use std::collections::{BTreeSet, HashSet};

use serde::{Deserialize, Serialize};

use crate::data::{RegisteredEntityValue, ResolvedValue, TokenizedEntityValue};
use crate::symbol_table::{ResolvedSymbolTable, TokenSymbolTable};

type Rank = u32;

#[derive(PartialEq, Debug, Serialize, Deserialize, Default)]
pub struct ParserRegistry {
    /// Symbol table for the raw tokens
    tokens_symbol_table: TokenSymbolTable,
    /// Symbol table for the resolved values
    /// The latter differs from the first one in that it can contain the same resolved value
    /// multiple times (to allow for multiple raw values corresponding to the same resolved value)
    resolved_symbol_table: ResolvedSymbolTable,
    /// Maps each token to set of resolved values containing it
    token_to_resolved_values: Vec<BTreeSet<u32>>,
    /// Maps resolved value to a tuple (rank, tokens)
    resolved_value_to_tokens: Vec<(Rank, Vec<u32>)>,
    /// Number of stop words to extract from the entity data
    n_stop_words: usize,
    /// External list of stop words
    additional_stop_words: Vec<u32>,
    /// Set of all stop words
    stop_words: BTreeSet<u32>,
    /// Values composed only of stop words
    edge_cases: BTreeSet<u32>,
    /// Keeps track of injected resolved values
    injected_values: BTreeSet<u32>,
}

impl ParserRegistry {
    /// Adds a single entity value, along with its rank, to the parser registry and returns
    /// the corresponding resolved value index or None if the entity value is empty.
    /// The ranks of the other entity values will not be changed
    pub fn add_value(&mut self, value: TokenizedEntityValue, rank: Rank) -> Option<u32> {
        if value.tokens.is_empty() {
            return None;
        }

        // We force add the new resolved value: even if it is already present in the symbol table
        // we duplicate it to allow several raw values to map to it
        let res_value_idx = self.resolved_symbol_table.add_symbol(value.resolved_value);
        for token in value.tokens {
            let token_idx = self.tokens_symbol_table.add_symbol(token);

            if token_idx as usize >= self.token_to_resolved_values.len() {
                self.token_to_resolved_values
                    .push(vec![res_value_idx].into_iter().collect());
            } else {
                self.token_to_resolved_values[token_idx as usize].insert(res_value_idx);
            }

            if res_value_idx as usize >= self.resolved_value_to_tokens.len() {
                self.resolved_value_to_tokens.push((rank, vec![token_idx]));
            } else {
                self.resolved_value_to_tokens[res_value_idx as usize]
                    .1
                    .push(token_idx);
            }
        }
        Some(res_value_idx)
    }

    /// Prepends a list of entity values to the parser and update the ranks accordingly.
    /// Returns the corresponding list of resolved value indices.
    pub fn prepend_values(&mut self, entity_values: Vec<TokenizedEntityValue>) -> Vec<u32> {
        let nb_inserted_values = entity_values.len() as u32;
        // update rank of previous values
        for res_val in 0..self.resolved_value_to_tokens.len() {
            self.resolved_value_to_tokens[res_val].0 += nb_inserted_values;
        }
        let res_values_indices = entity_values
            .into_iter()
            .enumerate()
            .flat_map(|(rank, entity_value)| self.add_value(entity_value.clone(), rank as Rank))
            .collect();

        // Update the stop words and edge cases
        self.set_top_stop_words(self.n_stop_words);
        res_values_indices
    }

    /// Retrieves the index used to identify the token in the symbol table
    pub fn get_token_idx(&self, symbol: &str) -> Option<&u32> {
        self.tokens_symbol_table.find_symbol(symbol)
    }

    /// Retrieves the resolved values which underlying tokens contain the provided token
    pub fn get_resolved_values(&self, token_idx: u32) -> &BTreeSet<u32> {
        &self.token_to_resolved_values[token_idx as usize]
    }

    /// Retrieves the rank of the resolved value along with the underlying tokens
    pub fn get_tokens(&self, resolved_value_idx: u32) -> &(Rank, Vec<u32>) {
        &self.resolved_value_to_tokens[resolved_value_idx as usize]
    }

    /// Checks if the provided token index corresponds to a stop word
    pub fn is_stop_word(&self, token_idx: u32) -> bool {
        self.stop_words.contains(&token_idx)
    }

    /// Checks if the provided resolved value index corresponds to an edge case
    pub fn is_edge_case(&self, resolved_value_idx: u32) -> bool {
        self.edge_cases.contains(&resolved_value_idx)
    }

    /// Updates an internal set of stop words and corresponding edge cases.
    /// The set of stop words is made of the `n_stop_words` most frequent raw tokens in the
    /// gazetteer used to generate the parser. An optional `additional_stop_words` vector of
    /// strings can be added to the stop words. The edge cases are defined to the be the resolved
    /// values whose raw value is composed only of stop words. There are examined separately
    /// during parsing, and will match if and only if they are present verbatim in the input
    /// string.
    pub fn set_stop_words<T>(&mut self, n_stop_words: usize, additional_stop_words: T)
    where
        T: Into<Option<Vec<String>>>,
    {
        self.additional_stop_words = additional_stop_words
            .into()
            .map(|stop_words| {
                stop_words
                    .into_iter()
                    .map(|stop_word| {
                        let tok_idx = self.tokens_symbol_table.add_symbol(stop_word);
                        if tok_idx as usize >= self.token_to_resolved_values.len() {
                            self.token_to_resolved_values.push(BTreeSet::default())
                        }
                        tok_idx
                    })
                    .collect()
            })
            .unwrap_or_else(Vec::new);

        self.set_top_stop_words(n_stop_words);
    }

    fn set_top_stop_words(&mut self, nb_stop_words: usize) {
        // Update the set of stop words with the most frequent words in the gazetteer
        let mut tokens_with_counts = self
            .token_to_resolved_values
            .iter()
            .enumerate()
            .map(|(idx, res_values)| (idx as u32, res_values.len()))
            .collect::<Vec<_>>();

        tokens_with_counts.sort_by_key(|&(_, count)| -(count as i32));
        self.n_stop_words = nb_stop_words;
        self.stop_words = tokens_with_counts
            .into_iter()
            .take(nb_stop_words)
            .map(|(idx, _)| idx)
            .chain(self.additional_stop_words.clone())
            .collect();

        // Update the set of edge_cases. i.e. resolved values that only contain stop words
        self.edge_cases = self
            .resolved_value_to_tokens
            .iter()
            .enumerate()
            .filter(|(_, (_, tokens))| tokens.iter().all(|token| self.stop_words.contains(token)))
            .map(|(res_val, _)| res_val as u32)
            .collect();
    }

    /// Gets the set of edge cases indices
    pub fn get_edge_cases_indices(&self) -> &BTreeSet<u32> {
        &self.edge_cases
    }

    /// Retrieves the resolved value from its index
    pub fn get_resolved_value(&self, resolved_value_index: u32) -> ResolvedValue {
        let resolved = self
            .resolved_symbol_table
            .find_index(&resolved_value_index)
            .cloned()
            .unwrap();
        let matched_value = self.resolved_value_to_tokens[resolved_value_index as usize]
            .1
            .iter()
            .map(|token_idx| self.tokens_symbol_table.find_index(token_idx).unwrap())
            .map(|token_string| token_string.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        ResolvedValue {
            resolved,
            raw_value: matched_value,
        }
    }

    /// Add new values to an already trained Parser. This function is used for entity injection.
    /// It takes as arguments a vector of EntityValue's to inject, and a boolean indicating
    /// whether the new values should be prepended to the already existing values (`prepend=true`)
    /// or appended (`prepend=false`). Setting `from_vanilla` to true allows to remove all
    /// previously injected values before adding the new ones.
    pub fn inject_new_values(
        self,
        new_values: Vec<TokenizedEntityValue>,
        prepend: bool,
        from_vanilla: bool,
    ) -> Self {
        let mut gazetteer: Vec<RegisteredEntityValue> = Vec::new();
        let base_gazetteer = self.get_entity_values(!from_vanilla);
        let nb_base_values = base_gazetteer.len();
        let cleaned_new_values: Vec<_> = new_values
            .into_iter()
            .filter(|v| !v.tokens.is_empty())
            .collect();
        let nb_injected = cleaned_new_values.len();
        if prepend {
            gazetteer.extend(
                cleaned_new_values
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| v.into_registered(true, i as u32)),
            );

            gazetteer.extend(base_gazetteer.into_iter().map(|v| {
                let new_rank = v.rank + (nb_injected as u32);
                v.update_rank(new_rank)
            }));
        } else {
            gazetteer.extend(base_gazetteer);
            gazetteer.extend(
                cleaned_new_values
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| v.into_registered(true, i as u32 + nb_base_values as u32)),
            );
        }
        let n_stop_words = self.n_stop_words;
        let additional_stop_words = self.get_additional_stop_words();
        let mut registry = ParserRegistry::default();
        if !from_vanilla {
            registry.injected_values = self.injected_values.clone();
        };
        for (rank, entity_value) in gazetteer.into_iter().enumerate() {
            let is_injected = entity_value.is_injected;
            let tokenized_value = entity_value.into_tokenized();
            if let Some(idx) = registry.add_value(tokenized_value, rank as u32) {
                if is_injected {
                    registry.injected_values.insert(idx);
                }
            }
        }
        registry.set_stop_words(
            n_stop_words,
            additional_stop_words.into_iter().collect::<Vec<_>>(),
        );
        registry
    }

    /// Restore the underlying entity values containing both their rankings and a boolean
    /// indicating if they were injected or not.
    /// The rankings are used to sort the resulting list.
    fn get_entity_values(&self, include_injected_values: bool) -> Vec<RegisteredEntityValue> {
        let mut entity_values: Vec<RegisteredEntityValue> = self
            .resolved_symbol_table
            .clone()
            .into_iter()
            .enumerate()
            .filter_map(|(res_value_idx, resolved_value)| {
                let is_injected = self.injected_values.contains(&(res_value_idx as u32));
                if !include_injected_values && is_injected {
                    return None;
                };
                let (rank, tokens_indices) = &self.resolved_value_to_tokens[res_value_idx];
                let tokens = tokens_indices
                    .iter()
                    .map(|token_idx| {
                        self.tokens_symbol_table
                            .find_index(token_idx)
                            .unwrap()
                            .clone()
                    })
                    .collect::<Vec<_>>();
                Some(RegisteredEntityValue {
                    tokens,
                    resolved_value,
                    is_injected,
                    rank: *rank,
                })
            })
            .collect();
        entity_values.sort_by_key(|v| v.rank);
        entity_values
    }

    /// Gets the set of stop words
    pub fn get_stop_words(&self) -> HashSet<String> {
        self.stop_words
            .iter()
            .map(|idx| self.tokens_symbol_table.find_index(idx).cloned().unwrap())
            .collect()
    }

    /// Gets the set of additional words
    pub fn get_additional_stop_words(&self) -> HashSet<String> {
        self.additional_stop_words
            .iter()
            .map(|idx| self.tokens_symbol_table.find_index(idx).cloned().unwrap())
            .collect()
    }

    /// Gets the set of edge cases, containing only stop words
    pub fn get_edge_cases(&self) -> HashSet<String> {
        self.edge_cases
            .iter()
            .map(|idx| self.resolved_symbol_table.find_index(idx).cloned().unwrap())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stop_words_and_edge_cases() {
        // Given
        let mut registry = ParserRegistry::default();
        registry.add_value(
            TokenizedEntityValue::new("The Flying Stones", vec!["the", "flying", "stones"]),
            0,
        );
        registry.add_value(
            TokenizedEntityValue::new("The Rolling Stones", vec!["the", "rolling", "stones"]),
            1,
        );
        registry.add_value(
            TokenizedEntityValue::new("The Stones Rolling", vec!["the", "stones", "rolling"]),
            2,
        );
        registry.add_value(
            TokenizedEntityValue::new("The Stones", vec!["the", "stones"]),
            3,
        );

        // When
        registry.set_stop_words(2, vec!["hello".to_string()]);

        // Then
        let expected_stop_words: HashSet<String> =
            vec!["the".to_string(), "stones".to_string(), "hello".to_string()]
                .into_iter()
                .collect();
        let expected_edge_cases: HashSet<String> =
            vec!["The Stones".to_string()].into_iter().collect();
        assert_eq!(expected_stop_words, registry.get_stop_words());
        assert_eq!(expected_edge_cases, registry.get_edge_cases());
    }

    #[test]
    fn test_add_value() {
        // Given
        let mut registry = ParserRegistry::default();

        let value1 = TokenizedEntityValue::new("Daft Punk", vec!["daft", "punk"]);
        let value2 = TokenizedEntityValue::new("Blink 182", vec!["blink", "one", "eight", "two"]);
        let idx1 = registry.add_value(value1, 0);
        let idx2 = registry.add_value(value2, 1);

        // When
        let retrieved_value1 = idx1.map(|idx| registry.get_resolved_value(idx));
        let retrieved_value2 = idx2.map(|idx| registry.get_resolved_value(idx));

        // Then
        let expected_retrieved_value1 = Some(ResolvedValue {
            resolved: "Daft Punk".to_string(),
            raw_value: "daft punk".to_string(),
        });
        let expected_retrieved_value2 = Some(ResolvedValue {
            resolved: "Blink 182".to_string(),
            raw_value: "blink one eight two".to_string(),
        });
        assert_eq!(expected_retrieved_value1, retrieved_value1);
        assert_eq!(expected_retrieved_value2, retrieved_value2);
    }

    #[test]
    fn test_prepend_values() {
        // Given
        let mut registry = ParserRegistry::default();

        let value = TokenizedEntityValue::new("Daft Punk", vec!["daft", "punk"]);
        let idx = registry.add_value(value, 0).unwrap();

        // When
        let prepended_value1 = TokenizedEntityValue::new("Blink", vec!["blink"]);
        let prepended_value2 = TokenizedEntityValue::new("Metronomy", vec!["metronomy"]);
        let prepended_indices = registry.prepend_values(vec![prepended_value1, prepended_value2]);

        // Then
        let value_rank = registry.get_tokens(idx).0;
        let prepended_ranks: Vec<u32> = prepended_indices
            .into_iter()
            .map(|i| registry.get_tokens(i).0)
            .collect();

        assert_eq!(2, value_rank);
        assert_eq!(vec![0, 1], prepended_ranks);
    }

    #[test]
    fn test_reconstruct_gazetteer() {
        // Given
        let mut registry = ParserRegistry::default();
        registry.add_value(
            TokenizedEntityValue::new("Daft Punk", vec!["daft", "punk"]),
            0,
        );
        registry.add_value(TokenizedEntityValue::new("Metronomy", vec!["metronomy"]), 2);
        registry.add_value(
            TokenizedEntityValue::new("Pink Floyd", vec!["pink", "floyd"]),
            1,
        );

        // When
        let gazetteer = registry.get_entity_values(true);

        // Then
        let expected_gazetteer = vec![
            RegisteredEntityValue::new("Daft Punk", vec!["daft", "punk"], false, 0),
            RegisteredEntityValue::new("Pink Floyd", vec!["pink", "floyd"], false, 1),
            RegisteredEntityValue::new("Metronomy", vec!["metronomy"], false, 2),
        ];
        assert_eq!(expected_gazetteer, gazetteer);
    }

    #[test]
    fn test_should_inject_values() {
        // Given
        let mut registry = ParserRegistry::default();
        registry.add_value(
            TokenizedEntityValue::new("Daft Punk", vec!["daft", "punk"]),
            0,
        );
        registry.add_value(TokenizedEntityValue::new("Metronomy", vec!["metronomy"]), 2);

        // When
        registry = registry.inject_new_values(
            vec![
                TokenizedEntityValue::new("Pink Floyd", vec!["pink", "floyd"]),
                TokenizedEntityValue::new("Blink", vec!["blink"]),
            ],
            true,
            true,
        );

        // Then
        let resulting_gazetteer = registry.get_entity_values(true);
        let expected_gazetteer = vec![
            RegisteredEntityValue::new("Pink Floyd", vec!["pink", "floyd"], true, 0),
            RegisteredEntityValue::new("Blink", vec!["blink"], true, 1),
            RegisteredEntityValue::new("Daft Punk", vec!["daft", "punk"], false, 2),
            RegisteredEntityValue::new("Metronomy", vec!["metronomy"], false, 3),
        ];
        assert_eq!(expected_gazetteer, resulting_gazetteer);
    }

    #[test]
    fn test_should_inject_values_multiple_times() {
        // Given
        let mut registry = ParserRegistry::default();
        registry.add_value(
            TokenizedEntityValue::new("Daft Punk", vec!["daft", "punk"]),
            0,
        );
        registry.add_value(TokenizedEntityValue::new("Metronomy", vec!["metronomy"]), 2);

        // When
        registry = registry.inject_new_values(
            vec![
                TokenizedEntityValue::new("Pink Floyd", vec!["pink", "floyd"]),
                TokenizedEntityValue::new("Blink", vec!["blink"]),
            ],
            true,
            true,
        );
        registry = registry.inject_new_values(
            vec![
                TokenizedEntityValue::new("Michael Jackson", vec!["michael", "jackson"]),
                TokenizedEntityValue::new("Blur", vec!["blur"]),
            ],
            false,
            false,
        );

        // Then
        let resulting_gazetteer = registry.get_entity_values(true);
        let expected_gazetteer = vec![
            RegisteredEntityValue::new("Pink Floyd", vec!["pink", "floyd"], true, 0),
            RegisteredEntityValue::new("Blink", vec!["blink"], true, 1),
            RegisteredEntityValue::new("Daft Punk", vec!["daft", "punk"], false, 2),
            RegisteredEntityValue::new("Metronomy", vec!["metronomy"], false, 3),
            RegisteredEntityValue::new("Michael Jackson", vec!["michael", "jackson"], true, 4),
            RegisteredEntityValue::new("Blur", vec!["blur"], true, 5),
        ];
        assert_eq!(expected_gazetteer, resulting_gazetteer);
    }

    #[test]
    fn test_should_inject_values_from_vanilla() {
        // Given
        let mut registry = ParserRegistry::default();
        registry.add_value(
            TokenizedEntityValue::new("Daft Punk", vec!["daft", "punk"]),
            0,
        );
        registry.add_value(TokenizedEntityValue::new("Metronomy", vec!["metronomy"]), 2);
        registry = registry.inject_new_values(
            vec![
                TokenizedEntityValue::new("Pink Floyd", vec!["pink", "floyd"]),
                TokenizedEntityValue::new("Blink", vec!["blink"]),
            ],
            true,
            true,
        );

        // When
        registry = registry.inject_new_values(
            vec![
                TokenizedEntityValue::new("Michael Jackson", vec!["michael", "jackson"]),
                TokenizedEntityValue::new("Blur", vec!["blur"]),
            ],
            true,
            true,
        );

        // Then
        let resulting_gazetteer = registry.get_entity_values(true);
        let expected_gazetteer = vec![
            RegisteredEntityValue::new("Michael Jackson", vec!["michael", "jackson"], true, 0),
            RegisteredEntityValue::new("Blur", vec!["blur"], true, 1),
            RegisteredEntityValue::new("Daft Punk", vec!["daft", "punk"], false, 2),
            RegisteredEntityValue::new("Metronomy", vec!["metronomy"], false, 3),
        ];
        assert_eq!(expected_gazetteer, resulting_gazetteer);
    }

    #[test]
    fn should_not_inject_empty_values() {
        // Given
        let mut registry = ParserRegistry::default();
        registry.add_value(TokenizedEntityValue::new("Blink", vec!["blink"]), 0);
        registry = registry.inject_new_values(
            vec![
                TokenizedEntityValue::new("  ", Vec::<String>::new()),
                TokenizedEntityValue::new("Pink", vec!["pink"]),
            ],
            true,
            true,
        );

        // When
        let entity_values = registry.get_entity_values(true);

        // Then
        let expected_values = vec![
            RegisteredEntityValue::new("Pink", vec!["pink"], true, 0),
            RegisteredEntityValue::new("Blink", vec!["blink"], false, 1),
        ];
        assert_eq!(expected_values, entity_values);
    }

    #[test]
    fn test_injection_should_update_stop_words() {
        let mut registry = ParserRegistry::default();
        registry.add_value(
            TokenizedEntityValue::new("The Rolling Stones", vec!["the", "rolling", "stones"]),
            0,
        );
        registry.add_value(
            TokenizedEntityValue::new("The Stones", vec!["the", "stones"]),
            1,
        );
        registry.set_stop_words(2, vec!["hello".to_string()]);

        let expected_stop_words: HashSet<String> =
            vec!["the".to_string(), "stones".to_string(), "hello".to_string()]
                .into_iter()
                .collect();

        let expected_edge_cases: HashSet<String> =
            vec!["The Stones".to_string()].into_iter().collect();

        assert_eq!(expected_stop_words, registry.get_stop_words());
        assert_eq!(expected_edge_cases, registry.get_edge_cases());

        let new_values = vec![
            TokenizedEntityValue::new("Rolling", vec!["rolling"]),
            TokenizedEntityValue::new("Rolling Two", vec!["rolling", "two"]),
        ];

        registry = registry.inject_new_values(new_values, true, false);

        let expected_stop_words: HashSet<String> = vec![
            "the".to_string(),
            "rolling".to_string(),
            "hello".to_string(),
        ]
        .into_iter()
        .collect();

        let expected_edge_cases: HashSet<String> =
            vec!["Rolling".to_string()].into_iter().collect();

        assert_eq!(expected_stop_words, registry.get_stop_words());
        assert_eq!(expected_edge_cases, registry.get_edge_cases());
    }
}
