use constants::RESTART_IDX;
use constants::{EPS, EPS_IDX, RESTART, SKIP, SKIP_IDX};
use data::EntityValue;
use data::Gazetteer;
use errors::SnipsResolverResult;
use snips_fst::string_paths_iterator::{StringPath, StringPathsIterator};
use snips_fst::symbol_table::SymbolTable;
use snips_fst::{fst, operations};
use std::ops::Range;
use utils::whitespace_tokenizer;
use utils::{check_threshold, fst_format_resolved_value, fst_unformat_resolved_value};

pub struct Resolver {
    pub fst: fst::Fst,
    pub symbol_table: SymbolTable,
    pub decoding_threshold: f32,
}

#[derive(Debug, PartialEq)]
pub struct ResolvedValue {
    pub resolved_value: String,
    pub range: Range<usize>, // character-level
    pub raw_value: String,
}

impl Resolver {
    /// Create an empty resolver. Its resolver has a single, start state. Its symbol table has
    /// epsilon and a skip symbol
    pub fn new(decoding_threshold: f32) -> SnipsResolverResult<Resolver> {
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
        Ok(Resolver {
            fst,
            symbol_table,
            decoding_threshold,
        })
    }

    /// This function returns a FST that checks that at least one of the words in `verbalized_value`
    /// is matched. This allows to disable many branches during the composition. It returns the
    /// state at the output of the bottleneck. On the left, the bottleneck is connected to the
    /// start state of the fst. Each token is consumed with weight `weight_by_token` and is
    /// returned. The bottleneck starts by consuming a RESTART symbol.
    fn make_bottleneck(
        &mut self,
        verbalized_value: &String,
        weight_by_token: f32,
    ) -> SnipsResolverResult<i32> {
        let start_state = self.fst.start();
        let current_head = self.fst.add_state();
        self.fst.add_arc(start_state, RESTART_IDX, EPS_IDX, 0.0, current_head);
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
        entity_value: &EntityValue,
        weight_by_token: f32,
    ) -> SnipsResolverResult<()> {
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
        let token_idx = self.symbol_table
            .add_symbol(&fst_format_resolved_value(&entity_value.resolved_value))?;
        self.fst
            .add_arc(current_head, EPS_IDX, token_idx, 0.0, next_head);
        // Make current head final, with weight given by entity value
        self.fst.set_final(next_head, entity_value.weight);
        Ok(())
    }

    /// Add a single entity value to the resolver. This function is kept private to promote
    /// creating the resolver with a higher level function (such as `from_gazetteer`) that
    /// performs additional global optimizations.
    fn add_value(&mut self, entity_value: &EntityValue) -> SnipsResolverResult<()> {
        // compute weight for each arc based on size of string
        let n_tokens = whitespace_tokenizer(&entity_value.raw_value).count();
        let weight_by_token = 1.0 / (n_tokens as f32);
        let current_head = self.make_bottleneck(&entity_value.raw_value, -weight_by_token)?;
        self.make_value_transducer(current_head, &entity_value, weight_by_token)?;
        Ok(())
    }

    /// Create a resolver from a gazetteer. This function adds the entity values from the gazetteer
    /// and performs several optimizations on the resulting FST. This is the recommended method
    /// to define a resolver
    pub fn from_gazetteer(gazetteer: &Gazetteer, threshold: f32) -> SnipsResolverResult<Resolver> {
        let mut resolver = Resolver::new(threshold)?;
        for entity_value in &gazetteer.data {
            resolver.add_value(&entity_value)?;
        }
        resolver.fst.optimize();
        resolver.fst.closure_plus();
        resolver.fst.arc_sort(true);
        Ok(resolver)
    }

    /// Create an input fst from a string to be resolved. Outputs the input fst and a vec of ranges
    // of the tokens composing it
    fn build_input_fst(&self, input: &str) -> SnipsResolverResult<(fst::Fst, Vec<Range<usize>>)> {
        // build the input fst
        let mut input_fst = fst::Fst::new();
        let mut tokens_ranges: Vec<Range<usize>> = vec![];
        let mut current_head = input_fst.add_state();
        input_fst.set_start(current_head);
        let mut restart_inserted: bool = false;
        for (token_range, token) in whitespace_tokenizer(input) {
            match self.symbol_table.find_symbol(&token)? {
                Some(value) => {
                    let next_head = input_fst.add_state();
                    input_fst.add_arc(current_head, value, value, 0.0, next_head);
                    tokens_ranges.push(token_range);
                    current_head = next_head;
                    restart_inserted = false;
                }
                None => {
                    if !restart_inserted {
                        let next_head = input_fst.add_state();
                        input_fst.add_arc(current_head, EPS_IDX, RESTART_IDX, 0.0, next_head);
                        current_head = next_head;
                        restart_inserted = true;
                    }
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

    fn decode_shortest_path(
        &self,
        shortest_path: &fst::Fst,
        tokens_range: &Vec<Range<usize>>,
    ) -> SnipsResolverResult<Vec<ResolvedValue>> {
        let mut path_iterator = StringPathsIterator::new(
            &shortest_path,
            &self.symbol_table,
            &self.symbol_table,
            true,
            true,
        );
        match path_iterator.next() {
            None => return Ok(vec![]), // this should not happen because the shortest path should contain at least a path
            Some(value) => {
                return Ok(Resolver::format_string_path(
                    &value?,
                    &tokens_range,
                    self.decoding_threshold,
                )?)
            }
        }
    }

    fn format_string_path(
        string_path: &StringPath,
        tokens_range: &Vec<Range<usize>>,
        threshold: f32,
    ) -> SnipsResolverResult<Vec<ResolvedValue>> {
        let mut input_iterator = whitespace_tokenizer(&string_path.istring);
        let mut resolved_values: Vec<ResolvedValue> = vec![];
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
                let tentative_new_token = input_iterator.next();
                match tentative_new_token {
                    Some((_, value)) => {
                        current_input_token = value;
                        current_input_token_idx += 1;
                    }
                    None => {}
                }
            }
            if current_input_token != token {
                let range_start = current_ranges.first().unwrap().start;
                let range_end = current_ranges.last().unwrap().end;
                if check_threshold(input_value_until_now.len(), n_skips, threshold) {
                    resolved_values.push(ResolvedValue {
                        raw_value: input_value_until_now.join(" "),
                        resolved_value: fst_unformat_resolved_value(&token),
                        range: range_start..range_end,
                    });
                }
                // Reinitialize accumulators
                n_skips = 0;
                input_value_until_now = vec![];
                current_ranges = vec![];
                advance_input = false;
            } else {
                input_value_until_now.push(token);
                current_ranges.push(tokens_range
                    .get(current_input_token_idx)
                    .ok_or_else(|| format_err!("Decoding went wrong"))?);
                advance_input = true;
            }
        }
        Ok(resolved_values)
    }

    /// Resolve the input string `input`
    pub fn run(&self, input: &str) -> SnipsResolverResult<Vec<ResolvedValue>> {
        // FIXME: implement logic of ranking of artists

        let (input_fst, tokens_range) = self.build_input_fst(input)?;
        // Compose with the resolver fst
        let composition = operations::compose(&input_fst, &self.fst);
        if composition.num_states() == 0 {
            return Ok(vec![]);
        }
        let shortest_path = composition.shortest_path(1, false, false);

        let resolution = self.decode_shortest_path(&shortest_path, &tokens_range)?;

        Ok(resolution)
    }
}

#[cfg(test)]
extern crate serde_json;
mod tests {
    use super::*;
    #[test]
    // #[ignore]
    fn test_resolver() {
        let mut gazetteer = Gazetteer { data: Vec::new() };
        gazetteer.add(EntityValue {
            weight: 1.0,
            resolved_value: "The Flying Stones".to_string(),
            raw_value: "the flying stones".to_string(),
        });
        gazetteer.add(EntityValue {
            weight: 1.0,
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        gazetteer.add(EntityValue {
            weight: 1.0,
            resolved_value: "Blink-182".to_string(),
            raw_value: "blink one eight two".to_string(),
        });
        gazetteer.add(EntityValue {
            weight: 1.0,
            resolved_value: "Je Suis Animal".to_string(),
            raw_value: "je suis animal".to_string(),
        });
        let resolver = Resolver::from_gazetteer(&gazetteer, 0.0).unwrap();

        let mut resolved = resolver.run("je veux écouter les rolling stones").unwrap();
        assert_eq!(
            resolved,
            vec![
                ResolvedValue {
                    raw_value: "je".to_string(),
                    resolved_value: "Je Suis Animal".to_string(),
                    range: 0..2,
                },
                ResolvedValue {
                    raw_value: "rolling stones".to_string(),
                    resolved_value: "The Rolling Stones".to_string(),
                    range: 20..34,
                },
            ]
        );

        resolved = resolver
            .run("je veux ecouter les \t rolling stones")
            .unwrap();
        assert_eq!(
            resolved,
            vec![
                ResolvedValue {
                    raw_value: "je".to_string(),
                    resolved_value: "Je Suis Animal".to_string(),
                    range: 0..2,
                },
                ResolvedValue {
                    raw_value: "rolling stones".to_string(),
                    resolved_value: "The Rolling Stones".to_string(),
                    range: 22..36,
                },
            ]
        );

        resolved = resolver
            .run("i want to listen to rolling stones and blink eight")
            .unwrap();
        assert_eq!(
            resolved,
            vec![
                ResolvedValue {
                    raw_value: "rolling stones".to_string(),
                    resolved_value: "The Rolling Stones".to_string(),
                    range: 20..34,
                },
                ResolvedValue {
                    raw_value: "blink eight".to_string(),
                    resolved_value: "Blink-182".to_string(),
                    range: 39..50,
                },
            ]
        );
        resolved = resolver.run("joue moi quelque chose").unwrap();
        assert_eq!(resolved, vec![]);
    }

    #[test]
    fn test_resolver_with_ranking() {
        /* Weight is here a proxy for the ranking of an artist in a popularity
        index */

        let mut gazetteer = Gazetteer { data: Vec::new() };
        gazetteer.add(EntityValue {
            weight: 500.0,
            resolved_value: "Jacques Brel".to_string(),
            raw_value: "jacques brel".to_string(),
        });
        gazetteer.add(EntityValue {
            weight: 100.0,
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        gazetteer.add(EntityValue {
            weight: 10000.0,
            resolved_value: "The Flying Stones".to_string(),
            raw_value: "the flying stones".to_string(),
        });
        gazetteer.add(EntityValue {
            weight: 23000.0,
            resolved_value: "Daniel Brel".to_string(),
            raw_value: "daniel brel".to_string(),
        });
        let resolver = Resolver::from_gazetteer(&gazetteer, 0.5).unwrap();

        /* When there is a tie in terms of number of token matched, match the most popular choice */
        let resolved = resolver.run("je veux écouter the stones").unwrap();
        assert_eq!(
            resolved,
            vec![ResolvedValue {
                raw_value: "the stones".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
                range: 16..26,
            }]
        );
        let resolved = resolver.run("je veux écouter brel").unwrap();
        assert_eq!(
            resolved,
            vec![ResolvedValue {
                raw_value: "brel".to_string(),
                resolved_value: "Jacques Brel".to_string(),
                range: 16..20,
            }]
        );

        // Resolve to the value with more words matching regardless of popularity
        let resolved = resolver.run("je veux écouter the flying stones").unwrap();
        assert_eq!(
            resolved,
            vec![ResolvedValue {
                raw_value: "the flying stones".to_string(),
                resolved_value: "The Flying Stones".to_string(),
                range: 16..33,
            }]
        );
        let resolved = resolver.run("je veux écouter daniel brel").unwrap();
        assert_eq!(
            resolved,
            vec![ResolvedValue {
                raw_value: "daniel brel".to_string(),
                resolved_value: "Daniel Brel".to_string(),
                range: 16..27,
            }]
        );
    }

    #[test]
    fn test_resolver_with_restart() {
        let mut gazetteer = Gazetteer { data: Vec::new() };
        gazetteer.add(EntityValue {
            weight: 1.0 / 100.0,
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        let resolver = Resolver::from_gazetteer(&gazetteer, 0.5).unwrap();

        let resolved = resolver
            .run("the music I want to listen to is rolling on stones")
            .unwrap();
        assert_eq!(
            resolved,
            vec![]
        );
    }

    #[test]
    fn test_resolver_with_mixed_ordered_entity() {
        let mut gazetteer = Gazetteer { data: Vec::new() };
        gazetteer.add(EntityValue {
            weight: 1.0 / 100.0,
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        let resolver = Resolver::from_gazetteer(&gazetteer, 0.5).unwrap();

        let resolved = resolver.run("rolling the stones").unwrap();
        assert_eq!(resolved, vec![]);
    }

    #[test]
    fn test_resolver_with_threshold() {
        // TODO: implement this test
        let mut gazetteer = Gazetteer { data: Vec::new() };
        gazetteer.add(EntityValue {
            weight: 1.0,
            resolved_value: "The Flying Stones".to_string(),
            raw_value: "the flying stones".to_string(),
        });
        gazetteer.add(EntityValue {
            weight: 1.0,
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        gazetteer.add(EntityValue {
            weight: 1.0,
            resolved_value: "Blink-182".to_string(),
            raw_value: "blink one eight two".to_string(),
        });
        gazetteer.add(EntityValue {
            weight: 1.0,
            resolved_value: "Je Suis Animal".to_string(),
            raw_value: "je suis animal".to_string(),
        });
        gazetteer.add(EntityValue {
            weight: 1.0,
            resolved_value: "Les Enfoirés".to_string(),
            raw_value: "les enfoirés".to_string(),
        });
        let resolver = Resolver::from_gazetteer(&gazetteer, 0.5).unwrap();
        let resolved = resolver.run("je veux écouter les rolling stones").unwrap();
        assert_eq!(
            resolved,
            vec![
                ResolvedValue {
                    resolved_value: "Les Enfoirés".to_string(),
                    range: 16..19,
                    raw_value: "les".to_string(),
                },
                ResolvedValue {
                    raw_value: "rolling stones".to_string(),
                    resolved_value: "The Rolling Stones".to_string(),
                    range: 20..34,
                },
            ]
        );

        let resolver = Resolver::from_gazetteer(&gazetteer, 0.3).unwrap();
        let resolved = resolver.run("je veux écouter les rolling stones").unwrap();
        assert_eq!(
            resolved,
            vec![
                ResolvedValue {
                    raw_value: "je".to_string(),
                    resolved_value: "Je Suis Animal".to_string(),
                    range: 0..2,
                },
                ResolvedValue {
                    resolved_value: "Les Enfoirés".to_string(),
                    range: 16..19,
                    raw_value: "les".to_string(),
                },
                ResolvedValue {
                    raw_value: "rolling stones".to_string(),
                    resolved_value: "The Rolling Stones".to_string(),
                    range: 20..34,
                },
            ]
        );

        let resolver = Resolver::from_gazetteer(&gazetteer, 0.6).unwrap();
        let resolved = resolver.run("je veux écouter les rolling stones").unwrap();
        assert_eq!(
            resolved,
            vec![ResolvedValue {
                raw_value: "rolling stones".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
                range: 20..34,
            }]
        );
    }
}
