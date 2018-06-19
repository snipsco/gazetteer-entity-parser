use std::cmp::max;
use data::Gazetteer;
use data::EntityValue;
use errors::SnipsResolverResult;
use snips_fst::{fst, operations, arc_iterator, arc, string_paths_iterator};
use snips_fst::symbol_table::SymbolTable;
use constants::{ EPS, SKIP };
use std::ops::Range;
use std::path::Path;

pub struct Resolver {
    pub fst: fst::Fst,
    pub symbol_table: SymbolTable,
}

pub struct ResolvedValue {
    resolved_value: String,
    range: Range<usize>,  // character-level
    raw_value: String
}

// TODO: replace String by &str unless in constructor if object needs to own said string

impl Resolver {
    pub fn new() -> SnipsResolverResult<Resolver> {
        // Add a FST with a single state and set it as start
        let mut fst = fst::Fst::new();
        let start_state = fst.add_state();
        fst.set_start(start_state);
        // Add a symbol table with epsilon
        let mut symbol_table = SymbolTable::new();
        let eps_idx = symbol_table.add_symbol(&EPS)?;
        assert_eq!(eps_idx, 0); // There must be a cleaner way to do this
        let skip_idx = symbol_table.add_symbol(&SKIP)?;
        assert_eq!(skip_idx, 1); // There must be a cleaner way to do this
        Ok(Resolver {
            fst,
            symbol_table
        })
    }

    pub fn add_value(&mut self, entity_value: &EntityValue) -> SnipsResolverResult<()> {
        // compute weight for each arc based on size of string
        let n_tokens = entity_value.verbalized_value.matches(" ").count() + 1;
        // let weight = 1.0 / (max(n_tokens - 1, 1) as f32);
        let weight = 1.0 / (n_tokens as f32);
        // add arcs consuming the raw value
        let mut current_head = self.fst.start();
        let mut next_head: i32;
        let mut token_idx: i32;

        // Experiment accepting the whole verbalized value and outputting the
        // whole raw value in one pass
        // next_head = self.fst.add_state();
        // let token_input_idx = self.symbol_table.add_symbol(&entity_value.verbalized_value)?;
        // let token_output_idx = self.symbol_table.add_symbol(&entity_value.raw_value)?;
        // self.fst
        //     .add_arc(current_head, token_input_idx, token_output_idx, fst::Fst::weight_one(), next_head);
        // current_head = next_head;

        next_head = self.fst.add_state();
        for token in entity_value.verbalized_value.split_whitespace() {
            token_idx = self.symbol_table.add_symbol(token)?;
            // self.fst
            //     .add_arc(current_head, token_idx, 0, fst::Fst::weight_one(), next_head);
            self.fst
                .add_arc(current_head, token_idx, 0, - weight, next_head);
        }
        current_head = next_head;
        if n_tokens > 1 {
            for token in entity_value.verbalized_value.split_whitespace() {
                next_head = self.fst.add_state();
                // println!("token: {:?}", token);
                token_idx = self.symbol_table.add_symbol(token)?;
                // println!("{:?}", token);
                // Each arc can either consume a token...
                // self.fst
                //     .add_arc(current_head, token_idx, 0, fst::Fst::weight_one(), next_head);

                self.fst
                    .add_arc(current_head, token_idx, 0, 0.0, next_head);

                // Or eps, with a certain weight -- output SKIP
                self.fst
                    .add_arc(current_head, 0, 1, weight, next_head);

                // Or insert a token with a certain weight
                // self.fst
                //     .add_arc(current_head, 1, 1, weight, next_head);

                // Update current head
                current_head = next_head;
            }
        }
        // Add arcs outputting the raw value
        // for token in entity_value.raw_value.split_whitespace() {
        //     next_head = self.fst.add_state();
        //     token_idx = self.symbol_table.add_symbol(token)?;
        //     self.fst
        //         .add_arc(current_head, 0, token_idx, fst::Fst::weight_one(), next_head);
        //     current_head = next_head;
        // }

        // Output the full raw value in one pass
        next_head = self.fst.add_state();
        token_idx = self.symbol_table.add_symbol(&entity_value.raw_value.replace(" ", "_"))?;
        // self.fst
        //     .add_arc(current_head, 0, token_idx, fst::Fst::weight_one(), next_head);
        self.fst
            .add_arc(current_head, 0, token_idx, 0.0, next_head);
        current_head = next_head;

        // Make current head final, with weight given by entity value
        // self.fst.set_final(current_head, entity_value.weight);
        self.fst.set_final(current_head, 0.0);
        Ok(())
    }

    pub fn from_gazetteer(gazetteer: &Gazetteer) -> SnipsResolverResult<Resolver> {
        let mut resolver = Resolver::new()?;
        for entity_value in &gazetteer.data {
            resolver.add_value(&entity_value)?;
        }
        // FIXME: The bench stops working when optimizing the fst...
        // resolver.fst.closure_plus();
        // resolver.fst.prune(1.0);
        resolver.fst.optimize();
        // resolver.fst.closure_plus();
        // resolver.fst.rmepsilon();
        // rhs should be arc sorted on input labels
        resolver.fst.arc_sort(true);
        // resolver.fst = operations::determinize(&resolver.fst);
        // resolver.fst.minimize();
        Ok(resolver)
    }

    pub fn run(&self, input: String) -> SnipsResolverResult<String> {
        // FIXME: implement logic when two paths exist in composition but with different weights
        // build the input fst
        let mut input_fst = fst::Fst::new();
        let mut current_head = input_fst.add_state();
        input_fst.set_start(current_head);
        let mut next_head: i32;
        // let mut token_idx: i32;

        // match self.symbol_table.find_symbol(&input)? {
        //     Some(value) => token_idx = value,
        //     None => {
        //         // println!("token {:?} NOT FOUND IN SYMBOL TABLE", token);
        //         return Ok("".to_string())}
        // }
        // next_head = input_fst.add_state();
        // input_fst.add_arc(current_head, token_idx, token_idx, fst::Fst::weight_one(), next_head);
        // current_head = next_head;

        for token in input.split_whitespace() {
            // println!("token: {:?}", token);

            // let token_idx = self.symbol_table.find_symbol(token)?;

            match self.symbol_table.find_symbol(token)? {
                Some(value) => {

                    // Allow inserting a token between words, with a certain
                    // weight (coming from the resolver fst, the inserted token)
                    // is consumed with a larger weight
                    // next_head = input_fst.add_state();
                    // input_fst.add_arc(current_head, 0, 1, fst::Fst::weight_one(), next_head);
                    // // skip to the next token
                    // input_fst.add_arc(current_head, 0, 0, fst::Fst::weight_one(), next_head);
                    // current_head = next_head;


                    next_head = input_fst.add_state();
                    input_fst.add_arc(current_head, value, value, fst::Fst::weight_one(), next_head);
                    // Allow skipping the word with a large weight
                    // input_fst.add_arc(current_head, value, 0, 100.0, next_head);
                    input_fst.set_final(next_head, fst::Fst::weight_one());
                    current_head = next_head;
                }
                None => {
                    // if the word is not in the symbol table, there is no
                    // chance of matching it: we skip
                    continue;
                    // Allow skipping the word with a large weight
                    // input_fst.add_arc(current_head, 0, 0, 100.0, next_head);
                    // println!("token {:?} NOT FOUND IN SYMBOL TABLE", token);
                    // return Ok("".to_string())}
                }
            }
            // if token_idx.is_some() {
            //     input_fst.add_arc(current_head, token_idx.unwrap(), token_idx.unwrap(), fst::Fst::weight_one(), next_head);
            // }
            // Allow skipping the word with a large weight
            // input_fst.add_arc(current_head, token_idx, 0, 100.0, next_head);
        }
        // Also do it at the end
        // Allow inserting a token between words, with a certain
        // weight (coming from the resolver fst, the inserted token)
        // is consumed with a larger weight
        // next_head = input_fst.add_state();
        // input_fst.add_arc(current_head, 0, 1, fst::Fst::weight_one(), next_head);
        // // skip to the next token
        // input_fst.add_arc(current_head, 0, 0, fst::Fst::weight_one(), next_head);
        // current_head = next_head;


        // Set final state
        input_fst.set_final(current_head, fst::Fst::weight_one());

        // input_fst.write_file(Path::new("input_fst.fst"))?;
        // self.symbol_table.write_file(Path::new("/Users/alaasaade/Documents/nr-builtin-resolver/symbol_table.txt"), false)?;
        // self.fst.write_file(Path::new("resolver_fst.fst"))?;

        input_fst.optimize();
        // input_fst.rmepsilon();
        input_fst.arc_sort(false);
        // Compose with the resolver fst
        let mut composition = operations::compose(&input_fst, &self.fst);
        // let mut composition = operations::lazy_compose_with_lookahead(&input_fst, &self.fst, 100000);
        // let mut composition = operations::compose(&self.fst, &input_fst);
        // if !(composition.num_states() > 0) {
        //     // println!("{:?}", "Empty composition");
        //     return Ok("".to_string())
        // }
        // composition.project(true);
        // DEBUG
        // composition.write_file(Path::new("composition.fst"))?;
        // Compute the shortest path: we try to get two to check if there is
        // a tie in terms of cost
        // println!("composition done");
        // println!("About to compute shortest_path");
        let shortest_path = composition.shortest_path(1, false, false);
        if shortest_path.num_states() == 0 {
            return Ok("".to_string())
        }
        // println!("Shortest path computed");
        // println!("shortest path computed");
        // DEBUG
        // composition.write_file(Path::new("composition.fst"))?;
        // input_fst.write_file(Path::new("input_fst.fst"))?;
        // self.fst.write_file(Path::new("resolver_fst.fst"))?;
        // shortest_path.optimize();
        // shortest_path.project(true);
        // shortest_path.write_file(Path::new("shortest_path.fst"))?;
        // self.symbol_table.write_file(Path::new("/Users/alaasaade/Documents/nr-builtin-resolver/symbol_table.txt"), false)?;
        // unimplemented!()

        // Decoding shortest path
        let mut path_iterator = string_paths_iterator::StringPathsIterator::new(&shortest_path, &self.symbol_table, &self.symbol_table, true, true);

        // We check decode using the first path if it's weight is strictly less than the second
        let first_path = path_iterator.next().unwrap()?;
        // println!("first path: {:?}", first_path);
        if path_iterator.done() {
            return Ok(first_path.ostring)
        }
        let second_path = path_iterator.next().unwrap()?;
        if second_path.weight > first_path.weight {
            return Ok(first_path.ostring)
        } else if first_path.weight > second_path.weight {
            return Ok(second_path.ostring)
        } else {
            return Ok("".to_string())
        }
    }
}


// Do a test that checks what happens if I add a the same symbol twice in a row

#[cfg(test)]
extern crate serde_json;
mod tests {
    use super::*;
    use data::EntityValue;
    use std::fs::File;
    use serde_json;
    use std::path::Path;

    #[test]
    fn test_resolver() {
        let mut gazetteer = Gazetteer { data: Vec::new() };
        gazetteer.add(EntityValue {
            weight: 1.0,
            raw_value: "The Flying Stones".to_string(),
            verbalized_value: "the flying stones".to_string(),
        });
        gazetteer.add(EntityValue {
            weight: 1.0,
            raw_value: "The Rolling Stones".to_string(),
            verbalized_value: "the rolling stones".to_string(),
        });
        gazetteer.add(EntityValue {
            weight: 1.0,
            raw_value: "Blink-182".to_string(),
            verbalized_value: "blink one eight two".to_string(),
        });
        gazetteer.add(EntityValue {
            weight: 1.0,
            raw_value: "Je Suis Animal".to_string(),
            verbalized_value: "je suis animal".to_string(),
        });

        let resolver = Resolver::from_gazetteer(&gazetteer).unwrap();

        // resolver.fst.write_file(Path::new("resolver_fst.fst")).unwrap();
        // resolver.symbol_table.write_file(Path::new("symbol_table.txt"), false).unwrap();

        // let resolved = resolver.run("je veux ecouter les rolling stones".to_string()).unwrap();
        // assert_eq!(resolved, "<skip> <skip> Je_Suis_Animal <skip> The_Rolling_Stones");

        let resolved = resolver.run("je veux ecouter les rolling stones".to_string()).unwrap();
        assert_eq!(resolved, "<skip> <skip> <skip> Je_Suis_Animal");

        let resolved = resolver.run("je veux ecouter les rolling stones".to_string()).unwrap();
        assert_eq!(resolved, "<skip> <skip> <skip> Je_Suis_Animal");

        // print!("Resolver fst num states {:?}", resolver.fst.num_states());
        // resolver.fst.write_file(Path::new("resolver_fst_1.fst")).unwrap();
        // resolver.symbol_table.write_file(Path::new("symbol_table_1.txt"), false).unwrap();
        let resolved = resolver.run("rolling stones".to_string()).unwrap();
        assert_eq!(resolved, "<skip> <skip> The_Rolling_Stones");

        // resolver.fst.write_file(Path::new("resolver_fst_2.fst")).unwrap();
        // resolver.symbol_table.write_file(Path::new("symbol_table_2.txt"), false).unwrap();
        let resolved = resolver.run("i want to listen to rolling stones and blink one eight".to_string()).unwrap();
        // assert_eq!(resolved, "<skip> The_Rolling_Stones <skip> Blink-182");
        assert_eq!(resolved, "<skip> <skip> The_Rolling_Stones");

        let resolved = resolver.run("joue moi the stones".to_string()).unwrap();
        assert_eq!(resolved, "");

        let resolved = resolver.run("joue moi quelque chose".to_string()).unwrap();
        assert_eq!(resolved, "");

        let resolved = resolver.run("joue moi quelque chose des rolling stones".to_string()).unwrap();
        assert_eq!(resolved, "<skip> <skip> The_Rolling_Stones");

        // let resolved = resolver.run("the stones".to_string()).unwrap();
        // assert_eq!(resolved, "");

        // let resolved = resolver.run("rolling stones".to_string()).unwrap();
        // assert_eq!(resolved, "The Rolling Stones");

        // let resolved = resolver.run("blink stones".to_string()).unwrap();
        // assert_eq!(resolved, "");

        // let resolved = resolver.run("blink one eight two".to_string()).unwrap();
        // assert_eq!(resolved, "Blink-182");
    }

   //  #[test]
   //  fn test_large_gazetteer() {
   //      let mut gazetteer = Gazetteer { data: Vec::new() };
   //      let file = File::open("/Users/alaasaade/Documents/snips-grammars/snips_grammars/resources/fr/music/artist.json").unwrap();
   //      let mut data: Vec<String> = serde_json::from_reader(file).unwrap();
   //      // for idx in 1..10 {
   //      //     println!("{:?}", data.get(idx));
   //      // }
   //      data.truncate(100);
   //      for val in data {
   //          // println!("{:?}", val);
   //          // if val == "The Stones" {
   //          //     println!("{:?}", val);
   //          // }
   //          gazetteer.add(EntityValue {
   //              weight: 1.0,
   //              raw_value: val.clone(),
   //              verbalized_value: val.clone().to_lowercase()
   //          })
   //      }
   //      let resolver = Resolver::from_gazetteer(&gazetteer).unwrap();
   //      resolver.fst.write_file(Path::new("resolver.fst")).unwrap();
   //      resolver.symbol_table.write_file(Path::new("symbol_table.txt"), false).unwrap();
   //      assert_eq!(resolver.run("veux ecouter brel".to_string()).unwrap(), "<skip> <skip> Jacques_Brel");
   //      // assert_eq!(resolver.run("ariana grande".to_string()).unwrap(), "Ariana Grande");
   //      // assert_eq!(resolver.run("the stones".to_string()).unwrap(), "The Rolling Stones");
   //      // assert_eq!(resolver.run("je veux ecouter ariana grande".to_string()).unwrap(), "The Rolling Stones");
   //      // assert_eq!(resolver.run("ariana".to_string()).unwrap(), "Ariana Grande");
   // }
}
