use data::Gazetteer;
use data::EntityValue;
use errors::SnipsResolverResult;
use snips_fst::{fst, operations, arc_iterator};
use snips_fst::symbol_table::SymbolTable;
use constants::EPS;
use std::path::Path;
use std::ops::Range;

pub struct Resolver {
    fst: fst::Fst,
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
        Ok(Resolver {
            fst,
            symbol_table
        })
    }

    pub fn add_value(&mut self, entity_value: &EntityValue) -> SnipsResolverResult<()> {
        // compute weight for each arc based on size of string
        let n_tokens = entity_value.verbalized_value.matches(" ").count();
        let weight = 1.0 / (n_tokens as f32);
        // add arcs consuming the raw value
        let mut current_head = self.fst.start();
        let mut next_head: i32;
        let mut token_idx: i32;

        for token in entity_value.verbalized_value.split_whitespace() {
            next_head = self.fst.add_state();
            // println!("token: {:?}", token);
            token_idx = self.symbol_table.add_symbol(token)?;
            // println!("{:?}", token);
            // Each arc can either consume a token...
            self.fst
                .add_arc(current_head, token_idx, 0, fst::Fst::weight_one(), next_head);

            // Or eps, with a certain weight
            self.fst
                .add_arc(current_head, 0, 0, weight, next_head);
            current_head = next_head;
        }
        // Add arcs outputting the raw value
        for token in entity_value.raw_value.split_whitespace() {
            next_head = self.fst.add_state();
            token_idx = self.symbol_table.add_symbol(token)?;
            self.fst
                .add_arc(current_head, 0, token_idx, fst::Fst::weight_one(), next_head);
            current_head = next_head;
        }
        // Make current head final, with weight given by entity value
        self.fst.set_final(current_head, entity_value.weight);
        Ok(())
    }

    pub fn from_gazetteer(gazetteer: &Gazetteer) -> SnipsResolverResult<Resolver> {
        let mut resolver = Resolver::new()?;
        for entity_value in &gazetteer.data {
            resolver.add_value(&entity_value)?;
        }
        // FIXME: The bench stops working when optimizing the fst...
        // resolver.fst.optimize();
        resolver.fst.rmepsilon();
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
        let mut token_idx: i32;
        for token in input.split_whitespace() {
            // println!("token: {:?}", token);
            match self.symbol_table.find_symbol(token)? {
                Some(value) => token_idx = value,
                None => {
                    // println!("token {:?} NOT FOUND IN SYMBOL TABLE", token);
                    return Ok("".to_string())}
            }
            next_head = input_fst.add_state();
            input_fst.add_arc(current_head, token_idx, token_idx, fst::Fst::weight_one(), next_head);
            current_head = next_head;
        }
        input_fst.set_final(current_head, fst::Fst::weight_one());
        input_fst.optimize();
        input_fst.arc_sort(false);
        // Compose with the resolver fst
        let mut composition = operations::compose(&input_fst, &self.fst);
        // let mut composition = operations::compose(&self.fst, &input_fst);
        if !(composition.num_states() > 0) {
            return Ok("".to_string())
        }
        composition.project(true);
        // DEBUG
        // composition.write_file(Path::new("composition.fst"))?;
        // Compute the shortest path: we try to get two to check if there is
        // a tie in terms of cost
        // println!("composition done");
        let shortest_path = composition.shortest_path(2, true, false);
        // println!("shortest path computed");
        // DEBUG
        // input_fst.write_file(Path::new("input_fst.fst"))?;
        // self.fst.write_file(Path::new("resolver_fst.fst"))?;
        // shortest_path.write_file(Path::new("shortest_path.fst"))?;
        // self.symbol_table.write_file(Path::new("/Users/alaasaade/Documents/nr-builtin-resolver/symbol_table.txt"), false)?;
        // unimplemented!()

        // Decode the shortest path
        let mut decoded_tokens: Vec<String> = Vec::new();
        let mut current_head = shortest_path.start();
        // println!("start computed");
        // println!("current head: {:?}", current_head);
        let mut num_arcs: i32;
        while !shortest_path.is_final(current_head) {
            let arc_iterator = arc_iterator::ArcIterator::new(&shortest_path, current_head);
            num_arcs = 0;
            for arc in arc_iterator {
                match num_arcs {
                    0 => num_arcs += 1,
                    // If there is more than one shortest path, the resolution
                    // fails
                    _ => return Ok("".to_string())
                }
                if arc.olabel() == 0 {
                    current_head = arc.nextstate();
                    continue;
                }
                decoded_tokens.push(self.symbol_table.find_index(arc.olabel())?.unwrap());
                current_head = arc.nextstate();
            }
        }
        Ok(decoded_tokens.join(" "))
        // DEBUG
        // shortest_path.write_file(Path::new("test.fst"))?;
        // // input_fst.write_file(Path::new("test.fst"))?;
        // self.symbol_table.write_file(Path::new("test.symt"), false)?;
        // unimplemented!()
    }

}

// Do a test that checks what happens if I add a the same symbol twice in a row

#[cfg(test)]
mod tests {
    use super::*;
    use data::EntityValue;
    #[test]
    fn test_resolver() {
        let mut gazetteer = Gazetteer { data: Vec::new() };
        gazetteer.add(EntityValue {
            weight: 1.0,
            raw_value: "The Rolling Stones".to_string(),
            verbalized_value: "the rolling stones".to_string(),
        });
        gazetteer.add(EntityValue {
            weight: 1.0,
            raw_value: "The Flying Stones".to_string(),
            verbalized_value: "the flying stones".to_string(),
        });
        gazetteer.add(EntityValue {
            weight: 1.0,
            raw_value: "Blink-182".to_string(),
            verbalized_value: "blink one eight two".to_string(),
        });
        let resolver = Resolver::from_gazetteer(&gazetteer).unwrap();

        let resolved = resolver.run("the stones".to_string()).unwrap();
        assert_eq!(resolved, "");

        let resolved = resolver.run("rolling stones".to_string()).unwrap();
        assert_eq!(resolved, "The Rolling Stones");

        let resolved = resolver.run("blink stones".to_string()).unwrap();
        assert_eq!(resolved, "");

        let resolved = resolver.run("blink one eight two".to_string()).unwrap();
        assert_eq!(resolved, "Blink-182");
    }
}
