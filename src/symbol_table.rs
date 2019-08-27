use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Implementation of a symbol table that
/// - always maps a given index to a single string
/// - allows mapping a string to several indices
#[derive(PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub struct TokenSymbolTable {
    string_to_index: BTreeMap<String, u32>,
    available_index: u32,
}

impl TokenSymbolTable {
    /// Add a symbol to the symbol table, if it doesn't already exist, and return
    /// the corresponding index
    pub fn add_symbol(&mut self, symbol: String) -> u32 {
        self.string_to_index
            .get(&symbol)
            .map(|idx| *idx)
            .unwrap_or_else(|| {
                let symbol_index = self.available_index;
                self.available_index += 1;
                self.string_to_index.insert(symbol.clone(), symbol_index);
                symbol_index
            })
    }

    /// Find the index of a symbol in the symbol table.
    pub fn find_symbol(&self, symbol: &str) -> Option<&u32> {
        self.string_to_index.get(symbol)
    }

    /// Find the unique symbol corresponding to an index in the symbol table
    pub fn find_index(&self, idx: &u32) -> Option<&String> {
        self.string_to_index
            .iter()
            .find(|(_, sym_idx)| *sym_idx == idx)
            .map(|(symbol, _)| symbol)
    }

    /// Remove the unique symbol corresponding to an index in the symbol table
    pub fn remove_index(&mut self, idx: &u32) -> Option<String> {
        let symbol = self.find_index(idx).cloned();
        symbol.and_then(|symbol| self.string_to_index.remove(&symbol).map(|_| symbol))
    }
}

#[derive(PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub struct ResolvedSymbolTable {
    index_to_resolved: BTreeMap<u32, String>,
    available_index: u32,
}

impl ResolvedSymbolTable {
    /// Add a symbol to the symbol table. If the symbol already exists, this will
    /// generate a new index to allow the symbol to be duplicated in the symbol table
    /// Returns the newly generated corresponding index
    pub fn add_symbol(&mut self, symbol: String) -> u32 {
        let available_index = self.available_index;
        self.index_to_resolved.insert(available_index, symbol);
        self.available_index += 1;
        available_index
    }

    /// Find a symbol from its index
    pub fn find_index(&self, index: &u32) -> Option<&String> {
        self.index_to_resolved.get(index)
    }

    /// Find all the indices corresponding to a single symbol
    pub fn find_symbol(&self, symbol: &str) -> Vec<u32> {
        self.index_to_resolved
            .iter()
            .filter(|(_, sym)| *sym == symbol)
            .map(|(idx, _)| *idx)
            .collect()
    }

    /// Remove a symbol and all its linked indices from the symbol table
    pub fn remove_symbol(&mut self, symbol: &str) -> Vec<u32> {
        let indices = self.find_symbol(symbol);
        indices
            .into_iter()
            .flat_map(|idx| self.index_to_resolved.remove(&idx).map(|_| idx))
            .collect()
    }

    /// Get a vec of all the integer values used to represent the symbols in the symbol table
    pub fn get_all_indices(&self) -> Vec<&u32> {
        self.index_to_resolved.keys().collect()
    }
}
