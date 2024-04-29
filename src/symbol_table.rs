use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Implementation of a symbol table that
/// - always maps a given index to a single string
/// - allows mapping a string to several indices
#[derive(Clone, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
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
            .copied()
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
}

#[derive(Clone, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub struct ResolvedSymbolTable {
    index_to_resolved: Vec<String>,
    length: u32,
}

impl ResolvedSymbolTable {
    /// Add a symbol to the symbol table. If the symbol already exists, this will
    /// generate a new index to allow the symbol to be duplicated in the symbol table
    /// Returns the newly generated corresponding index
    pub fn add_symbol(&mut self, symbol: String) -> u32 {
        self.index_to_resolved.push(symbol);
        self.length += 1;
        self.length - 1
    }

    /// Find a symbol from its index
    pub fn find_index(&self, index: &u32) -> Option<&String> {
        if *index >= self.length {
            None
        } else {
            Some(&self.index_to_resolved[*index as usize])
        }
    }
}

impl IntoIterator for ResolvedSymbolTable {
    type Item = String;
    type IntoIter = std::vec::IntoIter<String>;

    fn into_iter(self) -> Self::IntoIter {
        self.index_to_resolved.into_iter()
    }
}
