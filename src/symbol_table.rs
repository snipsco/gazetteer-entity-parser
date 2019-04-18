/// Implementation of a symbol table that
/// - always maps a given index to a single string
/// - allows mapping a string to several indices
use std::collections::BTreeMap;

use serde_derive::*;

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
            .cloned()
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
    pub fn find_index(&self, idx: u32) -> Option<&str> {
        self.string_to_index
            .iter()
            .find(|(_, sym_idx)| **sym_idx == idx)
            .map(|(symbol, _)| &**symbol)
    }

    /// Remove the unique symbol corresponding to an index in the symbol table
    pub fn remove_index(&mut self, idx: u32) -> Option<String> {
        self.find_index(idx)
            .map(|sym| sym.to_string())
            .and_then(|symbol| {
                self.string_to_index
                    .remove(symbol.as_str())
                    .map(|_| symbol.to_string())
            })
    }
}

#[derive(PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub struct ResolvedSymbolTable {
    index_to_resolved: BTreeMap<u32, String>,
    index_to_resolved_id: BTreeMap<u32, String>,
    available_index: u32,
}

impl ResolvedSymbolTable {
    /// Add a symbol to the symbol table, along with its optional identifier. If the symbol already
    /// exists, this will generate a new index to allow the symbol to be duplicated in the symbol
    /// table
    /// Returns the newly generated corresponding index
    pub fn add_symbol(&mut self, symbol: String, symbol_id: Option<String>) -> u32 {
        let available_index = self.available_index;
        self.index_to_resolved.insert(available_index, symbol);
        if let Some(id) = symbol_id {
            self.index_to_resolved_id.insert(available_index, id);
        }
        self.available_index += 1;
        available_index
    }

    /// Find a symbol from its index
    pub fn find_index(&self, index: u32) -> Option<&str> {
        self.index_to_resolved.get(&index).map(|sym| &**sym)
    }

    /// Get the ids associated to this symbol
    pub fn get_associated_ids(&self, symbol: &str) -> Vec<&str> {
        self.find_symbol(symbol)
            .into_iter()
            .flat_map(|index| self.index_to_resolved_id.get(&index).map(|id| &**id))
            .collect()
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
            .flat_map(|idx| {
                self.index_to_resolved_id.remove(&idx);
                self.index_to_resolved.remove(&idx).map(|_| idx)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_in_token_symbol_table() {
        // Given
        let mut symtable = TokenSymbolTable::default();

        // When
        let symbol = "hello";
        let index = symtable.add_symbol(symbol.to_string());

        // Then
        assert_eq!(Some(&index), symtable.find_symbol(symbol));
        assert_eq!(Some(symbol), symtable.find_index(index));
    }

    #[test]
    fn test_remove_in_token_symbol_table() {
        // Given
        let mut symtable = TokenSymbolTable::default();
        let index_hello = symtable.add_symbol("hello".to_string());
        symtable.add_symbol("world".to_string());

        // When
        let removed_hello = symtable.remove_index(index_hello);
        let hello_sym = symtable.find_index(index_hello);
        let hello_idx = symtable.find_symbol("hello");

        // Then
        assert_eq!(Some("hello".to_string()), removed_hello);
        assert_eq!(None, hello_sym);
        assert_eq!(None, hello_idx);
    }

    #[test]
    fn test_add_in_resolved_symbol_table() {
        // Given
        let mut symtable = ResolvedSymbolTable::default();

        // When
        let symbol = "hello";
        let index_1 = symtable.add_symbol(symbol.to_string(), Some("id_42".to_string()));
        let index_2 = symtable.add_symbol(symbol.to_string(), Some("id_43".to_string()));
        symtable.add_symbol("world".to_string(), Some("id_44".to_string()));

        // Then
        assert_eq!(vec![index_1, index_2], symtable.find_symbol(symbol));
        assert_eq!(Some("hello"), symtable.find_index(index_1));
        assert_eq!(Some("hello"), symtable.find_index(index_2));
        assert_eq!(vec!["id_42", "id_43"], symtable.get_associated_ids("hello"))
    }

    #[test]
    fn test_remove_in_resolved_symbol_table() {
        // Given
        let mut symtable = ResolvedSymbolTable::default();
        let index_hello_1 = symtable.add_symbol("hello".to_string(), Some("42".to_string()));
        let index_hello_2 = symtable.add_symbol("hello".to_string(), Some("43".to_string()));
        symtable.add_symbol("world".to_string(), None);

        // When
        let removed_hello_indices = symtable.remove_symbol("hello");
        let hello_sym = symtable.find_index(index_hello_1);
        let hello_indices = symtable.find_symbol("hello");

        // Then
        assert_eq!(vec![index_hello_1, index_hello_2], removed_hello_indices);
        assert_eq!(None, hello_sym);
        assert_eq!(Vec::<u32>::new(), hello_indices);
        assert_eq!(Vec::<String>::new(), symtable.get_associated_ids("hello"));
    }
}
