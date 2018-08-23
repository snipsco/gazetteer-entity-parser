/// Implementation of a symbol table that
/// - always maps a given index to a single string
/// - allows mapping a string to several indices

use errors::*;
use fnv::FnvHashMap as HashMap;
use serde::Deserializer;
use serde::Serializer;
use serde::{Deserialize, Serialize};

#[derive(PartialEq, Eq, Debug, Default)]
pub struct GazetteerParserSymbolTable {
    index_to_string: HashMap<u32, String>,
    string_to_indices: HashMap<String, Vec<u32>>,
    available_index: u32,
}

/// We define another struct representing a serialized symbol table.
/// This allows not serializing the two hashmaps but only one of them, to save space
#[derive(Serialize, Deserialize)]
struct SerializedGazetteerParserSymbolTable {
    index_to_string: HashMap<u32, String>,
    available_index: u32,
}

impl Serialize for GazetteerParserSymbolTable {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Manually copy the index to string table
        let mut index_to_string: HashMap<u32, String> =
            HashMap::with_capacity_and_hasher(self.index_to_string.len(), Default::default());
        for (idx, val) in &self.index_to_string {
            index_to_string.insert(*idx, val.to_string());
        }
        SerializedGazetteerParserSymbolTable {
            index_to_string,
            available_index: self.available_index,
        }.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for GazetteerParserSymbolTable {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let serialized_symt = SerializedGazetteerParserSymbolTable::deserialize(deserializer)?;
        // Recompute string_to_indices
        let mut string_to_indices: HashMap<String, Vec<u32>> = HashMap::default();
        for (idx, symbol) in &serialized_symt.index_to_string {
            string_to_indices
                .entry(symbol.to_string())
                .and_modify(|v| {
                    v.push(*idx);
                })
                .or_insert(vec![*idx]);
        }

        Ok(GazetteerParserSymbolTable {
            index_to_string: serialized_symt.index_to_string,
            string_to_indices,
            available_index: serialized_symt.available_index,
        })
    }
}

impl GazetteerParserSymbolTable {
    /// Add a new string symbol to the symbol table. The boolean force_add can be set to true to
    /// force adding the value once more, even though it may already be in the symbol table
    /// This function raises an error if called with force_add set to false on a symbol that is
    /// already present several times in the symbol table
    pub fn add_symbol(&mut self, symbol: String, force_add: bool) -> GazetteerParserResult<u32, SymbolTableAddSymbolError> {
        if force_add || !self.string_to_indices.contains_key(&symbol) {
            let available_index = self.available_index;
            self.index_to_string
                .insert(available_index, symbol.clone());
            self.string_to_indices
                .entry(symbol)
                .and_modify(|v| {
                    v.push(available_index);
                })
                .or_insert({ vec![available_index] });
            self.available_index += 1;
            Ok(available_index)
        } else {
            let indices = self
                .string_to_indices
                .get(&symbol)
                .ok_or_else(|| SymbolTableAddSymbolError::MissingKeyError {
                    key: symbol.clone()
                })?;
            if indices.len() > 1 {
                return Err(
                    SymbolTableAddSymbolError::DuplicateSymbolError {
                        symbol
                    }
                )
            }
            Ok(*indices.first().ok_or_else(||
                SymbolTableAddSymbolError::MissingKeyError {
                    key: symbol
                })?
            )
        }
    }

    pub fn remove_symbol(&mut self, symbol: &str) -> Option<Vec<u32>> {
        let indices_values: Vec<u32>;
        if let Some(indices) = self.find_symbol(symbol) {
            indices_values = indices.clone();
        } else {
            return None
        }

        // Remove the indices from both hashmaps
        for idx in &indices_values {
            self.index_to_string.remove(idx);
        }
        self.string_to_indices.remove(symbol);
        return Some(indices_values);
    }

    /// Get a vec of all the values in the symbol table
    pub fn _get_all_symbols(&self) -> Vec<&String> {
        self.string_to_indices.keys().collect()
    }

    /// Get a vec of all the integer values used to reprent the symbols in the symbol table
    pub fn get_all_indices(&self) -> Vec<&u32> {
        self.index_to_string.keys().collect()
    }

    /// Find the indices of a symbol in the symbol table.
    pub fn find_symbol(&self, symbol: &str) -> Option<&Vec<u32>> {
        self.string_to_indices.get(symbol)
    }

    /// Find the unique index of a symbol, and raise an error if it has more than one index
    pub fn find_single_symbol(&self, symbol: &str) -> GazetteerParserResult<Option<u32>, SymbolTableFindSingleSymbolError> {
        match self.find_symbol(symbol) {
            Some(vec) if vec.len() == 1 => Ok(Some(*vec.first().unwrap())),
            Some(vec) if vec.len() == 0 => {
                return Err(
                    SymbolTableFindSingleSymbolError::MissingKeyError {
                        key: symbol.to_string()
                    }
                )
            },
            Some(vec) if vec.len() > 0 => {
                return Err(
                    SymbolTableFindSingleSymbolError::DuplicateSymbolError {
                        symbol: symbol.to_string()
                    }
                )
            }
            _ => Ok(None),
        }
    }

    /// Find the unique symbol corresponding to an index in the symbol table
    pub fn find_index(&self, idx: &u32) -> GazetteerParserResult<String, SymbolTableFindIndexError> {
        let symb = self
            .index_to_string
            .get(idx)
            .ok_or_else(||
                SymbolTableFindIndexError::MissingKeyError {
                    key: *idx
                }
            )?;
        Ok(symb.to_string())
    }
}
