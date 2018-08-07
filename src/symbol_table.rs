/// Implementation of a symbol table that
/// - always maps a given index to a single string
/// - allows mapping a string to several indices

use std::ops::Range;
use std::path::Path;
use errors::GazetteerParserResult;
// use std::collections::{HashMap};
use fnv::FnvHashMap as HashMap;
use std::fs;
use serde::{Serialize};
use rmps::{Serializer, from_read};


#[derive(PartialEq)]
pub struct GazetteerParserSymbolTable {
    index_to_string: Vec<String>,
    string_to_indices: HashMap<String, Vec<u32>>
}

impl GazetteerParserSymbolTable {
    pub fn new() -> GazetteerParserSymbolTable {
        let index_to_string = Vec::new();
        let string_to_indices = HashMap::default();
        GazetteerParserSymbolTable{index_to_string, string_to_indices }
    }

    /// Add a new string symbol to the symbol table. The boolean force_add can be set to true to
    /// force adding the value once more, even though it may already be in the symbol table
    /// This function raises an error if called with force_add set to false on a symbol that is
    /// already present several times in the symbol table
    pub fn add_symbol(&mut self, symbol: &str, force_add: bool) -> GazetteerParserResult<u32> {
        if force_add || !self.string_to_indices.contains_key(symbol) {
            self.index_to_string.push(symbol.to_string());
            let symbol_idx = (self.index_to_string.len() - 1) as u32;
            self.string_to_indices.entry(symbol.to_string())
                .and_modify(|v| {v.push(symbol_idx);})
                .or_insert({
                    vec![symbol_idx]
                });
            Ok(symbol_idx)
        } else {
            let indices = self.string_to_indices.get(symbol)
                .ok_or_else(|| format_err!("Symbol {:?} missing from symbol table",
                                          symbol))?;
            if indices.len() > 1 {
                bail!("Symbol {:?} is already present several times in the symbol table, cannot determine which index to return", symbol);
            }
            Ok(*indices.first().ok_or_else(|| format_err!("Symbol {:?} not mapped to any index in symbol table", symbol))?)
        }
    }

    /// Get a vec of all the values in the symbol table
    pub fn get_all_symbols(&self) -> Vec<&String> {
        self.string_to_indices.keys().collect()
    }

    /// Get the range of the integer values used to reprent the symbols in the symbol table
    pub fn get_indices_range(&self) -> Range<u32> {
        0..self.index_to_string.len() as u32
    }

    /// Find the indices of a symbol in the symbol table.
    pub fn find_symbol(&self, symbol: &str) -> GazetteerParserResult<Option<&Vec<u32>>> {
        Ok(self.string_to_indices.get(symbol))
    }

    /// Find the unique index of a symbol, and raise an error if it has more than one index
    // #[inline(never)]
    pub fn find_single_symbol(&self, symbol: &str) -> GazetteerParserResult<Option<u32>> {
        match self.find_symbol(symbol)? {
            Some(vec) if vec.len() == 1 => {Ok(Some(*vec.first().unwrap()))}
            Some(vec) if vec.len() == 0 => {bail!("Symbol {:?} missing from symbol table", symbol)}
            Some(vec) if vec.len() > 0 => {bail!("Symbol {:?} present more than once in symbol table", symbol)}
            _ => Ok(None)
        }
    }

    /// Find the unique symbol corresponding to an index in the symbol table
    // #[inline(never)]
    pub fn find_index(&self, idx: u32) -> GazetteerParserResult<String> {
        let symb = self.index_to_string.get(idx as usize).ok_or_else(|| format_err!("Could not find index {:?} in the symbol table", idx))?;
        Ok(symb.to_string())
    }

    /// The only part of the symbol table that is serialized is the index_to_string vec.
    /// The converse hashmap is built upon loading
    pub fn write_file<P: AsRef<Path>>(&self, filename: P) -> GazetteerParserResult<()> {
        let mut writer = fs::File::create(filename)?;
        self.index_to_string.serialize(&mut Serializer::new(&mut writer))?;
        Ok(())
    }

    pub fn from_path<P: AsRef<Path>>(filename: P) -> GazetteerParserResult<GazetteerParserSymbolTable> {
        let reader = fs::File::open(filename)?;
        let index_to_string: Vec<String> = from_read(reader)?;
        let mut string_to_indices: HashMap<String, Vec<u32>> = HashMap::default();
        for (idx, symbol) in index_to_string.iter().enumerate() {
            string_to_indices.entry(symbol.to_string())
                .and_modify(|v| {v.push(idx as u32);})
                .or_insert(vec![idx as u32]);
        }
        Ok(GazetteerParserSymbolTable{index_to_string, string_to_indices})
    }

}
