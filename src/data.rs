use crate::utils::whitespace_tokenizer;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::cmp::Ordering;
use std::ops::Range;
use std::result::Result;

/// Struct representing the value of an entity to be added to the parser
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
pub struct EntityValue {
    pub resolved_value: String,
    pub raw_value: String,
}

impl EntityValue {
    pub fn into_tokenized(self) -> TokenizedEntityValue {
        TokenizedEntityValue {
            resolved_value: self.resolved_value,
            tokens: whitespace_tokenizer(&self.raw_value)
                .map(|(_, token)| token)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TokenizedEntityValue {
    pub resolved_value: String,
    pub tokens: Vec<String>,
}

impl TokenizedEntityValue {
    pub fn into_registered(self, is_injected: bool, rank: u32) -> RegisteredEntityValue {
        RegisteredEntityValue {
            resolved_value: self.resolved_value,
            tokens: self.tokens,
            is_injected,
            rank,
        }
    }
}

#[cfg(test)]
impl TokenizedEntityValue {
    pub fn new<T, U>(resolved_value: T, tokens: Vec<U>) -> Self
    where
        T: ToString,
        U: ToString,
    {
        Self {
            resolved_value: resolved_value.to_string(),
            tokens: tokens.into_iter().map(|t| t.to_string()).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RegisteredEntityValue {
    pub resolved_value: String,
    pub tokens: Vec<String>,
    pub is_injected: bool,
    pub rank: u32,
}

impl RegisteredEntityValue {
    pub fn new<T, U>(resolved_value: T, tokens: Vec<U>, is_injected: bool, rank: u32) -> Self
    where
        T: ToString,
        U: ToString,
    {
        Self {
            resolved_value: resolved_value.to_string(),
            tokens: tokens.into_iter().map(|t| t.to_string()).collect(),
            is_injected,
            rank,
        }
    }

    pub fn update_rank(mut self, new_rank: u32) -> Self {
        self.rank = new_rank;
        self
    }
}

impl RegisteredEntityValue {
    pub fn into_tokenized(self) -> TokenizedEntityValue {
        TokenizedEntityValue {
            resolved_value: self.resolved_value,
            tokens: self.tokens,
        }
    }
}

/// Struct holding a gazetteer, i.e. an ordered list of `EntityValue` to be added to the parser.
/// The values should be added in order of popularity or probability, with the most popular value
/// added first (see Parser).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Gazetteer {
    pub data: Vec<EntityValue>,
}

impl Serialize for Gazetteer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.data.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Gazetteer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let entity_values = <Vec<EntityValue>>::deserialize(deserializer)?;
        Ok(Gazetteer {
            data: entity_values,
        })
    }
}

impl Gazetteer {
    /// Add a single value to the Gazetteer
    pub fn add(&mut self, value: EntityValue) {
        self.data.push(value);
    }

    /// Extend the Gazetteer with the values of another Gazetteer
    pub fn extend(&mut self, gazetteer: Self) {
        self.data.extend(gazetteer.data)
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
    #[allow(clippy::non_canonical_partial_ord_impl)]
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
