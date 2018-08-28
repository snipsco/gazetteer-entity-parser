use std::result::Result;

use serde::Deserializer;
use serde::Serializer;
use serde::{Deserialize, Serialize};


/// Struct representing the value of an entity to be added to the parser
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct EntityValue {
    pub resolved_value: String,
    pub raw_value: String,
}

/// Struct holding a gazetteer, i.e. an ordered list of `EntityValue` to be added to the parser.
/// The values should be added in order of popularity or probability, with the most popular value
/// added first (see Parser).
#[derive(Debug, Clone, PartialEq)]
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
        Ok(Gazetteer { data: entity_values })
    }
}

impl Gazetteer {
    /// Instanciate a new empty gazetteer
    pub fn new() -> Gazetteer {
        Gazetteer { data: Vec::new() }
    }

    /// Add a single value to the Gazetteer
    pub fn add(&mut self, value: EntityValue) {
        self.data.push(value);
    }
}
