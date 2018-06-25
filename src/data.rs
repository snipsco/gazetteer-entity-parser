use std::path::Path;
use std::fs::File;
use serde_json;

use errors::SnipsResolverResult;

#[derive(Debug)]
pub struct InternalEntityValue<'a> {
    pub weight: f32,
    pub resolved_value: &'a str,
    pub raw_value: &'a str
}

/// Struct representing the value of an entity to be added to the resolver
#[derive(Debug, Deserialize)]
pub struct EntityValue {
    pub resolved_value: String,
    pub raw_value: String
}

impl<'a> InternalEntityValue<'a> {
    pub fn new(entity_value: &'a EntityValue, rank: usize) -> InternalEntityValue {
        InternalEntityValue {
            resolved_value: &entity_value.resolved_value,
            raw_value: &entity_value.raw_value,
            weight: 1.0 - 1.0 / (1.0 + rank as f32)  // Adding 1 ensures rank is > 0
        }
    }
}

/// Struct holding a gazetteer, i.e. an ordered list of entity values to be added to the resolver. The values should be added in order of popularity or probability, with the most popular value added first (see Resolver).
#[derive(Debug)]
pub struct Gazetteer {
    pub data: Vec<EntityValue>,
}

impl Gazetteer {

    /// Add a single value to the Gazetteer
    pub fn add(&mut self, value: EntityValue) {
        self.data.push(value);
    }

    /// Instanciate a Gazetteer from a json file containing an ordered list of entries of the form:
    /// ```
    /// {
    ///     "raw_value": "the strokes",
    ///     "resolved_value": "The Strokes"
    /// }
    /// ```
    pub fn from_json<P: AsRef<Path>>(filename: P, limit: Option<usize>) -> SnipsResolverResult<Gazetteer> {
        let file = File::open(filename)?;
        let mut data: Vec<EntityValue> = serde_json::from_reader(file)?;
        match limit {
            None => (),
            Some(value) => {
                if value == 0 {
                    panic!("limit should be > 0")
                }
                data.truncate(value);
            }
        };
        Ok(Gazetteer{data})
    }
}
