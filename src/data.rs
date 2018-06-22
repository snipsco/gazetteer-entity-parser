use std::path::Path;
use std::fs::File;
use serde_json;

use errors::SnipsResolverResult;

#[derive(Debug)]
pub(crate) struct InternalEntityValue<'a> {
    pub(crate) weight: f32,
    pub(crate) resolved_value: &'a str,
    pub(crate) raw_value: &'a str
}

#[derive(Debug, Deserialize)]
pub struct EntityValue {
    pub resolved_value: String,
    pub raw_value: String
}

impl<'a> InternalEntityValue<'a> {
    pub(crate) fn new(entity_value: &'a EntityValue, rank: usize) -> InternalEntityValue {
        InternalEntityValue {
            resolved_value: &entity_value.resolved_value,
            raw_value: &entity_value.raw_value,
            weight: 1.0 - 1.0 / (1.0 + rank as f32)  // Adding 1 ensures rank is > 0
        }
    }
}

#[derive(Debug)]
pub struct Gazetteer {
    pub data: Vec<EntityValue>,
}

impl Gazetteer {

    pub fn add(&mut self, value: EntityValue) {
        self.data.push(value);
    }

    pub fn from_json(filename: &Path, limit: Option<usize>) -> SnipsResolverResult<Gazetteer> {
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
