use std::fs::File;
use std::path::Path;

use failure::ResultExt;
use serde_json;

use errors::GazetteerParserResult;

/// Struct representing the value of an entity to be added to the parser
#[derive(Debug, Deserialize, Clone)]
pub struct EntityValue {
    pub resolved_value: String,
    pub raw_value: String,
}

/// Struct holding a gazetteer, i.e. an ordered list of `EntityValue` to be added to the parser.
/// The values should be added in order of popularity or probability, with the most popular value
/// added first (see Parser).
#[derive(Debug)]
pub struct Gazetteer {
    pub data: Vec<EntityValue>,
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

    /// Instanciate a Gazetteer from a json file containing an ordered list of entries
    /// of the form:
    ///
    /// ```json
    /// {
    ///     "raw_value": "the strokes",
    ///     "resolved_value": "The Strokes"
    /// }
    /// ```
    pub fn from_json<P: AsRef<Path>>(
        filename: P,
        limit: Option<usize>,
    ) -> GazetteerParserResult<Gazetteer> {
        let file = File::open(filename.as_ref())
            .with_context(|_| format!("Cannot open gazetter file {:?}", filename.as_ref()))?;
        let mut data: Vec<EntityValue> = serde_json::from_reader(file)?;
        match limit {
            None => (),
            Some(0) => Err(format_err!("limit should be > 0"))?,
            Some(value) => data.truncate(value),
        };
        Ok(Gazetteer { data })
    }
}
