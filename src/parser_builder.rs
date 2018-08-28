use data::Gazetteer;
use errors::*;
use parser::Parser;
use std::result::Result;

/// Struct exposing a builder allowing to configure and build a Parser
#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
pub struct ParserBuilder {
    gazetteer: Gazetteer,
    threshold: f32,
    n_gazetteer_stop_words: Option<usize>,
    additional_stop_words: Option<Vec<String>>,
}

impl ParserBuilder {
    /// Instantiate a new ParserBuilder with values for the non-optional attributes
    pub fn new(gazetteer: Gazetteer, threshold: f32) -> ParserBuilder {
        ParserBuilder {
            gazetteer,
            threshold,
            n_gazetteer_stop_words: None,
            additional_stop_words: None,
        }
    }

    /// Set the desired number of stop words to be extracted from the gazetteer. Setting the
    /// number of stop words to `n` will ensure that the `n` most frequent words in the gazetteer
    /// will be considered as stop words
    pub fn n_stop_words(mut self, n: usize) -> ParserBuilder {
        self.n_gazetteer_stop_words = Some(n);
        self
    }

    /// Set additional stop words manually
    pub fn additional_stop_words(mut self, asw: Vec<String>) -> ParserBuilder {
        self.additional_stop_words = Some(asw);
        self
    }

    /// Instantiate a Parser from the ParserBuilder
    pub fn build(self) -> Result<Parser, BuildError> {
        let mut parser = Parser::default();
        for (rank, entity_value) in self.gazetteer.data.into_iter().enumerate() {
            parser
                .add_value(entity_value, rank as u32)
                .map_err(|cause| BuildError {
                    cause: BuildRootError::AddValueError(cause),
                })?;
        }
        parser.set_threshold(self.threshold);
        if let Some(n) = self.n_gazetteer_stop_words {
            parser
                .set_stop_words(n, self.additional_stop_words)
                .map_err(|cause| BuildError {
                    cause: BuildRootError::SetStopWordsError(cause),
                })?;
        } else if let Some(_) = self.additional_stop_words {
            parser
                .set_stop_words(0, self.additional_stop_words)
                .map_err(|cause| BuildError {
                    cause: BuildRootError::SetStopWordsError(cause),
                })?;
        }
        Ok(parser)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::EntityValue;
    use serde_json;
    use serde_json::Value;

    #[test]
    fn test_parser_builder() {
        let mut gazetteer = Gazetteer::new();
        gazetteer.add(EntityValue {
            resolved_value: "The Flying Stones".to_string(),
            raw_value: "the flying stones".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the rolling stones".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "The Rolling Stones".to_string(),
            raw_value: "the stones".to_string(),
        });

        let parser_from_builder = ParserBuilder::new(gazetteer.clone(), 0.5)
            .n_stop_words(2)
            .additional_stop_words(vec!["hello".to_string()])
            .build()
            .unwrap();

        let mut parser_manual = Parser::default();
        for (rank, entity_value) in gazetteer.data.into_iter().enumerate() {
            parser_manual.add_value(entity_value, rank as u32).unwrap();
        }
        parser_manual.set_threshold(0.5);
        parser_manual
            .set_stop_words(2, Some(vec!["hello".to_string()]))
            .unwrap();

        assert_eq!(parser_from_builder, parser_manual);
    }

    #[test]
    fn test_serialization_deserialization() {
        let test_serialization_str = r#"
        {
            "gazetteer": [
                {
                    "raw_value": "yolo",
                    "resolved_value":"yala"
                }
            ],
            "threshold": 0.6,
            "n_gazetteer_stop_words": 30,
            "additional_stop_words": ["hello", "world"]
        }
        "#;

        let mut gazetteer = Gazetteer::new();
        gazetteer.add(EntityValue {
            resolved_value: "yala".to_string(),
            raw_value: "yolo".to_string(),
        });
        let builder = ParserBuilder::new(gazetteer, 0.6)
            .n_stop_words(30)
            .additional_stop_words(vec!["hello".to_string(), "world".to_string()]);

        // Deserialize builder from string and assert result
        let deserialized_builder: ParserBuilder =
            serde_json::from_str(test_serialization_str).unwrap();
        assert_eq!(deserialized_builder, builder);

        // Serialize builder to string and assert
        let serialized_builder: Value =
            serde_json::from_str(&serde_json::to_string(&builder).unwrap()).unwrap();
        let ground_true_serialized_builder: Value =
            serde_json::from_str(test_serialization_str).unwrap();
        assert_eq!(serialized_builder, ground_true_serialized_builder);
    }
}
