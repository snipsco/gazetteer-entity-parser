use data::Gazetteer;
use errors::*;
use parser::Parser;
use EntityValue;

/// Struct exposing a builder allowing to configure and build a Parser
#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
pub struct ParserBuilder {
    gazetteer: Gazetteer,
    threshold: f32,
    n_gazetteer_stop_words: Option<usize>,
    additional_stop_words: Option<Vec<String>>,
}

impl Default for ParserBuilder {
    fn default() -> Self {
        ParserBuilder {
            gazetteer: Gazetteer::default(),
            threshold: 1.0,
            n_gazetteer_stop_words: None,
            additional_stop_words: None,
        }
    }
}

impl ParserBuilder {
    /// Define the gazetteer that the parser will use. This will replace any previously specified
    /// gazetteer.
    pub fn gazetteer(mut self, gazetteer: Gazetteer) -> Self {
        self.gazetteer = gazetteer;
        self
    }

    /// Add a gazetteer to the parser. This will extend any previously specified gazetteer.
    pub fn extend_with_gazetteer(mut self, gazetteer: Gazetteer) -> Self {
        if self.gazetteer.data.is_empty() {
            self.gazetteer = gazetteer;
        } else {
            self.gazetteer.extend(gazetteer);
        }
        self
    }

    /// Add a value to the parser's gazetteer
    pub fn add_value(mut self, entity_value: EntityValue) -> Self {
        self.gazetteer.add(entity_value);
        self
    }

    /// Set the minimum tokens ratio that will be used to filter entity matches.
    pub fn minimum_tokens_ratio(mut self, ratio: f32) -> Self {
        self.threshold = ratio;
        self
    }

    /// Set the desired number of stop words to be extracted from the gazetteer. Setting the
    /// number of stop words to `n` will ensure that the `n` most frequent words in the gazetteer
    /// will be considered as stop words
    pub fn n_stop_words(mut self, n: usize) -> Self {
        self.n_gazetteer_stop_words = Some(n);
        self
    }

    /// Set additional stop words manually
    pub fn additional_stop_words(mut self, asw: Vec<String>) -> Self {
        self.additional_stop_words = Some(asw);
        self
    }

    /// Instantiate a Parser from the ParserBuilder
    pub fn build(self) -> Result<Parser> {
        if self.threshold < 0.0 || self.threshold > 1.0 {
            return Err(
                format_err!("Invalid value for threshold ({}), it must be between 0.0 and 1.0",
                self.threshold))
        }
        let mut parser = self.gazetteer.data
            .into_iter()
            .enumerate()
            .fold(Parser::default(), |mut parser, (rank, entity_value)| {
                parser.add_value(entity_value, rank as u32);
                parser
            });
        parser.set_threshold(self.threshold);
        parser.set_stop_words(self.n_gazetteer_stop_words.unwrap_or(0),
                              self.additional_stop_words);
        Ok(parser)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::EntityValue;
    use serde_json;

    #[test]
    fn test_parser_builder_using_gazetteer() {
        // Given
        let entity_values = vec![
            EntityValue {
                resolved_value: "The Flying Stones".to_string(),
                resolved_value_id: None,
                raw_value: "the flying stones".to_string(),
            },
            EntityValue {
                resolved_value: "The Rolling Stones".to_string(),
                resolved_value_id: None,
                raw_value: "the rolling stones".to_string(),
            },
            EntityValue {
                resolved_value: "The Rolling Stones".to_string(),
                resolved_value_id: None,
                raw_value: "the stones".to_string(),
            }
        ];
        let gazetteer = Gazetteer { data: entity_values };
        let builder = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer.clone())
            .n_stop_words(2)
            .additional_stop_words(vec!["hello".to_string()]);

        // When
        let parser_from_builder = builder
            .build()
            .unwrap();

        // Then
        let mut expected_parser = Parser::default();
        for (rank, entity_value) in gazetteer.data.into_iter().enumerate() {
            expected_parser.add_value(entity_value, rank as u32);
        }
        expected_parser.set_threshold(0.5);
        expected_parser.set_stop_words(2, Some(vec!["hello".to_string()]));

        assert_eq!(expected_parser, parser_from_builder);
    }

    #[test]
    fn test_parser_builder_using_extended_gazetteer() {
        // Given
        let entity_values_1 = vec![
            EntityValue {
                resolved_value: "The Flying Stones".to_string(),
                resolved_value_id: None,
                raw_value: "the flying stones".to_string(),
            }
        ];

        let entity_values_2 = vec![
            EntityValue {
                resolved_value: "The Rolling Stones".to_string(),
                resolved_value_id: None,
                raw_value: "the rolling stones".to_string(),
            },
            EntityValue {
                resolved_value: "The Rolling Stones".to_string(),
                resolved_value_id: None,
                raw_value: "the stones".to_string(),
            }
        ];
        let gazetteer_1 = Gazetteer { data: entity_values_1.clone() };
        let gazetteer_2 = Gazetteer { data: entity_values_2.clone() };
        let builder = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .gazetteer(gazetteer_1)
            .extend_with_gazetteer(gazetteer_2)
            .n_stop_words(2)
            .additional_stop_words(vec!["hello".to_string()]);

        // When
        let parser_from_builder = builder
            .build()
            .unwrap();

        // Then
        let mut expected_parser = Parser::default();
        for (rank, entity_value) in entity_values_1.into_iter().enumerate() {
            expected_parser.add_value(entity_value, rank as u32);
        }
        for (rank, entity_value) in entity_values_2.into_iter().enumerate() {
            expected_parser.add_value(entity_value, 1 + rank as u32);
        }
        expected_parser.set_threshold(0.5);
        expected_parser.set_stop_words(2, Some(vec!["hello".to_string()]));

        assert_eq!(expected_parser, parser_from_builder);
    }

    #[test]
    fn test_parser_builder_using_values() {
        // Given
        let entity_values = vec![
            EntityValue {
                resolved_value: "The Flying Stones".to_string(),
                resolved_value_id: None,
                raw_value: "the flying stones".to_string(),
            },
            EntityValue {
                resolved_value: "The Rolling Stones".to_string(),
                resolved_value_id: None,
                raw_value: "the rolling stones".to_string(),
            },
            EntityValue {
                resolved_value: "The Rolling Stones".to_string(),
                resolved_value_id: None,
                raw_value: "the stones".to_string(),
            }
        ];
        let builder = ParserBuilder::default()
            .minimum_tokens_ratio(0.5)
            .add_value(entity_values[0].clone())
            .add_value(entity_values[1].clone())
            .add_value(entity_values[2].clone())
            .n_stop_words(2)
            .additional_stop_words(vec!["hello".to_string()]);

        // When
        let parser_from_builder = builder
            .build()
            .unwrap();

        // Then
        let mut expected_parser = Parser::default();
        for (rank, entity_value) in entity_values.into_iter().enumerate() {
            expected_parser.add_value(entity_value, rank as u32);
        }
        expected_parser.set_threshold(0.5);
        expected_parser.set_stop_words(2, Some(vec!["hello".to_string()]));

        assert_eq!(expected_parser, parser_from_builder);
    }

    #[test]
    fn test_serialization_deserialization() {
        let test_serialization_str = r#"{
  "gazetteer": [
    {
      "resolved_value": "yala",
      "raw_value": "yolo"
    },
    {
      "resolved_value": "Value With Id",
      "resolved_value_id": "42",
      "raw_value": "value with id"
    }
  ],
  "threshold": 0.6,
  "n_gazetteer_stop_words": 30,
  "additional_stop_words": [
    "hello",
    "world"
  ]
}"#;
        let mut gazetteer = Gazetteer::default();
        gazetteer.add(EntityValue {
            resolved_value: "yala".to_string(),
            resolved_value_id: None,
            raw_value: "yolo".to_string(),
        });
        gazetteer.add(EntityValue {
            resolved_value: "Value With Id".to_string(),
            resolved_value_id: Some("42".to_string()),
            raw_value: "value with id".to_string(),
        });
        let builder = ParserBuilder::default()
            .minimum_tokens_ratio(0.6)
            .gazetteer(gazetteer)
            .n_stop_words(30)
            .additional_stop_words(vec!["hello".to_string(), "world".to_string()]);

        // Deserialize builder from string and assert result
        let deserialized_builder: ParserBuilder =
            serde_json::from_str(test_serialization_str).unwrap();
        assert_eq!(deserialized_builder, builder);

        // Serialize builder to string and assert
        let serialized_builder = serde_json::to_string_pretty(&builder).unwrap();
        assert_eq!(serialized_builder, test_serialization_str);
    }
}
