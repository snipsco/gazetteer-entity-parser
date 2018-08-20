use parser::Parser;
use data::Gazetteer;
use errors::GazetteerParserResult;


/// Struct exposing a builder allowing to configure and build a Parser
pub struct ParserBuilder<'a> {
    gazetteer: &'a Gazetteer,
    threshold: f32,
    n_gazetteer_stop_words: Option<usize>,
    additional_stop_words: Option<Vec<&'a str>>
}

impl<'a> ParserBuilder<'a> {

    /// Instantiate a new ParserBuilder with values for the non-optional attributes
    pub fn new(gazetteer: &Gazetteer, threshold: f32) -> ParserBuilder {
        ParserBuilder {
            gazetteer,
            threshold,
            n_gazetteer_stop_words: None,
            additional_stop_words: None
        }
    }

    /// Set the desired number of stop words to be extracted from the gazetteer. Setting the
    /// number of stop words to `n` will ensure that the `n` most frequent words in the gazetteer
    /// will be considered as stop words
    pub fn n_stop_words(&mut self, n: usize) -> &'a mut ParserBuilder {
        self.n_gazetteer_stop_words = Some(n);
        self
    }

    /// Set additional stop words manually. In order for this method to be used, the number of
    /// stop words to be extracted from the gazetteer should also be set using the `n_stop_words`
    /// method
    pub fn additional_stop_words(&mut self, asw: Vec<&'a str>) -> &'a mut ParserBuilder {
        self.additional_stop_words = Some(asw);
        self
    }

    /// Instanciate a Parser from the ParserBuilder
    pub fn build(&self) -> GazetteerParserResult<Parser> {
        let mut parser = Parser::from_gazetteer(&self.gazetteer)?;
        parser.set_threshold(self.threshold);
        if let Some(n) = &self.n_gazetteer_stop_words {
            parser.set_stop_words(*n, self.additional_stop_words.clone())?;
        } else if let Some(_) = &self.additional_stop_words {
            bail!("The `n_stop_words` attribute needs to be set using the `n_stop_words` method")
        }
        Ok(parser)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::EntityValue;

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

        let parser_from_builder = ParserBuilder::new(&gazetteer, 0.5)
            .n_stop_words(2)
            .additional_stop_words(vec!["hello"]).build().unwrap();

        let mut parser_from_gaz = Parser::from_gazetteer(&gazetteer).unwrap();
        parser_from_gaz.set_threshold(0.5);
        parser_from_gaz.set_stop_words(2, Some(vec!["hello"])).unwrap();

        assert_eq!(parser_from_builder, parser_from_gaz);
    }
}
