extern crate gazetteer_entity_parser;

use gazetteer_entity_parser::*;

fn main() {
    let parser = ParserBuilder::default()
        .add_value(EntityValue {
            raw_value: "king of pop".to_string(),
            resolved_value: "Michael Jackson".to_string(),
        })
        .add_value(EntityValue {
            raw_value: "the rolling stones".to_string(),
            resolved_value: "The Rolling Stones".to_string(),
        })
        .add_value(EntityValue {
            raw_value: "the fab four".to_string(),
            resolved_value: "The Beatles".to_string(),
        })
        .add_value(EntityValue {
            raw_value: "queen of soul".to_string(),
            resolved_value: "Aretha Franklin".to_string(),
        })
        .add_value(EntityValue {
            raw_value: "the red hot chili peppers".to_string(),
            resolved_value: "The Red Hot Chili Peppers".to_string(),
        })
        .minimum_tokens_ratio(2. / 3.)
        .build()
        .unwrap();

    let sentence = "My favourite artists are the stones and the fab four";
    let extracted_entities = parser.run(sentence).unwrap();
    assert_eq!(extracted_entities,
               vec![
                   ParsedValue {
                       raw_value: "the stones".to_string(),
                       resolved_value: "The Rolling Stones".to_string(),
                       range: 25..35,
                   },
                   ParsedValue {
                       raw_value: "the fab four".to_string(),
                       resolved_value: "The Beatles".to_string(),
                       range: 40..52,
                   }]);
}
