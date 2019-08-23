Gazetteer Entity Parser
=======================

.. image:: https://travis-ci.org/snipsco/gazetteer-entity-parser.svg?branch=master
   :target: https://travis-ci.org/snipsco/gazetteer-entity-parser

This Rust library allows to parse and resolve entity values based on a gazetteer, in the context of
an `Information Extraction <https://en.wikipedia.org/wiki/Information_extraction>`_ task.

Example
-------

.. code-block:: rust

    extern crate gazetteer_entity_parser;

    use gazetteer_entity_parser::*;

    fn main() {
        let gazetteer = gazetteer!(
            ("king of pop", "Michael Jackson"),
            ("the rolling stones", "The Rolling Stones"),
            ("the beatles", "The Beatles"),
            ("queen of soul", "Aretha Franklin"),
            ("the red hot chili peppers", "The Red Hot Chili Peppers"),
        );
        let parser = ParserBuilder::default()
            .gazetteer(gazetteer)
            .minimum_tokens_ratio(2. / 3.)
            .build()
            .unwrap();

        let sentence = "My favourite artists are the stones and the amazing fab four";
        let extracted_entities = parser.run(sentence).unwrap();
        assert_eq!(extracted_entities,
                   vec![
                       ParsedValue {
                           raw_value: "the stones".to_string(),
                           matched_value: "the rolling stones".to_string(),
                           resolved_value: "The Rolling Stones".to_string(),
                           range: 25..35,
                       },
                       ParsedValue {
                           raw_value: "fab four".to_string(),
                           matched_value: "the fab four".to_string(),
                           resolved_value: "The Beatles".to_string(),
                           range: 52..60,
                       }]);
    }


License
-------

Licensed under either of
 * Apache License, Version 2.0 (`LICENSE-APACHE <LICENSE-APACHE>`_ or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license (`LICENSE-MIT <LICENSE-MIT>`_) or http://opensource.org/licenses/MIT)
at your option.

Contribution
------------

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
