#[macro_use]
extern crate criterion;
extern crate dinghy_test;
extern crate gazetteer_entity_parser;
extern crate rand;
extern crate serde_json;

use criterion::Criterion;
use gazetteer_entity_parser::*;
use rand::distributions::Alphanumeric;
use rand::rngs::ThreadRng;
use rand::seq::IteratorRandom;
use rand::thread_rng;
use rand::Rng;
use std::collections::HashSet;

pub fn test_data_path() -> ::std::path::PathBuf {
    ::dinghy_test::try_test_file_path("data").unwrap_or_else(|| "data".into())
}

/// Function generating a random string representing a single word of various length
fn generate_random_string(rng: &mut ThreadRng) -> String {
    let n_char = rng.gen_range(3, 8);
    rng.sample_iter(&Alphanumeric).take(n_char).collect()
}

/// Random string generator with tunable redundancy to make it harder for the parser
#[derive(Clone)]
struct RandomStringGenerator {
    vocabulary: Vec<String>,
    max_words: usize,
    rng: ThreadRng,
    already_generated: HashSet<String>,
}

impl RandomStringGenerator {
    fn new(vocab_size: usize, max_words: usize) -> RandomStringGenerator {
        let mut rng = thread_rng();
        let unique_strings = (0..vocab_size)
            .map(|_| generate_random_string(&mut rng))
            .collect();
        RandomStringGenerator {
            vocabulary: unique_strings,
            max_words,
            rng,
            already_generated: HashSet::new(),
        }
    }
}

impl Iterator for RandomStringGenerator {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        loop {
            let n_words = self.rng.gen_range(1, self.max_words);
            let generated_value = self
                .vocabulary
                .iter()
                .choose_multiple(&mut self.rng, n_words)
                .iter()
                .map(|sample_string| sample_string.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            if !self.already_generated.contains(&generated_value) {
                self.already_generated.insert(generated_value.clone());
                break Some(generated_value);
            }
        }
    }
}

fn generate_random_gazetteer(
    vocab_size: usize,
    nb_entity_values: usize,
    max_words: usize,
) -> (Gazetteer, RandomStringGenerator) {
    let rsg = RandomStringGenerator::new(vocab_size, max_words);
    let entity_values = rsg
        .clone()
        .take(nb_entity_values)
        .map(|string| EntityValue {
            resolved_value: string.to_lowercase(),
            raw_value: string,
        })
        .collect();
    let gazetteer = Gazetteer {
        data: entity_values,
    };
    (gazetteer, rsg)
}

fn generate_random_parser(
    vocab_size: usize,
    nb_entity_values: usize,
    max_words: usize,
    minimum_tokens_ratio: f32,
    n_stop_words: usize,
) -> (Parser, RandomStringGenerator) {
    let (gazetteer, rsg) = generate_random_gazetteer(vocab_size, nb_entity_values, max_words);
    let parser = ParserBuilder::default()
        .gazetteer(gazetteer)
        .minimum_tokens_ratio(minimum_tokens_ratio)
        .n_stop_words(n_stop_words)
        .build()
        .unwrap();
    (parser, rsg)
}

fn get_low_redundancy_parser() -> (Parser, RandomStringGenerator) {
    generate_random_parser(10000, 100000, 10, 0.5, 50)
}

fn get_high_redundancy_parser() -> (Parser, RandomStringGenerator) {
    generate_random_parser(100, 100000, 5, 0.5, 50)
}

fn parsing_low_redundancy(c: &mut Criterion) {
    let (parser, mut rsg) = get_low_redundancy_parser();
    c.bench_function("Parse random value - low redundancy", move |b| {
        b.iter(|| parser.run(&rsg.next().unwrap(), 10))
    });
}

fn parsing_high_redundancy(c: &mut Criterion) {
    let (parser, mut rsg) = get_high_redundancy_parser();
    c.bench_function("Parse random value - high redundancy", move |b| {
        b.iter(|| parser.run(&rsg.next().unwrap(), 10))
    });
}

fn loading(c: &mut Criterion) {
    let (gazetteer, _) = generate_random_gazetteer(100, 1000, 5);
    let parser_directory = test_data_path().join("benches").join("parser");
    if !parser_directory.exists() {
        let parser = ParserBuilder::default()
            .gazetteer(gazetteer)
            .minimum_tokens_ratio(0.5)
            .n_stop_words(50)
            .build()
            .unwrap();

        parser.dump(&parser_directory).unwrap();
    }
    c.bench_function(
        "Loading random gazetteer parser with low redundancy",
        move |b| b.iter(|| Parser::from_folder(parser_directory.clone()).unwrap()),
    );
}

criterion_group!(
    benches,
    parsing_low_redundancy,
    parsing_high_redundancy,
    loading
);
criterion_main!(benches);
