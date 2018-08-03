#[macro_use]
extern crate criterion;
extern crate gazetteer_entity_parser;
extern crate rand;
extern crate mio_httpc;
extern crate serde_json;

use gazetteer_entity_parser::{EntityValue, Gazetteer, Parser};
use rand::distributions::Alphanumeric;
use rand::seq::sample_iter;
use rand::thread_rng;
use rand::Rng;
use std::collections::{HashSet};
use mio_httpc::CallBuilder;

use criterion::Criterion;

/// Function generating a random string representing a single word of various length
fn generate_random_string(rng: &mut rand::ThreadRng) -> String {
    let n_char = rng.gen_range(3, 8);
    rng.sample_iter(&Alphanumeric).take(n_char).collect()
}

/// Random string generator with a bit of redundancy to make it harder for the parser
struct RandomStringGenerator {
    unique_strings: Vec<String>,
    rng: rand::ThreadRng,
    already_generated: HashSet<String>
}

impl RandomStringGenerator {
    fn new(n_unique_strings: usize) -> RandomStringGenerator {
        let mut rng = thread_rng();
        let unique_strings = (0..n_unique_strings)
            .map(|_| generate_random_string(&mut rng))
            .collect();
        RandomStringGenerator {
            unique_strings,
            rng: rng,
            already_generated: HashSet::new()
        }
    }

    fn generate(&mut self, max_words: usize) -> String {
        loop {
            let n_words = self.rng.gen_range(1, max_words);
            let mut s: Vec<String> = vec![];
            for sample_idx in sample_iter(&mut self.rng, 0..self.unique_strings.len(), n_words).unwrap()
            {
                s.push(self.unique_strings.get(sample_idx).unwrap().to_string());
            }
            let value = s.join(" ");
            if !self.already_generated.contains(&value) {
                self.already_generated.insert(value.clone());
                break value
            }
        }
    }
}

fn artist_gazetteer(c: &mut Criterion) {
    // Real-world artist gazetteer
    // let (_, body) = CallBuilder::get().max_response(20000000).timeout_ms(60000).url("https://s3.amazonaws.com/snips/nlu-lm/test/gazetteer-entity-parser/artist_gazetteer_formatted.json").unwrap().exec().unwrap();
    // let data: Vec<EntityValue> = serde_json::from_reader(&*body).unwrap();
    // let gaz = Gazetteer{ data };
    // DEBUG
    let gaz = Gazetteer::from_json("local_testing/artist_gazetteer_formatted.json", None).unwrap();
    let fraction = 0.01;
    // DEBUG
    println!("FRACTION {:?}", fraction);
    let mut parser = Parser::from_gazetteer(&gaz).unwrap();
    parser.compute_stop_words(fraction);
    // DEBUG
    println!("STOP WORDS {:?}", parser.get_stop_words().unwrap());
    println!("EDGE CASES WORDS {:?}", parser.get_edge_cases().unwrap());
    println!("NUM STOP WORDS {:?}", parser.get_stop_words().unwrap().len());
    println!("NUM EDGE CASES {:?}", parser.get_edge_cases().unwrap().len());

    c.bench_function("Parse artist request - rolling stones - threhold 0.6", move |b| {
        b.iter(|| parser.run("I'd like to listen to some rolling stones", 0.6))
    });
    let mut parser = Parser::from_gazetteer(&gaz).unwrap();
    parser.compute_stop_words(fraction);

    c.bench_function("Parse artist request - the stones - threshold 0.6", move |b| {
        b.iter(|| parser.run("I'd like to listen to the stones", 0.6))
    });

}

fn album_gazetteer(c: &mut Criterion) {
    // Real-world albums gazetteer
    // let (_, body) = CallBuilder::get().max_response(20000000).timeout_ms(60000).url("https://s3.amazonaws.com/snips/nlu-lm/test/gazetteer-entity-parser/album_gazetteer_formatted.json").unwrap().exec().unwrap();
    // let data: Vec<EntityValue> = serde_json::from_reader(&*body).unwrap();
    // let gaz = Gazetteer{ data };
    // DEBUG
    let gaz = Gazetteer::from_json("local_testing/album_gazetteer_formatted.json", None).unwrap();
    let fraction = 0.01;
    // DEBUG
    println!("FRACTION {:?}", fraction);

    let mut parser = Parser::from_gazetteer(&gaz).unwrap();
    // parser.compute_stop_words(fraction);
    // DEBUG
    println!("STOP WORDS {:?}", parser.get_stop_words().unwrap());
    // println!("EDGE CASES WORDS {:?}", parser.get_edge_cases().unwrap());
    println!("STOP WORDS {:?}", parser.get_stop_words().unwrap().len());
    println!("EDGE CASES {:?}", parser.get_edge_cases().unwrap().len());

    // c.bench_function("Parse album request - black and white album - threhold 0.6", move |b| {
    //     b.iter(|| parser.run("Je veux écouter le black and white album", 0.6))
    // });

    // let mut parser = Parser::from_gazetteer(&gaz).unwrap();
    // parser.compute_stop_words(fraction);
    // DEBUG
    println!("PARSING: {:?}", parser.run("je veux écouter dark side of the moon", 0.6));
    c.bench_function("Parse album request - je veux ecouter dark side of the moon - threshold 0.6", move |b| {
        b.iter(|| parser.run("je veux écouter dark side of the moon", 0.6))
    });

    let mut parser = Parser::from_gazetteer(&gaz).unwrap();
    parser.compute_stop_words(fraction);
    println!("PARSING: {:?}", parser.run("je veux écouter dark side of the moon", 0.5));
    c.bench_function("Parse album request - je veux ecouter dark side of the moon - threshold 0.5", move |b| {
        b.iter(|| parser.run("je veux écouter dark side of the moon", 0.5))
    });

    let mut parser = Parser::from_gazetteer(&gaz).unwrap();
    parser.compute_stop_words(fraction);
    println!("PARSING: {:?}", parser.run("je veux écouter dark side of the moon", 0.7));
    c.bench_function("Parse album request - je veux ecouter dark side of the moon - threshold 0.7", move |b| {
        b.iter(|| parser.run("je veux écouter dark side of the moon", 0.7))
    });

    let mut parser = Parser::from_gazetteer(&gaz).unwrap();
    parser.compute_stop_words(fraction);
    // DEBUG
    println!("PARSING: {:?}", parser.run("je veux écouter dark side of the moon", 0.6));
    c.bench_function("Parse album request - the veux ecouter dark side of the moon - threshold 0.6", move |b| {
        b.iter(|| parser.run("the veux écouter dark side of the moon", 0.6))
    });
}

fn random_strings(c: &mut Criterion) {

    // Random gazetteer with low redundancy
    let mut rsg = RandomStringGenerator::new(10000);
    let mut gazetteer = Gazetteer::new();
    for _ in 1..150000 {
        let val = rsg.generate(10);
        gazetteer.add(EntityValue {
            raw_value: val.clone().to_lowercase(),
            resolved_value: val,
        });
    }
    let mut parser = Parser::from_gazetteer(&gazetteer).unwrap();
    parser.compute_stop_words(0.01);
    // DEBUG
    println!("STOP WORDS {:?}", parser.get_stop_words().unwrap().len());
    println!("EDGE CASES {:?}", parser.get_edge_cases().unwrap().len());

    c.bench_function("Parse random value - low redundancy", move |b| {
        b.iter(|| parser.run(&rsg.generate(10), 0.5))
    });

    // Random gazetteer with high redundancy
    let mut rsg = RandomStringGenerator::new(100);
    let mut gazetteer = Gazetteer::new();
    for _ in 1..100000 {
        let val = rsg.generate(5);
        gazetteer.add(EntityValue {
            raw_value: val.clone(),
            resolved_value: val.to_lowercase(),
        });
    }
    let mut parser = Parser::from_gazetteer(&gazetteer).unwrap();
    // parser.compute_stop_words(0.5);
    // // DEBUG
    // println!("STOP WORDS {:?}", parser.stop_words.len());
    // println!("EDGE CASES {:?}", parser.edge_cases.len());

    c.bench_function("Parse random value - high redundancy", move |b| {
        b.iter(|| parser.run(&rsg.generate(4), 0.6))
    });
}

// DEBUG
// criterion_group!(benches, random_strings, artist_gazetteer, album_gazetteer);
criterion_group!(benches, album_gazetteer);
criterion_main!(benches);
