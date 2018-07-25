#[macro_use]
extern crate criterion;
extern crate gazetteer_entity_parser;
extern crate rand;

use gazetteer_entity_parser::{EntityValue, Gazetteer, Parser};
use rand::distributions::Alphanumeric;
use rand::seq::sample_iter;
use rand::thread_rng;
use rand::Rng;

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
        }
    }

    fn generate(&mut self) -> String {
        let n_words = self.rng.gen_range(1, 10);
        let mut s: Vec<String> = vec![];
        for sample_idx in sample_iter(&mut self.rng, 0..self.unique_strings.len(), n_words).unwrap()
        {
            s.push(self.unique_strings.get(sample_idx).unwrap().to_string())
        }
        s.join(" ")
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rsg = RandomStringGenerator::new(10000);
    let mut gazetteer = Gazetteer::new();
    for _ in 1..150000 {
        let val = rsg.generate();
        gazetteer.add(EntityValue {
            raw_value: val.clone(),
            resolved_value: val.to_lowercase(),
        });
    }
    let parser = Parser::from_gazetteer(&gazetteer).unwrap();

    c.bench_function("Parse random value", move |b| {
        b.iter(|| parser.run(&rsg.generate(), 0.5))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
