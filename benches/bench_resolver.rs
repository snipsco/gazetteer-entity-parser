#[macro_use]
extern crate criterion;
extern crate nr_builtin_resolver;
extern crate rand;
// extern crate itertools;

// use itertools::Itertools;

use rand::Rng;
use rand::thread_rng;
// use rand::prelude::*;
use nr_builtin_resolver::data::{ EntityValue, Gazetteer };
use nr_builtin_resolver::resolver::Resolver;
use std::path::Path;

use criterion::Criterion;


fn generate_random_string() -> String {
    let mut rng = thread_rng();
    let n_words = rng.gen_range(1, 4);
    // let n_char = rng.gen_range(3, 8);
    // (1..n_char).map(|_| rng.gen::<char>()).collect()
    (1..n_words + 1).map(|_| {
        let n_char = rng.gen_range(3, 8);
        rng.gen_ascii_chars().take(n_char).collect()
    }).collect::<Vec<String>>().join(" ")
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut gazetteer = Gazetteer { data: Vec::new() };
    gazetteer.add(EntityValue {
        weight: 1.0,
        raw_value: "The Rolling Stones".to_string(),
        verbalized_value: "the rolling stones".to_string()
    });
    for _ in 1..10000 {
        let name = generate_random_string();
        // println!("{:?}", name);
        let verbalized = name.to_lowercase();
        gazetteer.add(EntityValue {
            weight: 1.0,
            raw_value: name,
            verbalized_value: verbalized,
        });
    }
    // println!("{:#?}", gazetteer);
    let resolver = Resolver::from_gazetteer(&gazetteer).unwrap();
    // resolver.symbol_table.write_file(Path::new("bench_symt"), false).unwrap();
    assert_eq!(resolver.run("the stones".to_string()).unwrap(), "The Rolling Stones");
    for _idx in 1..10 {
        println!("{:?}", resolver.run("the stones".to_string()).unwrap());
    }
    c.bench_function("Resolve the stones", move |b| b.iter(|| resolver.run("the stones".to_string())));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
