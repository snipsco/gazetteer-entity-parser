#[macro_use]
extern crate criterion;
extern crate nr_builtin_resolver;
extern crate rand;
extern crate serde_json;
// extern crate itertools;

// use itertools::Itertools;

use rand::thread_rng;
use rand::Rng;
// use rand::prelude::*;
use nr_builtin_resolver::data::{EntityValue, Gazetteer};
use nr_builtin_resolver::resolver::{ResolvedValue, Resolver};
use std::path::Path;

use criterion::Criterion;

// fn generate_random_string() -> String {
//     let mut rng = thread_rng();
//     let n_words = rng.gen_range(1, 4);
//     // let n_char = rng.gen_range(3, 8);
//     // (1..n_char).map(|_| rng.gen::<char>()).collect()
//     (1..n_words + 1).map(|_| {
//         let n_char = rng.gen_range(3, 8);
//         rng.gen_ascii_chars().take(n_char).collect()
//     }).collect::<Vec<String>>().join(" ")
// }

fn criterion_benchmark(c: &mut Criterion) {
    let mut gazetteer = Gazetteer::from_json(
        Path::new("/Users/alaasaade/Documents/nr-builtin-resolver/local_testing/artist_gazeteer_formatted.json"),
        Some(100000)).unwrap();
    gazetteer.add(EntityValue {
                resolved_value: "Jacques".to_string(),
                raw_value: "jacques".to_string(),
            });
    let resolver = Resolver::from_gazetteer(&gazetteer, 0.5).unwrap();
    // gazetteer = Gazetteer { data: vec!() }
    assert_eq!(
        resolver.run("je veux écouter jacques").unwrap(),
        vec![ResolvedValue {
            raw_value: "jacques".to_string(),
            resolved_value: "Jacques".to_string(),
            range: 16..23,
        }]
    );

    assert_eq!(
        resolver.run("je veux écouter brel").unwrap(),
        vec![ResolvedValue {
            raw_value: "brel".to_string(),
            resolved_value: "Jacques Brel".to_string(),
            range: 16..20,
        }]
    );

    assert_eq!(
        resolver.run("je veux ecouter les rolling stones").unwrap(),
        vec![
            ResolvedValue {
                resolved_value: "Les Enfoirés".to_string(),
                range: 16..19,
                raw_value: "les".to_string(),
            },
            ResolvedValue {
                raw_value: "rolling stones".to_string(),
                resolved_value: "The Rolling Stones".to_string(),
                range: 20..34,
            },
        ]
    );

    // assert_eq!(resolver.run("veux ecouter brel".to_string()).unwrap(), "<skip> <skip> Jacques_Brel");
    // assert_eq!(resolver.run("the rolling".to_string()).unwrap(), "<skip> The_Rolling_Stones");
    // assert_eq!(resolver.run("the stones".to_string()).unwrap(), "The Stones");
    // for _idx in 1..10 {
    //     println!("{:?}", resolver.run("the stones".to_string()).unwrap());
    // }
    let resolver_1 = Resolver::from_gazetteer(&gazetteer, 0.5).unwrap();
    c.bench_function("Resolve je veux ecouter les stones", move |b| {
        b.iter(|| resolver_1.run("je veux ecouter les stones"))
    });
    let resolver_1 = Resolver::from_gazetteer(&gazetteer, 0.5).unwrap();
    c.bench_function("Resolve the stones", move |b| {
        b.iter(|| resolver_1.run("ecouter the stones"))
    });
    let resolver_2 = Resolver::from_gazetteer(&gazetteer, 0.5).unwrap();
    c.bench_function("Resolve the rolling", move |b| {
        b.iter(|| resolver_2.run("the rolling"))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
