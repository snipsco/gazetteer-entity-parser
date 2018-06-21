#[macro_use]
extern crate criterion;
extern crate nr_builtin_resolver;
extern crate rand;
extern crate serde_json;
// extern crate itertools;

// use itertools::Itertools;

use std::fs::File;
use rand::Rng;
use rand::thread_rng;
// use rand::prelude::*;
use nr_builtin_resolver::data::{ EntityValue, Gazetteer };
use nr_builtin_resolver::resolver::{Resolver, ResolvedValue};
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
    let mut gazetteer = Gazetteer { data: Vec::new() };
    let file = File::open("/Users/alaasaade/Documents/snips-grammars/snips_grammars/resources/fr/music/artist.json").unwrap();
    let mut data: Vec<String> = serde_json::from_reader(file).unwrap();
    // for idx in 1..10 {
    //     println!("{:?}", data.get(idx));
    // }
    data.truncate(100000);
    for val in data {
        // println!("{:?}", val);
        // if val == "The Stones" {
        //     println!("{:?}", val);
        // }
        gazetteer.add(EntityValue {
            weight: 1.0,
            resolved_value: val.clone(),
            raw_value: val.clone().to_lowercase()
        })
    }
    // gazetteer.add(EntityValue {
    //     weight: 1.0,
    //     resolved_value: "The Rolling Stones".to_string(),
    //     raw_value: "the rolling stones".to_string()
    // });
    // gazetteer.add(EntityValue {
    //     weight: 1.0,
    //     resolved_value: "The Flying Stones".to_string(),
    //     raw_value: "the flying stones".to_string()
    // });
    // for _ in 1..10000 {
    //     let name = generate_random_string();
    //     // println!("{:?}", name);
    //     let verbalized = name.to_lowercase();
    //     gazetteer.add(EntityValue {
    //         weight: 1.0,
    //         resolved_value: name,
    //         raw_value: verbalized,
    //     });
    // }
    // println!("{:#?}", gazetteer);
    let resolver = Resolver::from_gazetteer(&gazetteer, 0.5).unwrap();
    // resolver.symbol_table.write_file(Path::new("bench_symt"), false).unwrap();
    // resolver.fst.write_file(Path::new("bench_fst")).unwrap();
    // assert_eq!(resolver.run("veux ecouter rolling stones").unwrap(),             vec!(
    //     ResolvedValue{ raw_value: "rolling stones".to_string(), resolved_value: "The Rolling Stones".to_string(), range: 13..27}));

        assert_eq!(
            resolver.run("je veux ecouter les rolling stones").unwrap(),
            vec!(
                ResolvedValue{ raw_value: "je".to_string(), resolved_value: "Je Suis Animal".to_string(), range: 0..2},
                ResolvedValue { resolved_value: "Les Paul".to_string(), range: 16..19, raw_value: "les".to_string() },
                ResolvedValue{ raw_value: "rolling stones".to_string(), resolved_value: "The Rolling Stones".to_string(), range: 20..34}
            )
        );

    // assert_eq!(resolver.run("veux ecouter brel".to_string()).unwrap(), "<skip> <skip> Jacques_Brel");
    // assert_eq!(resolver.run("the rolling".to_string()).unwrap(), "<skip> The_Rolling_Stones");
    // assert_eq!(resolver.run("the stones".to_string()).unwrap(), "The Stones");
    // for _idx in 1..10 {
    //     println!("{:?}", resolver.run("the stones".to_string()).unwrap());
    // }
    let resolver_1 = Resolver::from_gazetteer(&gazetteer, 0.1).unwrap();
    c.bench_function("Resolve je veux ecouter les stones", move |b| b.iter(|| resolver_1.run("je veux ecouter les stones")));
    let resolver_1 = Resolver::from_gazetteer(&gazetteer, 0.5).unwrap();
    c.bench_function("Resolve the stones", move |b| b.iter(|| resolver_1.run("ecouter the stones")));
    let resolver_2 = Resolver::from_gazetteer(&gazetteer, 0.5).unwrap();
    c.bench_function("Resolve the rolling", move |b| b.iter(|| resolver_2.run("the rolling")));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
