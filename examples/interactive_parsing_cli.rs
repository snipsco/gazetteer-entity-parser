extern crate clap;
extern crate gazetteer_entity_parser;
extern crate serde_json;

use std::io::Write;
use std::{fs, io};

use clap::{App, Arg};

use gazetteer_entity_parser::{Gazetteer, Parser, ParserBuilder};

fn main() {
    let mut app = App::new("gazetteer-entity-parser-demo")
        .about("Interactive CLI for parsing gazetteer entities")
        .arg(
            Arg::with_name("parser")
                .short("p")
                .long("--parser")
                .takes_value(true)
                .help("path to the parser directory"),
        )
        .arg(
            Arg::with_name("gazetteer")
                .short("g")
                .long("--gazetteer")
                .takes_value(true)
                .help("path to the json gazetteer file"),
        )
        .arg(
            Arg::with_name("opt_nb_stop_words")
                .short("n")
                .long("--nb-stop-words")
                .takes_value(true)
                .help("number of stop words to use"),
        )
        .arg(
            Arg::with_name("opt_tokens_ratio")
                .short("r")
                .long("--ratio")
                .takes_value(true)
                .help("minimum tokens ratio for the parser"),
        )
        .arg(
            Arg::with_name("opt_max_alternatives")
                .short("a")
                .long("--alternatives")
                .takes_value(true)
                .help("maximum number of alternative resolved values"),
        );
    let matches = app.clone().get_matches();

    let opt_nb_stop_words = matches
        .value_of("opt_nb_stop_words")
        .map(|nb_str| nb_str.to_string().parse::<usize>().unwrap());

    let opt_tokens_ratio = matches
        .value_of("opt_tokens_ratio")
        .map(|ratio_str| ratio_str.to_string().parse::<f32>().unwrap());
    let max_alternatives = matches
        .value_of("opt_max_alternatives")
        .map(|max_str| max_str.to_string().parse::<usize>().unwrap())
        .unwrap_or(5);

    if let Some(parser) = matches
        .value_of("parser")
        .map(|parser_dir| {
            println!("\nLoading the parser...");
            let mut parser = Parser::from_folder(parser_dir).unwrap();
            if let Some(ratio) = opt_tokens_ratio {
                parser.set_threshold(ratio);
            };
            if let Some(nb_stop_words) = opt_nb_stop_words {
                parser.set_stop_words(nb_stop_words, None);
            };
            parser
        })
        .or_else(|| {
            matches.value_of("gazetteer").map(|gazetteer_path| {
                println!("\nLoading the gazetteer...");
                let gazetteer_file = fs::File::open(&gazetteer_path).unwrap();
                let gazetteer: Gazetteer = serde_json::from_reader(gazetteer_file).unwrap();

                println!("\nBuilding the parser...");
                ParserBuilder::default()
                    .gazetteer(gazetteer)
                    .n_stop_words(opt_nb_stop_words.unwrap_or(0))
                    .minimum_tokens_ratio(opt_tokens_ratio.unwrap_or(1.0))
                    .build()
                    .unwrap()
            })
        })
    {
        loop {
            print!("> ");
            io::stdout().flush().unwrap();
            let mut query = String::new();
            io::stdin().read_line(&mut query).unwrap();
            let result = parser.run(query.trim(), max_alternatives).unwrap();
            println!("{:?}", result);
        }
    } else {
        app.print_long_help().unwrap();
    }
}
