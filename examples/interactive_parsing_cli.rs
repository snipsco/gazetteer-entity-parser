use std::io;
use std::io::Write;

use clap::{App, Arg};

use gazetteer_entity_parser::Parser;

fn main() {
    let matches = App::new("gazetteer-entity-parser-demo")
        .about("Interactive CLI for parsing gazetteer entities")
        .arg(
            Arg::with_name("PARSER_DIR")
                .required(true)
                .takes_value(true)
                .index(1)
                .help("path to the parser directory"),
        )
        .get_matches();

    let parser_dir = matches.value_of("PARSER_DIR").unwrap();
    println!("\nLoading the parser...");
    let parser = Parser::from_folder(parser_dir).unwrap();
    loop {
        print!("> ");
        io::stdout().flush().unwrap();
        let mut query = String::new();
        io::stdin().read_line(&mut query).unwrap();
        let result = parser.run(query.trim()).unwrap();
        println!("{:?}", result);
    }
}
