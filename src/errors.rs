use parser::PossibleMatch;
use rmps;
use serde_json;
use std::io;
use std::path::PathBuf;


#[derive(Debug, Fail, Clone)]
#[fail(
    display = "Symbol {} is already present several times in the symbol table, cannot
    determine which index to return. If this error is raised when adding a symbol, you may
    want to try to force add the symbol.",
    symbol
)]
pub struct DuplicateSymbolError { pub symbol: String }

#[derive(Debug, Fail, Clone)]
pub enum SymbolTableAddSymbolError {
    #[fail(display = "Key {} missing from symbol table", key)]
    MissingKeyError { key: String },
    #[fail(display="")]
    DuplicateSymbolError(DuplicateSymbolError),
}

#[derive(Debug, Fail, Clone)]
pub enum SymbolTableFindSingleSymbolError {
    #[fail(display = "Key {} missing from symbol table", key)]
    MissingKeyError { key: String },
    #[fail(display = "Caused by: ")]
    DuplicateSymbolError(DuplicateSymbolError),
}

#[derive(Debug, Fail, Clone)]
pub enum SymbolTableFindIndexError {
    #[fail(display = "Index {} missing from symbol table", key)]
    MissingKeyError { key: u32 },
}

#[derive(Debug, Fail, Clone)]
pub enum TokensFromResolvedValueError {
    #[fail(display = "Key {} missing from tokens to resolved value table", key)]
    MissingKeyError { key: u32 },
}

#[derive(Debug, Fail, Clone)]
pub enum ResolvedValuesFromTokenError {
    #[fail(display = "Key {} missing from tokens to resolved value table", key)]
    MissingKeyError { key: u32 },
}

#[derive(Debug)]
pub enum AddValueErrorKind {
    ResolvedValue,
    RawValue,
}

#[derive(Debug, Fail)]
#[fail(display = "Failed to add value of kind {:?} to the parser: {:?}", kind, value)]
pub struct AddValueError {
    pub value: String,
    pub kind: AddValueErrorKind,
    #[cause]
    pub cause: SymbolTableAddSymbolError,
}

#[derive(Debug, Fail)]
#[fail(display = "Failed to set stop words and edge cases")]
pub struct SetStopWordsError {
    #[cause]
    pub cause: SymbolTableAddSymbolError,
}

#[derive(Debug, Fail)]
#[fail(display = "Failed to get stop words")]
pub struct GetStopWordsError {
    #[cause]
    pub cause: SymbolTableFindIndexError,
}

#[derive(Debug, Fail)]
#[fail(display = "Failed to get edge cases")]
pub struct GetEdgeCasesError {
    #[cause]
    pub cause: SymbolTableFindIndexError,
}

#[derive(Debug, Fail)]
pub enum InjectionRootError {
    #[fail(display = "Caused by: ")]
    TokensFromResolvedValueError(#[cause] TokensFromResolvedValueError),
    #[fail(display = "Caused by: ")]
    ResolvedValuesFromTokenError(#[cause] ResolvedValuesFromTokenError),
    #[fail(display = "Caused by: ")]
    SymbolTableFindIndexError(#[cause] SymbolTableFindIndexError),
    #[fail(display = "Caused by: ")]
    AddValueError(#[cause] AddValueError),
    #[fail(display = "Caused by: ")]
    SetStopWordsError(#[cause] SetStopWordsError),
}

#[derive(Debug, Fail)]
#[fail(display = "Failed to inject new values in Parser")]
pub struct InjectionError {
    #[cause]
    pub cause: InjectionRootError,
}

#[derive(Debug, Fail)]
pub enum FindPossibleMatchRootError {
    #[fail(display = "Tokens list {:?} should contain value {} but doesn't", token_list, value)]
    MissingTokenFromList { token_list: Vec<u32>, value: u32 },
    #[fail(display = "Caused by: ")]
    PossibleMatchRootError(#[cause] PossibleMatchRootError),
    #[fail(display = "Caused by: ")]
    SymbolTableFindSingleSymbolError(#[cause] SymbolTableFindSingleSymbolError),
    #[fail(display = "Caused by: ")]
    ResolvedValuesFromTokenError(#[cause] ResolvedValuesFromTokenError),
}

#[derive(Debug, Fail)]
#[fail(display = "Error finding possible matches")]
pub struct FindPossibleMatchError {
    #[cause]
    pub cause: FindPossibleMatchRootError,
}

#[derive(Debug, Fail)]
#[fail(display = "Error parsing input")]
pub struct ParseInputError {
    #[cause]
    pub cause: SymbolTableFindIndexError,
}

#[derive(Debug, Fail)]
pub enum RunRootError {
    #[fail(display = "Caused by: ")]
    ParseInputError(#[cause] ParseInputError),
    #[fail(display = "Caused by: ")]
    FindPossibleMatchError(#[cause] FindPossibleMatchError),
}

#[derive(Debug, Fail)]
#[fail(display = "Error running parser on input string {}", input)]
pub struct RunError {
    pub input: String,
    #[cause]
    pub cause: RunRootError,
}

#[derive(Debug, Fail)]
pub enum GetParserConfigRootError {
    #[fail(display = "Caused by: ")]
    GetStopWordsError(#[cause] GetStopWordsError),
    #[fail(display = "Caused by: ")]
    GetEdgeCasesError(#[cause] GetEdgeCasesError),
}

#[derive(Debug, Fail)]
#[fail(display = "Error getting parser config")]
pub struct GetParserConfigError {
    #[cause]
    pub cause: GetParserConfigRootError,
}

#[derive(Debug, Fail)]
pub enum DumpRootError {
    #[fail(display = "Caused by: ")]
    GetParserConfigError(#[cause] GetParserConfigError),
    #[fail(display = "Caused by: ")]
    SerializationError(#[cause] SerializationError),
}

#[derive(Debug, Fail)]
#[fail(display = "Error dumping parser")]
pub struct DumpError {
    #[cause]
    pub cause: DumpRootError,
}

#[derive(Debug, Fail)]
#[fail(display = "Error loading parser")]
pub struct LoadError {
    #[cause]
    pub cause: DeserializationError,
}

#[derive(Debug, Fail)]
pub enum BuildRootError {
    #[fail(display = "Caused by: ")]
    AddValueError(#[cause] AddValueError),
    #[fail(display = "Caused by: ")]
    SetStopWordsError(#[cause] SetStopWordsError),
}

#[derive(Debug, Fail)]
#[fail(display = "Error building parser")]
pub struct BuildError {
    #[cause]
    pub cause: BuildRootError,
}

#[derive(Debug, Fail)]
#[fail(display = "Error loading gazetteer")]
pub struct GazetteerLoadingError {
    #[cause]
    pub cause: DeserializationError,
}

/// Low-level errors

#[derive(Debug, Fail)]
pub enum PossibleMatchRootError {
    #[fail(
        display = "Possible match consumed more tokens than are available: {:?}", possible_match
    )]
    PossibleMatchConsumedError { possible_match: PossibleMatch },
    #[fail(
        display = "Possible match skipped more tokens than are available: {:?}", possible_match
    )]
    PossibleMatchSkippedError { possible_match: PossibleMatch },
}

#[derive(Debug, Fail)]
pub enum SerializationError {
    #[fail(display = "Io error {:?}", path)]
    Io {
        path: PathBuf,
        #[cause]
        cause: io::Error,
    },
    #[fail(display = "Unable to write config in JSON to {:?}", path)]
    InvalidConfigFormat {
        path: PathBuf,
        #[cause]
        cause: serde_json::Error,
    },
    #[fail(display = "Unable to serialize Parser to {:?}", path)]
    ParserSerializationError {
        path: PathBuf,
        #[cause]
        cause: rmps::encode::Error,
    },
}

#[derive(Debug, Fail)]
pub enum DeserializationError {
    #[fail(display = "Io error {:?}", path)]
    Io {
        path: PathBuf,
        #[cause]
        cause: io::Error,
    },
    #[fail(display = "Unable to read JSON config at {:?}", path)]
    ReadConfigError {
        path: PathBuf,
        #[cause]
        cause: serde_json::Error,
    },
    #[fail(display = "Unable to deserialize Parser to {:?}", path)]
    ParserDeserializationError {
        path: PathBuf,
        #[cause]
        cause: rmps::decode::Error,
    },
    #[fail(display = "Unable to read JSON gazetteer at {:?}", path)]
    ReadGazetteerError {
        path: PathBuf,
        #[cause]
        cause: serde_json::Error,
    },
    #[fail(display = "Invalid limit value 0")]
    InvalidGazetteerLimit,
}
