use std::io;
use std::path::{PathBuf};
use parser::PossibleMatch;
use serde_json;
use rmps;

pub type GazetteerParserResult<T, E> = ::std::result::Result<T, E>;

#[derive(Debug, Fail, Clone)]
pub enum SymbolTableAddSymbolError {
    #[fail(display = "Key {} missing from collection {}", key, collection)]
    MissingKeyError {
        key: String,
        collection: String
    },
    #[fail(display = "Symbol {} is already present several times in the symbol table, cannot determine which index to return. If this error is raised when adding a symbol, you may want to try to force add the symbol.", symbol)]
    DuplicateSymbolError {
        symbol: String
    }
}

#[derive(Debug)]
pub enum AddValueErrorKind {
    ResolvedValue,
    RawValue
}

#[derive(Debug, Fail)]
#[fail(display = "Failed to add value of kind {:?} to the parser: {:?}", kind, value)]
pub struct AddValueError {
    pub value: String,
    pub kind: AddValueErrorKind,
    #[cause]
    pub cause: SymbolTableAddSymbolError
}

#[derive(Debug, Fail)]
#[fail(display = "Failed to set stop words and edge cases")]
pub struct SetStopWordsError {
    #[cause]
    pub cause: SymbolTableAddSymbolError
}

#[derive(Debug, Fail)]
#[fail(display = "Failed to get stop words")]
pub struct GetStopWordsError {
    #[cause]
    pub cause: GazetteerParserRootError
}

#[derive(Debug, Fail)]
#[fail(display = "Failed to get edge cases")]
pub struct GetEdgeCasesError {
    #[cause]
    pub cause: GazetteerParserRootError
}

#[derive(Debug, Fail)]
pub enum InjectionRootError {
    #[fail(display = "")]
    GazetteerParserRootError(
        #[cause]
        GazetteerParserRootError),
    #[fail(display = "")]
    AddValueError(
        #[cause]
        AddValueError),
    #[fail(display = "")]
    SetStopWordsError(
        #[cause]
        SetStopWordsError)
}

#[derive(Debug, Fail)]
#[fail(display = "Failed to inject new values in Parser")]
pub struct InjectionError {
    #[cause]
    pub cause: InjectionRootError
}

#[derive(Debug, Fail)]
pub enum FindPossibleMatchRootError {
    #[fail(display = "")]
    GazetteerParserRootError(
        #[cause]
        GazetteerParserRootError),
    #[fail(display = "")]
    PossibleMatchRootError(
        #[cause]
        PossibleMatchRootError)
}


#[derive(Debug, Fail)]
#[fail(display = "Error finding possible matches")]
pub struct FindPossibleMatchError {
    #[cause]
    pub cause: FindPossibleMatchRootError
}

// #[derive(Debug, Fail)]
// pub enum ParseInputRootError {
//     GazetteerParserRootError(GazetteerParserRootError)
// }

#[derive(Debug, Fail)]
#[fail(display = "Error parsing input")]
pub struct ParseInputError {
    #[cause]
    pub cause: GazetteerParserRootError
}

#[derive(Debug, Fail)]
pub enum RunRootError {
    #[fail(display = "")]
    ParseInputError(
        #[cause]
        ParseInputError),
    #[fail(display = "")]
    FindPossibleMatchError(
        #[cause]
        FindPossibleMatchError)
}

#[derive(Debug, Fail)]
#[fail(display = "Error running parser")]
pub struct RunError {
    #[cause]
    pub cause: RunRootError
}

#[derive(Debug, Fail)]
pub enum GetParserConfigRootError {
    #[fail(display = "")]
    GetStopWordsError(
        #[cause]
        GetStopWordsError),
    #[fail(display = "")]
    GetEdgeCasesError(
        #[cause]
        GetEdgeCasesError)
}

#[derive(Debug, Fail)]
#[fail(display = "Error getting parser config")]
pub struct GetParserConfigError {
    #[cause]
    pub cause: GetParserConfigRootError
}

#[derive(Debug, Fail)]
pub enum DumpRootError {
    #[fail(display = "")]
    GetParserConfigError(
        #[cause]
        GetParserConfigError),
    #[fail(display = "")]
    GazetteerParserRootError(
        #[cause]
        GazetteerParserRootError)
}

#[derive(Debug, Fail)]
#[fail(display = "Error dumping parser")]
pub struct DumpError {
    #[cause]
    pub cause: DumpRootError
}

#[derive(Debug, Fail)]
#[fail(display = "Error loading parser")]
pub struct LoadError {
    #[cause]
    pub cause: GazetteerParserRootError
}

#[derive(Debug, Fail)]
pub enum BuildRootError {
    #[fail(display = "")]
    AddValueError(
        #[cause]
        AddValueError),
    #[fail(display = "")]
    SetStopWordsError(
        #[cause]
        SetStopWordsError)
}

#[derive(Debug, Fail)]
#[fail(display = "Error building parser")]
pub struct BuildError {
    #[cause]
    pub cause: BuildRootError
}

#[derive(Debug, Fail)]
#[fail(display = "Error loading gazetteer")]
pub struct GazetteerLoadingError {
    #[cause]
    pub cause: GazetteerParserRootError
}


/// Higher-level errors
// #[derive(Debug, Fail)]
// pub enum GazetteerParserError {
//
//     #[fail(display = "Failed to add value to the parser: {:?}", entity_value)]
//     AddValueError {
//         entity_value: EntityValue,
//         #[cause]
//         cause: GazetteerParserErrorKind
//     }
// }

// One struct deriving fail per high level error
// cause: GazetterParserLowLevelError (qui est un enum)



/// Low-level errors

#[derive(Debug, Fail)]
pub enum PossibleMatchRootError {
    #[fail(display = "Possible match consumed more tokens than are available: {:?}", possible_match)]
    PossibleMatchConsumedError {
        possible_match: PossibleMatch
    },
    #[fail(display = "Possible match skipped more tokens than are available: {:?}", possible_match)]
    PossibleMatchSkippedError {
        possible_match: PossibleMatch
    }
}

#[derive(Debug, Fail)]
pub enum GazetteerParserRootError {
    #[fail(display = "Key {} missing from collection {}", key, collection)]
    MissingKeyError {
        key: String,
        collection: String
    },
    #[fail(display = "Symbol {} is already present several times in the symbol table, cannot determine which index to return. If this error is raised when adding a symbol, you may want to try to force add the symbol.", symbol)]
    DuplicateSymbolError {
        symbol: String
    },
    #[fail(display = "Tokens list {:?} should contain value {} but doesn't", token_list, value)]
    MissingTokenFromList {
        token_list: Vec<u32>,
        value: u32
    },
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
    #[fail(display = "Unable to read JSON config at {:?}", path)]
    ReadConfigError {
        path: PathBuf,
        #[cause]
        cause: serde_json::Error,
    },
    #[fail(display = "Unable to serialize Parser to {:?}", path)]
    ParserSerializationError {
        path: PathBuf,
        #[cause]
        cause: rmps::encode::Error
    },
    #[fail(display = "Unable to deserialize Parser to {:?}", path)]
    ParserDeserializationError {
        path: PathBuf,
        #[cause]
        cause: rmps::decode::Error
    },
    #[fail(display = "Unable to read JSON gazetteer at {:?}", path)]
    ReadGazetteerError {
        path: PathBuf,
        #[cause]
        cause: serde_json::Error,
    },
    #[fail(display = "Invalid limit value 0")]
    InvalidGazetteerLimit
}
