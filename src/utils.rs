use std::iter::Peekable;
use std::ops::Range;
use std::str::Chars;

/// Check whether the best parsing matches the threshold condition or not
pub fn check_threshold(n_decoded: u32, n_skips: u32, threshold: f32) -> bool {
    (n_decoded as f32) / (n_decoded as f32 + n_skips as f32) >= threshold
}

#[derive(PartialEq, Eq, Debug)]
pub struct Token {
    pub term: String,
    pub range: Range<usize>,
}

pub trait Tokenizer<'a> {
    type TokenIter: Iterator<Item = Token>;

    fn tokenize(&self, input: &'a str) -> Self::TokenIter;
}

pub struct CharTokenIter<'a> {
    current_idx: usize,
    is_separator: fn(&char) -> bool,
    //    is_not_separator: fn(&char) -> bool,
    char_iterator: Peekable<Chars<'a>>,
}

impl<'a> CharTokenIter<'a> {
    pub fn new(is_separator: fn(&char) -> bool, input: &'a str) -> Self {
        CharTokenIter {
            current_idx: 0,
            char_iterator: input.chars().peekable(),
            is_separator,
            //            is_not_separator: |c| !is_separator(c),
        }
    }
}

impl<'a> Iterator for CharTokenIter<'a> {
    type Item = Token;

    fn next(&mut self) -> Option<Token> {
        // Absorb any number of whitespaces from where we are
        loop {
            match self.char_iterator.peek() {
                None => return None,
                Some(c) if !(self.is_separator)(c) => break,
                Some(_) => {}
            }
            self.char_iterator.next();
            self.current_idx += 1;
        }
        // Start a new token
        let start_token_idx = self.current_idx;
        let mut new_token: Vec<char> = vec![];
        // Absorb any number of non-whitespaces and put them in current token
        loop {
            match self.char_iterator.peek() {
                None => break,
                Some(c) if !(self.is_separator)(c) => new_token.push(*c),
                Some(_) => break,
            }
            self.char_iterator.next();
            self.current_idx += 1;
        }
        let end_token_idx = self.current_idx;
        Some(Token {
            term: new_token.into_iter().collect(),
            range: start_token_idx..end_token_idx,
        })
    }
}

#[derive(Debug)]
pub struct WhitespaceTokenizer;

impl<'a> Tokenizer<'a> for WhitespaceTokenizer {
    type TokenIter = CharTokenIter<'a>;

    fn tokenize(&self, input: &'a str) -> Self::TokenIter {
        CharTokenIter::new(|c| c.is_whitespace(), input)
    }
}

//#[derive(Debug)]
//pub struct WhitespaceTokenizer<'a> {
//    current_idx: usize,
//    char_iterator: Peekable<Chars<'a>>,
//}
//
///// Creates a tokenizer that splits on whitespace and is robust to mutilple and types of whitespaces
//pub fn whitespace_tokenizer(string: &str) -> WhitespaceTokenizer {
//    WhitespaceTokenizer {
//        char_iterator: string.chars().peekable(),
//        current_idx: 0,
//    }
//}
//
///// Iterator that outputs the next token along with its range in the input string
//impl<'a> Iterator for WhitespaceTokenizer<'a> {
//    type Item = (Range<usize>, String);
//
//    fn next(&mut self) -> Option<(Range<usize>, String)> {
//        // Absorb any number of whitespaces from where we are
//        loop {
//            match self.char_iterator.peek() {
//                None => return None,
//                Some(c) if !c.is_whitespace() => break,
//                Some(_) => {}
//            }
//            self.char_iterator.next();
//            self.current_idx += 1;
//        }
//        // Start a new token
//        let start_token_idx = self.current_idx;
//        let mut new_token: Vec<char> = vec![];
//        // Absorb any number of non-whitespaces and put them in current token
//        loop {
//            match self.char_iterator.peek() {
//                None => break,
//                Some(c) if !c.is_whitespace() => new_token.push(*c),
//                Some(_) => break,
//            }
//            self.char_iterator.next();
//            self.current_idx += 1;
//        }
//        let end_token_idx = self.current_idx;
//        Some((
//            start_token_idx..end_token_idx,
//            new_token.into_iter().collect(),
//        ))
//    }
//}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace_tokenizer_works_with_mutiple_spaces() {
        let tokens = WhitespaceTokenizer {}
            .tokenize("ceci est un   \t test ")
            .collect();
        let expected_tokens = vec![
            Token {
                range: 0..4,
                term: "ceci".to_string(),
            },
            Token {
                range: 5..8,
                term: "est".to_string(),
            },
            Token {
                range: 9..11,
                term: "un".to_string(),
            },
            Token {
                range: 16..20,
                term: "test".to_string(),
            },
        ];
        assert_eq!(expected_tokens, tokens);
    }

    #[test]
    fn whitespace_tokenizer_works_with_utf_8() {
        let tokens = WhitespaceTokenizer {}
            .tokenize("c\'est épatant\r\n")
            .collect();
        let expected_tokens = vec![
            Token {
                range: 0..5,
                term: "c\'est".to_string(),
            },
            Token {
                range: 6..13,
                term: "épatant".to_string(),
            },
        ];
        assert_eq!(expected_tokens, tokens);

        let tokens = WhitespaceTokenizer {}
            .tokenize("дра \t नमस्ते")
            .collect();
        let expected_tokens = vec![
            Token {
                range: 0..3,
                term: "дра".to_string(),
            },
            Token {
                range: 6..12,
                term: "नमस्ते".to_string(),
            },
        ];
        assert_eq!(expected_tokens, tokens);

        let tokens = WhitespaceTokenizer {}
            .tokenize("je veux écouter les rolling stones")
            .collect();

        let expected_tokens = vec![
            Token {
                range: 0..2,
                term: "je".to_string()
            },
            Token {
                range: 3..7,
                term: "veux".to_string()
            },
            Token {
                range: 8..15,
                term: "écouter".to_string()
            },
            Token {
                range: 16..19,
                term: "les".to_string()
            },
            Token {
                range: 20..27,
                term: "rolling".to_string()
            },
            Token {
                range: 28..34,
                term: "stones".to_string()
            }
        ];
        assert_eq!(expected_tokens, tokens);
    }
}
