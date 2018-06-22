use constants::RESOLVED_SYMBOL;
use std::ops::Range;
use std::str::Chars;

/// This function formats the resolved value output by the resolver fst. It's inverse is
/// fst_unformat_resolved_value
pub fn fst_format_resolved_value(string: &str) -> String {
    format!("{}:{}", RESOLVED_SYMBOL, string.replace(" ", "_"))
}

/// This function is the inverse of fst_format_resolved_value. It parses the output of the resolver fst to resturn the resolved value
pub fn fst_unformat_resolved_value(string: &str) -> String {
    string
        .replace(&format!("{}:", RESOLVED_SYMBOL), "")
        .replace("_", " ")
}

pub fn check_threshold(n_decoded: usize, n_skips: usize, threshold: f32) -> bool {
    // we use n_skip - 1 because the bottleneck takes away one good token
    // that ends uo being skipped
    (n_decoded as f32) / (n_decoded as f32 + n_skips as f32 - 1.0) >= threshold
}

#[derive(Debug)]
pub struct WhitespaceTokenizer<'a> {
    current_idx: usize,
    char_iterator: Chars<'a>,
    is_done: bool,
}

/// Creates a tokenizer that splits on whitespace and is robust to mutilple and types of whitespaces
pub fn whitespace_tokenizer(string: &str) -> WhitespaceTokenizer {
    WhitespaceTokenizer {
        char_iterator: string.chars(),
        is_done: false,
        current_idx: 0,
    }
}

/// Iterator that outputs the next token along with its range in the input string
impl<'a> Iterator for WhitespaceTokenizer<'a> {
    type Item = (Range<usize>, String);

    fn next(&mut self) -> Option<(Range<usize>, String)> {
        if self.is_done {
            return None;
        }
        let mut next_char: char;
        // Absorb any number of whitespaces from where we are
        loop {
            match self.char_iterator.next() {
                None => return None,
                Some(_char) => {
                    next_char = _char;
                    self.current_idx += 1;
                }
            }
            if !next_char.is_whitespace() {
                break;
            }
        }
        // Start a new token
        let start_token_idx = self.current_idx - 1; // we've overshot the start of the token
        let mut new_token: Vec<char> = vec![next_char];
        // Absorb any number of non-whitespaces and put them in current token
        loop {
            match self.char_iterator.next() {
                None => {
                    self.is_done = true;
                }
                Some(_char) => {
                    next_char = _char;
                }
            }
            self.current_idx += 1;
            if next_char.is_whitespace() || self.is_done {
                break;
            } else {
                new_token.push(next_char);
            }
        }
        let end_token_idx = self.current_idx - 1; // Overshot end of token
        Some((
            start_token_idx..end_token_idx,
            new_token.into_iter().collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fst_format_works() {
        assert_eq!(
            fst_format_resolved_value("hello world"),
            "__RESOLVED__:hello_world"
        );
        assert_eq!(
            fst_unformat_resolved_value("__RESOLVED__:hello_world"),
            "hello world"
        );
    }

    #[test]
    fn whitespace_tokenizer_works_with_mutiple_spaces() {
        let mut tokenizer = whitespace_tokenizer("ceci est un   \t test ");
        assert_eq!(tokenizer.next(), Some((0..4, "ceci".to_string())));
        assert_eq!(tokenizer.next(), Some((5..8, "est".to_string())));
        assert_eq!(tokenizer.next(), Some((9..11, "un".to_string())));
        assert_eq!(tokenizer.next(), Some((16..20, "test".to_string())));
        assert_eq!(tokenizer.next(), None);
    }

    #[test]
    fn whitespace_tokenizer_works_with_utf_8() {
        let mut tokenizer = whitespace_tokenizer("c\'est épatant\r\n");
        assert_eq!(tokenizer.next(), Some((0..5, "c\'est".to_string())));
        assert_eq!(tokenizer.next(), Some((6..13, "épatant".to_string())));

        let mut tokenizer = whitespace_tokenizer("дра \t नमस्ते");
        assert_eq!(tokenizer.next(), Some((0..3, "дра".to_string())));
        assert_eq!(
            tokenizer.next(),
            Some((6..12, "नमस्ते".to_string()))
        );

        let mut tokenizer = whitespace_tokenizer("je veux écouter les rolling stones");
        assert_eq!(tokenizer.next(), Some((0..2, "je".to_string())));
        assert_eq!(tokenizer.next(), Some((3..7, "veux".to_string())));
        assert_eq!(tokenizer.next(), Some((8..15, "écouter".to_string())));
        assert_eq!(tokenizer.next(), Some((16..19, "les".to_string())));
        assert_eq!(tokenizer.next(), Some((20..27, "rolling".to_string())));
        assert_eq!(tokenizer.next(), Some((28..34, "stones".to_string())));
    }
}
