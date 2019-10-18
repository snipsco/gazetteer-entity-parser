use std::iter::Peekable;
use std::ops::Range;
use std::str::Chars;

/// Check whether the best parsing matches the threshold condition or not
pub fn check_threshold(n_decoded: u32, n_skips: u32, threshold: f32) -> bool {
    (n_decoded as f32) / (n_decoded as f32 + n_skips as f32) >= threshold
}

#[derive(Debug)]
pub struct WhitespaceTokenizer<'a> {
    current_idx: usize,
    char_iterator: Peekable<Chars<'a>>,
}

/// Creates a tokenizer that splits on whitespace and is robust to multiple and types of whitespaces
pub fn whitespace_tokenizer(string: &str) -> WhitespaceTokenizer {
    WhitespaceTokenizer {
        char_iterator: string.chars().peekable(),
        current_idx: 0,
    }
}

/// Iterator that outputs the next token along with its range in the input string
impl<'a> Iterator for WhitespaceTokenizer<'a> {
    type Item = (Range<usize>, String);

    fn next(&mut self) -> Option<(Range<usize>, String)> {
        // Absorb any number of whitespaces from where we are
        loop {
            match self.char_iterator.peek() {
                None => return None,
                Some(c) if !c.is_whitespace() => break,
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
                Some(c) if !c.is_whitespace() => new_token.push(*c),
                Some(_) => break,
            }
            self.char_iterator.next();
            self.current_idx += 1;
        }
        let end_token_idx = self.current_idx;
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
    fn whitespace_tokenizer_works_with_multiple_spaces() {
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
