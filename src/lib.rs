extern crate failure;
extern crate snips_fst;

pub mod data;
pub mod errors;
pub mod resolver;
pub mod constants;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
