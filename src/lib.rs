#![allow(dead_code)]

pub mod vec_i;
pub mod vec_f32;

#[cfg(test)]
mod test {
    use crate::vec_i::VecInteger;

    #[test]
    fn test() {
        let a = VecInteger::new_full_i32(128);
        let b = VecInteger::new_full_i32(128);
        let c = a * b;
        let d = c.load_i64();
        println!("{:?}", d);
    }
}