#![allow(dead_code)]

mod vec4i;

mod test {
    use crate::vec4i::Vec4I;

    #[test]
    fn test() {
        let a = Vec4I::new_full_i32(128);
        let b = Vec4I::new_full_i32(128);
        let c = a * b;
        let d = c.load_i64();
        println!("{:?}", d);
    }
}