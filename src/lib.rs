#[feature(enabled = "avx2")]
#[allow(dead_code)]
mod avx2 {
    use std::arch::x86_64::{__m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_add_epi64, _mm256_add_epi8, _mm256_and_si256, _mm256_loadu_si256, _mm256_mul_epi32, _mm256_mul_epu32, _mm256_or_si256, _mm256_set1_epi64x, _mm256_setr_epi16, _mm256_setr_epi32, _mm256_setr_epi64x, _mm256_setr_epi8, _mm256_setzero_si256, _mm256_storeu_si256, _mm256_sub_epi16, _mm256_sub_epi32, _mm256_sub_epi64, _mm256_sub_epi8, _mm256_xor_si256};
    use std::fmt::{Debug, Display, Formatter};
    use std::ops::{Add, BitAnd, BitOr, BitXor, Mul, Not, Sub};
    use std::ptr::addr_of;

    pub struct Vec4I(__m256i);

    impl Vec4I {
        pub fn new() -> Self {
            Self(unsafe { _mm256_setzero_si256() })
        }

        pub fn new_i64(slice: &[i64; 4]) -> Self {
            Self(unsafe { _mm256_setr_epi64x(slice[0], slice[1], slice[2], slice[3]) })
        }

        pub fn new_i32(slice: &[i32; 8]) -> Self {
            Self(unsafe { _mm256_setr_epi32(slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7]) })
        }

        pub fn new_i16(slice: &[i16; 16]) -> Self {
            Self(unsafe { _mm256_setr_epi16(slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7], slice[8], slice[9], slice[10], slice[11], slice[12], slice[13], slice[14], slice[15]) })
        }

        pub fn new_i8(slice: &[i8; 32]) -> Self {
            Self(unsafe { _mm256_setr_epi8(slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7], slice[8], slice[9], slice[10], slice[11], slice[12], slice[13], slice[14], slice[15], slice[16], slice[17], slice[18], slice[19], slice[20], slice[21], slice[22], slice[23], slice[24], slice[25], slice[26], slice[27], slice[28], slice[29], slice[30], slice[31]) })
        }

        pub fn new_u64(slice: &[u64; 4]) -> Self {
            Self(unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) })
        }

        pub fn new_u32(slice: &[u32; 8]) -> Self {
            Self(unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) })
        }

        pub fn new_u16(slice: &[u16; 16]) -> Self {
            Self(unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) })
        }

        pub fn new_u8(slice: &[u8; 32]) -> Self {
            Self(unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) })
        }

        pub fn load_i64(&self) -> [i64; 4] {
            let slice = [0i64; 4];
            unsafe { _mm256_storeu_si256(slice.as_ptr() as *mut __m256i, self.0) };
            slice
        }

        pub fn load_i32(&self) -> [i32; 8] {
            let slice = [0i32; 8];
            unsafe { _mm256_storeu_si256(slice.as_ptr() as *mut __m256i, self.0) };
            slice
        }

        pub fn load_i16(&self) -> [i16; 16] {
            let slice = [0i16; 16];
            unsafe { _mm256_storeu_si256(slice.as_ptr() as *mut __m256i, self.0) };
            slice
        }

        pub fn load_i8(&self) -> [i8; 32] {
            let slice = [0i8; 32];
            unsafe { _mm256_storeu_si256(slice.as_ptr() as *mut __m256i, self.0) };
            slice
        }

        pub fn load_u64(&self) -> [u64; 4] {
            let slice = [0u64; 4];
            unsafe { _mm256_storeu_si256(slice.as_ptr() as *mut __m256i, self.0) };
            slice
        }

        pub fn load_u32(&self) -> [u32; 8] {
            let slice = [0u32; 8];
            unsafe { _mm256_storeu_si256(slice.as_ptr() as *mut __m256i, self.0) };
            slice
        }

        pub fn load_u16(&self) -> [u16; 16] {
            let slice = [0u16; 16];
            unsafe { _mm256_storeu_si256(slice.as_ptr() as *mut __m256i, self.0) };
            slice
        }

        pub fn load_u8(&self) -> [u8; 32] {
            let slice = [0u8; 32];
            unsafe { _mm256_storeu_si256(slice.as_ptr() as *mut __m256i, self.0) };
            slice
        }

        pub fn add_i64(&self, rhs: &Vec4I) -> Self {
            Self(unsafe { _mm256_add_epi64(self.0, rhs.0)})
        }

        pub fn add_i32(&self, rhs: &Vec4I) -> Self {
            Self(unsafe { _mm256_add_epi32(self.0, rhs.0)})
        }

        pub fn add_i16(&self, rhs: &Vec4I) -> Self {
            Self(unsafe { _mm256_add_epi16(self.0, rhs.0)})
        }

        pub fn add_i8(&self, rhs: &Vec4I) -> Self {
            Self(unsafe { _mm256_add_epi8(self.0, rhs.0)})
        }

        pub fn mul_i32(&self, rhs: &Vec4I) -> Self {
            Self(unsafe { _mm256_mul_epi32(self.0, rhs.0)})
        }

        pub fn mul_u32(&self, rhs: &Vec4I) -> Self {
            Self(unsafe { _mm256_mul_epu32(self.0, rhs.0)})
        }

        pub fn sub_i64(&self, rhs: &Vec4I) -> Self {
            Self(unsafe { _mm256_sub_epi64(self.0, rhs.0)})
        }

        pub fn sub_i32(&self, rhs: &Vec4I) -> Self {
            Self(unsafe { _mm256_sub_epi32(self.0, rhs.0)})
        }

        pub fn sub_i16(&self, rhs: &Vec4I) -> Self {
            Self(unsafe { _mm256_sub_epi16(self.0, rhs.0)})
        }

        pub fn sub_i8(&self, rhs: &Vec4I) -> Self {
            Self(unsafe { _mm256_sub_epi8(self.0, rhs.0)})
        }
    }

    impl Add for Vec4I {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(unsafe { _mm256_add_epi64(self.0, rhs.0)})
        }
    }

    impl Mul for Vec4I {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            Self(unsafe { _mm256_mul_epi32(self.0, rhs.0)})
        }
    }

    impl Sub for Vec4I {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self(unsafe { _mm256_sub_epi64(self.0, rhs.0)})
        }
    }

    impl BitAnd for Vec4I {
        type Output = Self;

        fn bitand(self, rhs: Self) -> Self::Output {
            Self(unsafe { _mm256_and_si256(self.0, rhs.0)})
        }
    }

    impl BitOr for Vec4I {
        type Output = Self;

        fn bitor(self, rhs: Self) -> Self::Output {
            Self(unsafe { _mm256_or_si256(self.0, rhs.0)})
        }
    }

    impl BitXor for Vec4I {
        type Output = Self;

        fn bitxor(self, rhs: Self) -> Self::Output {
            Self(unsafe { _mm256_xor_si256(self.0, rhs.0)})
        }
    }

    impl Not for Vec4I {
        type Output = Self;

        fn not(self) -> Self::Output {
            Self(unsafe { _mm256_xor_si256(self.0, _mm256_set1_epi64x(-1))})
        }
    }

    impl Default for Vec4I {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Display for Vec4I {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:0X?}", self.load_i64())
        }
    }

    impl Debug for Vec4I {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:0X?}", self.load_i64())
        }
    }

    impl Clone for Vec4I {
        fn clone(&self) -> Self {
            Self(unsafe { _mm256_loadu_si256(addr_of!(self.0))})
        }
    }

    impl Copy for Vec4I {}
}