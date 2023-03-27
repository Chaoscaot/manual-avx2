use std::arch::x86_64::{__m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_add_epi64, _mm256_add_epi8, _mm256_and_si256, _mm256_andnot_si256, _mm256_cmpeq_epi16, _mm256_cmpeq_epi32, _mm256_cmpeq_epi64, _mm256_cmpeq_epi8, _mm256_cmpgt_epi16, _mm256_cmpgt_epi32, _mm256_cmpgt_epi64, _mm256_cmpgt_epi8, _mm256_loadu_si256, _mm256_mul_epi32, _mm256_mul_epu32, _mm256_or_si256, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set1_epi64x, _mm256_set1_epi8, _mm256_setzero_si256, _mm256_storeu_si256, _mm256_sub_epi16, _mm256_sub_epi32, _mm256_sub_epi64, _mm256_sub_epi8, _mm256_xor_si256};
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, BitAnd, BitOr, BitXor, Mul, Not, Sub};
use std::ptr::addr_of;

#[derive(Debug, Clone, Copy)]
pub enum Vec4IType {
    I64X4,
    I32X8,
    I16X16,
    I8X32,
    U64X4,
    U32X8,
    U16X16,
    U8X32,
}

impl Vec4IType {
    pub fn get_size(&self) -> usize {
        match self {
            Vec4IType::I64X4 => 4,
            Vec4IType::I32X8 => 8,
            Vec4IType::I16X16 => 16,
            Vec4IType::I8X32 => 32,
            Vec4IType::U64X4 => 4,
            Vec4IType::U32X8 => 8,
            Vec4IType::U16X16 => 16,
            Vec4IType::U8X32 => 32,
        }
    }
}

pub struct Vec4I(__m256i, Vec4IType);

#[feature(enable = "avx2")]
impl Vec4I {
    #[inline(always)]
    pub fn new() -> Self {
        Self(unsafe { _mm256_setzero_si256() }, Vec4IType::I64X4)
    }

    #[inline(always)]
    pub fn new_i64(slice: &[i64; 4]) -> Self {
        Self(unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) }, Vec4IType::I64X4)
    }

    #[inline(always)]
    pub fn new_i32(slice: &[i32; 8]) -> Self {
        Self(unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) }, Vec4IType::I32X8)
    }

    #[inline(always)]
    pub fn new_i16(slice: &[i16; 16]) -> Self {
        Self(unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) }, Vec4IType::I16X16)
    }

    #[inline(always)]
    pub fn new_i8(slice: &[i8; 32]) -> Self {
        Self(unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) }, Vec4IType::I8X32)
    }

    #[inline(always)]
    pub fn new_u64(slice: &[u64; 4]) -> Self {
        Self(unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) }, Vec4IType::U64X4)
    }

    #[inline(always)]
    pub fn new_u32(slice: &[u32; 8]) -> Self {
        Self(unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) }, Vec4IType::U32X8)
    }

    #[inline(always)]
    pub fn new_u16(slice: &[u16; 16]) -> Self {
        Self(unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) }, Vec4IType::U16X16)
    }

    #[inline(always)]
    pub fn new_u8(slice: &[u8; 32]) -> Self {
        Self(unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) }, Vec4IType::U8X32)
    }

    #[inline(always)]
    pub fn new_full_i64(value: i64) -> Self {
        Self(unsafe { _mm256_set1_epi64x(value) }, Vec4IType::I64X4)
    }

    #[inline(always)]
    pub fn new_full_i32(value: i32) -> Self {
        Self(unsafe { _mm256_set1_epi32(value) }, Vec4IType::I32X8)
    }

    #[inline(always)]
    pub fn new_full_i16(value: i16) -> Self {
        Self(unsafe { _mm256_set1_epi16(value) }, Vec4IType::I16X16)
    }

    #[inline(always)]
    pub fn new_full_i8(value: i8) -> Self {
        Self(unsafe { _mm256_set1_epi8(value) }, Vec4IType::I8X32)
    }

    #[inline(always)]
    pub fn load_i64(&self) -> [i64; 4] {
        let slice = [0i64; 4];
        unsafe { _mm256_storeu_si256(slice.as_ptr() as *mut __m256i, self.0) };
        slice
    }

    #[inline(always)]
    pub fn load_i32(&self) -> [i32; 8] {
        let slice = [0i32; 8];
        unsafe { _mm256_storeu_si256(slice.as_ptr() as *mut __m256i, self.0) };
        slice
    }

    #[inline(always)]
    pub fn load_i16(&self) -> [i16; 16] {
        let slice = [0i16; 16];
        unsafe { _mm256_storeu_si256(slice.as_ptr() as *mut __m256i, self.0) };
        slice
    }

    #[inline(always)]
    pub fn load_i8(&self) -> [i8; 32] {
        let slice = [0i8; 32];
        unsafe { _mm256_storeu_si256(slice.as_ptr() as *mut __m256i, self.0) };
        slice
    }

    #[inline(always)]
    pub fn load_u64(&self) -> [u64; 4] {
        let slice = [0u64; 4];
        unsafe { _mm256_storeu_si256(slice.as_ptr() as *mut __m256i, self.0) };
        slice
    }

    #[inline(always)]
    pub fn load_u32(&self) -> [u32; 8] {
        let slice = [0u32; 8];
        unsafe { _mm256_storeu_si256(slice.as_ptr() as *mut __m256i, self.0) };
        slice
    }

    #[inline(always)]
    pub fn load_u16(&self) -> [u16; 16] {
        let slice = [0u16; 16];
        unsafe { _mm256_storeu_si256(slice.as_ptr() as *mut __m256i, self.0) };
        slice
    }

    #[inline(always)]
    pub fn load_u8(&self) -> [u8; 32] {
        let slice = [0u8; 32];
        unsafe { _mm256_storeu_si256(slice.as_ptr() as *mut __m256i, self.0) };
        slice
    }

    #[inline(always)]
    pub fn load_b64(&self) -> [bool; 4] {
        self.load_i64().map(|x| x != 0)
    }

    #[inline(always)]
    pub fn load_b32(&self) -> [bool; 8] {
        self.load_i32().map(|x| x != 0)
    }

    #[inline(always)]
    pub fn load_b16(&self) -> [bool; 16] {
        self.load_i16().map(|x| x != 0)
    }

    #[inline(always)]
    pub fn load_b8(&self) -> [bool; 32] {
        self.load_i8().map(|x| x != 0)
    }

    #[inline(always)]
    pub fn add_i64(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_add_epi64(self.0, rhs.0)}, Vec4IType::I64X4)
    }

    #[inline(always)]
    pub fn add_i32(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_add_epi32(self.0, rhs.0)}, Vec4IType::I32X8)
    }

    #[inline(always)]
    pub fn add_i16(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_add_epi16(self.0, rhs.0)}, Vec4IType::I16X16)
    }

    #[inline(always)]
    pub fn add_i8(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_add_epi8(self.0, rhs.0)}, Vec4IType::I8X32)
    }

    #[inline(always)]
    pub fn mul_i32(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_mul_epi32(self.0, rhs.0)}, Vec4IType::I64X4)
    }

    #[inline(always)]
    pub fn mul_u32(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_mul_epu32(self.0, rhs.0)}, Vec4IType::I64X4)
    }

    #[inline(always)]
    pub fn sub_i64(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_sub_epi64(self.0, rhs.0)}, Vec4IType::I64X4)
    }

    #[inline(always)]
    pub fn sub_i32(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_sub_epi32(self.0, rhs.0)}, Vec4IType::I32X8)
    }

    #[inline(always)]
    pub fn sub_i16(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_sub_epi16(self.0, rhs.0)}, Vec4IType::I16X16)
    }

    #[inline(always)]
    pub fn sub_i8(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_sub_epi8(self.0, rhs.0)}, Vec4IType::I8X32)
    }

    #[inline(always)]
    pub fn eq(&self, rhs: &Vec4I) -> Self {
        match self.1 {
            Vec4IType::I64X4 => self.eq_i64(rhs),
            Vec4IType::I32X8 => self.eq_i32(rhs),
            Vec4IType::I16X16 => self.eq_i16(rhs),
            Vec4IType::I8X32 => self.eq_i8(rhs),
            Vec4IType::U64X4 => self.eq_i64(rhs),
            Vec4IType::U32X8 => self.eq_i32(rhs),
            Vec4IType::U16X16 => self.eq_i16(rhs),
            Vec4IType::U8X32 => self.eq_i8(rhs),
        }
    }

    #[inline(always)]
    pub fn eq_i64(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_cmpeq_epi64(self.0, rhs.0)}, Vec4IType::I64X4)
    }

    #[inline(always)]
    pub fn eq_i32(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_cmpeq_epi32(self.0, rhs.0)}, Vec4IType::I32X8)
    }

    #[inline(always)]
    pub fn eq_i16(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_cmpeq_epi16(self.0, rhs.0)}, Vec4IType::I16X16)
    }

    #[inline(always)]
    pub fn eq_i8(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_cmpeq_epi8(self.0, rhs.0)}, Vec4IType::I8X32)
    }

    #[inline(always)]
    pub fn gt(&self, rhs: &Vec4I) -> Self {
        match self.1 {
            Vec4IType::I64X4 => self.gt_i64(rhs),
            Vec4IType::I32X8 => self.gt_i32(rhs),
            Vec4IType::I16X16 => self.gt_i16(rhs),
            Vec4IType::I8X32 => self.gt_i8(rhs),
            Vec4IType::U64X4 => self.gt_i64(rhs),
            Vec4IType::U32X8 => self.gt_i32(rhs),
            Vec4IType::U16X16 => self.gt_i16(rhs),
            Vec4IType::U8X32 => self.gt_i8(rhs),
        }
    }

    #[inline(always)]
    pub fn gt_i64(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_cmpgt_epi64(self.0, rhs.0)}, Vec4IType::I64X4)
    }

    #[inline(always)]
    pub fn gt_i32(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_cmpgt_epi32(self.0, rhs.0)}, Vec4IType::I32X8)
    }

    #[inline(always)]
    pub fn gt_i16(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_cmpgt_epi16(self.0, rhs.0)}, Vec4IType::I16X16)
    }

    #[inline(always)]
    pub fn gt_i8(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_cmpgt_epi8(self.0, rhs.0)}, Vec4IType::I8X32)
    }

    #[inline(always)]
    pub fn ge(&self, rhs: &Vec4I) -> Self {
        match self.1 {
            Vec4IType::I64X4 => self.ge_i64(rhs),
            Vec4IType::I32X8 => self.ge_i32(rhs),
            Vec4IType::I16X16 => self.ge_i16(rhs),
            Vec4IType::I8X32 => self.ge_i8(rhs),
            Vec4IType::U64X4 => self.ge_i64(rhs),
            Vec4IType::U32X8 => self.ge_i32(rhs),
            Vec4IType::U16X16 => self.ge_i16(rhs),
            Vec4IType::U8X32 => self.ge_i8(rhs),
        }
    }

    #[inline(always)]
    pub fn ge_i64(&self, rhs: &Vec4I) -> Self {
        self.gt_i64(rhs) | self.eq_i64(rhs)
    }

    #[inline(always)]
    pub fn ge_i32(&self, rhs: &Vec4I) -> Self {
        self.gt_i32(rhs) | self.eq_i32(rhs)
    }

    #[inline(always)]
    pub fn ge_i16(&self, rhs: &Vec4I) -> Self {
        self.gt_i16(rhs) | self.eq_i16(rhs)
    }

    #[inline(always)]
    pub fn ge_i8(&self, rhs: &Vec4I) -> Self {
        self.gt_i8(rhs) | self.eq_i8(rhs)
    }

    #[inline(always)]
    pub fn lt(&self, rhs: &Vec4I) -> Self {
        match self.1 {
            Vec4IType::I64X4 => self.lt_i64(rhs),
            Vec4IType::I32X8 => self.lt_i32(rhs),
            Vec4IType::I16X16 => self.lt_i16(rhs),
            Vec4IType::I8X32 => self.lt_i8(rhs),
            Vec4IType::U64X4 => self.lt_i64(rhs),
            Vec4IType::U32X8 => self.lt_i32(rhs),
            Vec4IType::U16X16 => self.lt_i16(rhs),
            Vec4IType::U8X32 => self.lt_i8(rhs),
        }
    }

    #[inline(always)]
    pub fn lt_i64(&self, rhs: &Vec4I) -> Self {
        !self.ge_i64(rhs)
    }

    #[inline(always)]
    pub fn lt_i32(&self, rhs: &Vec4I) -> Self {
        !self.ge_i32(rhs)
    }

    #[inline(always)]
    pub fn lt_i16(&self, rhs: &Vec4I) -> Self {
        !self.ge_i16(rhs)
    }

    #[inline(always)]
    pub fn lt_i8(&self, rhs: &Vec4I) -> Self {
        !self.ge_i8(rhs)
    }

    #[inline(always)]
    pub fn le(&self, rhs: &Vec4I) -> Self {
        match self.1 {
            Vec4IType::I64X4 => self.le_i64(rhs),
            Vec4IType::I32X8 => self.le_i32(rhs),
            Vec4IType::I16X16 => self.le_i16(rhs),
            Vec4IType::I8X32 => self.le_i8(rhs),
            Vec4IType::U64X4 => self.le_i64(rhs),
            Vec4IType::U32X8 => self.le_i32(rhs),
            Vec4IType::U16X16 => self.le_i16(rhs),
            Vec4IType::U8X32 => self.le_i8(rhs),
        }
    }

    #[inline(always)]
    pub fn le_i64(&self, rhs: &Vec4I) -> Self {
        !self.gt_i64(rhs)
    }

    #[inline(always)]
    pub fn le_i32(&self, rhs: &Vec4I) -> Self {
        !self.gt_i32(rhs)
    }

    #[inline(always)]
    pub fn le_i16(&self, rhs: &Vec4I) -> Self {
        !self.gt_i16(rhs)
    }

    #[inline(always)]
    pub fn le_i8(&self, rhs: &Vec4I) -> Self {
        !self.gt_i8(rhs)
    }

    #[inline(always)]
    pub fn and(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_and_si256(self.0, rhs.0)}, self.1)
    }

    #[inline(always)]
    pub fn or(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_or_si256(self.0, rhs.0)}, self.1)
    }

    #[inline(always)]
    pub fn xor(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_xor_si256(self.0, rhs.0)}, self.1)
    }

    #[inline(always)]
    pub fn andnot(&self, rhs: &Vec4I) -> Self {
        Self(unsafe { _mm256_andnot_si256(self.0, rhs.0)}, self.1)
    }

    #[inline(always)]
    pub fn type_of(&self) -> Vec4IType {
        self.1
    }
}

impl Add for Vec4I {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        match self.1 {
            Vec4IType::I64X4 => self.add_i64(&rhs),
            Vec4IType::I32X8 => self.add_i32(&rhs),
            Vec4IType::I16X16 => self.add_i16(&rhs),
            Vec4IType::I8X32 => self.add_i8(&rhs),
            Vec4IType::U64X4 => self.add_i64(&rhs),
            Vec4IType::U32X8 => self.add_i32(&rhs),
            Vec4IType::U16X16 => self.add_i16(&rhs),
            Vec4IType::U8X32 => self.add_i8(&rhs),
        }
    }
}

impl Mul for Vec4I {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        match self.1 {
            Vec4IType::I32X8 => self.mul_i32(&rhs),
            Vec4IType::U32X8 => self.mul_u32(&rhs),
            _ => panic!("Unsupported type"),
        }
    }
}

impl Sub for Vec4I {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        match self.1 {
            Vec4IType::I64X4 => self.sub_i64(&rhs),
            Vec4IType::I32X8 => self.sub_i32(&rhs),
            Vec4IType::I16X16 => self.sub_i16(&rhs),
            Vec4IType::I8X32 => self.sub_i8(&rhs),
            Vec4IType::U64X4 => self.sub_i64(&rhs),
            Vec4IType::U32X8 => self.sub_i32(&rhs),
            Vec4IType::U16X16 => self.sub_i16(&rhs),
            Vec4IType::U8X32 => self.sub_i8(&rhs),
        }
    }
}

impl BitAnd for Vec4I {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        self.and(&rhs)
    }
}

impl BitOr for Vec4I {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        self.or(&rhs)
    }
}

impl BitXor for Vec4I {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        self.xor(&rhs)
    }
}

impl Not for Vec4I {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output {
        Self(unsafe { _mm256_xor_si256(self.0, Vec4I::new_full_i64(-1).0)}, self.1)
    }
}

impl Default for Vec4I {
    #[inline(always)]
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

impl Copy for Vec4I {}

impl Clone for Vec4I {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self(unsafe { _mm256_loadu_si256(addr_of!(self.0))}, self.1)
    }
}