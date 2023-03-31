use std::arch::x86_64::*;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Sub};
use std::ptr::addr_of;

pub struct VecF32(__m256);

pub enum Rounding {
    Nearest = 0x00,
    Down = 0x01,
    Up = 0x02,
    Truncate = 0x03,
}

impl VecF32 {
    #[inline(always)]
    pub fn new_full_f32(value: f32) -> Self {
        let a = unsafe { _mm256_set1_ps(value) };
        Self(a)
    }

    #[inline(always)]
    pub fn new(value: [f32; 8]) -> Self {
        let a = unsafe { _mm256_loadu_ps(value.as_ptr()) };
        Self(a)
    }

    #[inline(always)]
    pub fn load_f32(&self) -> [f32; 8] {
        let slice = [0f32; 8];
        unsafe { _mm256_storeu_ps(slice.as_ptr() as *mut f32, self.0) };
        slice
    }

    #[inline(always)]
    pub fn sqrt(&self) -> Self {
        let a = unsafe { _mm256_sqrt_ps(self.0) };
        Self(a)
    }

    #[inline(always)]
    pub fn reciprocal(&self) -> Self {
        let a = unsafe { _mm256_rcp_ps(self.0) };
        Self(a)
    }

    #[inline(always)]
    pub fn reciprocal_sqrt(&self) -> Self {
        let a = unsafe { _mm256_rsqrt_ps(self.0) };
        Self(a)
    }

    #[inline(always)]
    pub fn max(&self, other: &Self) -> Self {
        let a = unsafe { _mm256_max_ps(self.0, other.0) };
        Self(a)
    }

    #[inline(always)]
    pub fn min(&self, other: &Self) -> Self {
        let a = unsafe { _mm256_min_ps(self.0, other.0) };
        Self(a)
    }

    #[inline(always)]
    pub fn abs(&self) -> Self {
        let a = unsafe { _mm256_andnot_ps(_mm256_set1_ps(-0.0), self.0) };
        Self(a)
    }

    #[inline(always)]
    pub fn round_int(&self) -> Self {
        return self.round(Rounding::Nearest);
    }

    #[inline(always)]
    pub fn round(&self, rounding: Rounding) -> Self {
        let a = match rounding {
            Rounding::Nearest => unsafe { _mm256_round_ps::<0x00>(self.0) },
            Rounding::Down => unsafe { _mm256_round_ps::<0x01>(self.0) },
            Rounding::Up => unsafe { _mm256_round_ps::<0x02>(self.0) },
            Rounding::Truncate => unsafe { _mm256_round_ps::<0x03>(self.0) },
        };
        Self(a)
    }

    #[inline(always)]
    pub fn floor(&self) -> Self {
        let a = unsafe { _mm256_floor_ps(self.0) };
        Self(a)
    }

    #[inline(always)]
    pub fn ceil(&self) -> Self {
        let a = unsafe { _mm256_ceil_ps(self.0) };
        Self(a)
    }

    #[inline(always)]
    pub fn eq(&self, other: &Self) -> Self {
        let a = unsafe { _mm256_cmp_ps::<_CMP_EQ_OQ>(self.0, other.0) };
        Self(a)
    }

    #[inline(always)]
    pub fn ne(&self, other: &Self) -> Self {
        let a = unsafe { _mm256_cmp_ps::<_CMP_NEQ_OQ>(self.0, other.0) };
        Self(a)
    }

    #[inline(always)]
    pub fn lt(&self, other: &Self) -> Self {
        let a = unsafe { _mm256_cmp_ps::<_CMP_LT_OQ>(self.0, other.0) };
        Self(a)
    }

    #[inline(always)]
    pub fn le(&self, other: &Self) -> Self {
        let a = unsafe { _mm256_cmp_ps::<_CMP_LE_OQ>(self.0, other.0) };
        Self(a)
    }

    #[inline(always)]
    pub fn gt(&self, other: &Self) -> Self {
        let a = unsafe { _mm256_cmp_ps::<_CMP_GT_OQ>(self.0, other.0) };
        Self(a)
    }

    #[inline(always)]
    pub fn ge(&self, other: &Self) -> Self {
        let a = unsafe { _mm256_cmp_ps::<_CMP_GE_OQ>(self.0, other.0) };
        Self(a)
    }

    #[inline(always)]
    pub fn and(&self, other: &Self) -> Self {
        let a = unsafe { _mm256_and_ps(self.0, other.0) };
        Self(a)
    }

    #[inline(always)]
    pub fn or(&self, other: &Self) -> Self {
        let a = unsafe { _mm256_or_ps(self.0, other.0) };
        Self(a)
    }

    #[inline(always)]
    pub fn xor(&self, other: &Self) -> Self {
        let a = unsafe { _mm256_xor_ps(self.0, other.0) };
        Self(a)
    }

    #[inline(always)]
    pub fn andnot(&self, other: &Self) -> Self {
        let a = unsafe { _mm256_andnot_ps(self.0, other.0) };
        Self(a)
    }

    #[inline(always)]
    pub fn blend(&self, other: &Self, mask: &Self) -> Self {
        let a = unsafe { _mm256_blendv_ps(self.0, other.0, mask.0) };
        Self(a)
    }

    #[inline(always)]
    pub fn select(&self, other: &Self, mask: &Self) -> Self {
        let a = unsafe { _mm256_blendv_ps(self.0, other.0, mask.0) };
        Self(a)
    }
}

impl Add for VecF32 {
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        let a = unsafe { _mm256_add_ps(self.0, other.0) };
        Self(a)
    }
}

impl Sub for VecF32 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        let a = unsafe { _mm256_sub_ps(self.0, other.0) };
        Self(a)
    }
}

impl Mul for VecF32 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        let a = unsafe { _mm256_mul_ps(self.0, other.0) };
        Self(a)
    }
}

impl Div for VecF32 {
    type Output = Self;

    #[inline(always)]
    fn div(self, other: Self) -> Self {
        let a = unsafe { _mm256_div_ps(self.0, other.0) };
        Self(a)
    }
}

impl BitAnd for VecF32 {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        return self.and(&other);
    }
}

impl BitOr for VecF32 {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, other: Self) -> Self {
        return self.or(&other);
    }
}

impl BitXor for VecF32 {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, other: Self) -> Self {
        return self.xor(&other);
    }
}

impl Display for VecF32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:0X?}", self.load_f32())
    }
}

impl Debug for VecF32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:0X?}", self.load_f32())
    }
}

impl Clone for VecF32 {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self(unsafe { _mm256_loadu_ps(addr_of!(self.0) as *const f32)})
    }
}

impl Copy for VecF32 {}