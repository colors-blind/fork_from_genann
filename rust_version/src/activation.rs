//! 激活函数模块
//!
//! 提供了神经网络中常用的激活函数，以及自定义激活函数的能力。
//!
//! # 设计思路
//!
//! 使用 Rust 的 trait 系统来定义激活函数接口，这样用户可以灵活地
//! 自定义自己的激活函数。同时提供预定义的激活函数实现。
//!
//! 与 C 版本的区别：
//! - C 版本使用函数指针，Rust 版本使用 trait，更加类型安全
//! - C 版本有 sigmoid 查找表优化，Rust 版本也实现了相同的功能

use std::f64::consts::E;

/// 激活函数 trait
///
/// 定义了激活函数需要实现的接口。
/// 任何实现了这个 trait 的类型都可以用作神经网络的激活函数。
///
/// # 示例
///
/// ```
/// use genann_rs::activation::ActivationFunction;
///
/// // 自定义 ReLU 激活函数
/// struct ReLU;
///
/// impl ActivationFunction for ReLU {
///     fn activate(&self, x: f64) -> f64 {
///         x.max(0.0)
///     }
///
///     fn derivative(&self, x: f64) -> f64 {
///         if x > 0.0 { 1.0 } else { 0.0 }
///     }
///
///     fn name(&self) -> &'static str {
///         "ReLU"
///     }
/// }
/// ```
pub trait ActivationFunction: Clone + std::fmt::Debug {
    /// 计算激活函数的值
    ///
    /// # 参数
    /// * `x` - 输入值
    ///
    /// # 返回值
    /// 激活后的输出值
    fn activate(&self, x: f64) -> f64;

    /// 计算激活函数的导数
    ///
    /// **注意**: 这里的 `x` 是激活后的输出值，而不是输入值。
    /// 这样设计是为了和 C 版本的行为保持一致。
    ///
    /// 对于 sigmoid 函数，导数可以表示为：
    /// f'(x) = f(x) * (1 - f(x))
    /// 其中 f(x) 是激活后的输出值。
    ///
    /// # 参数
    /// * `output` - 激活后的输出值
    ///
    /// # 返回值
    /// 导数值
    fn derivative(&self, output: f64) -> f64;

    /// 返回激活函数的名称
    fn name(&self) -> &'static str;
}

// ============================================================================
// Sigmoid 激活函数
// ============================================================================

/// Sigmoid 激活函数（带缓存优化）
///
/// 使用查找表来加速计算，与 C 版本的 `genann_act_sigmoid_cached` 行为一致。
///
/// 公式: σ(x) = 1 / (1 + e^(-x))
///
/// # 特性
/// - 输出范围: (0, 1)
/// - 优点: 平滑、可导、输出可以解释为概率
/// - 缺点: 可能梯度消失、计算相对较慢
///
/// # 查找表优化
///
/// 预计算了 x ∈ [-15, 15] 范围内的 sigmoid 值，使用 4096 个点。
/// 超出范围的值使用边界值。
#[derive(Clone, Debug)]
pub struct SigmoidCached {
    /// 查找表
    lookup: [f64; LOOKUP_SIZE],
    /// 间隔因子，用于快速计算索引
    interval: f64,
}

/// 查找表大小
const LOOKUP_SIZE: usize = 4096;
/// sigmoid 查找表的定义域最小值
const SIGMOID_DOM_MIN: f64 = -15.0;
/// sigmoid 查找表的定义域最大值
const SIGMOID_DOM_MAX: f64 = 15.0;

impl Default for SigmoidCached {
    fn default() -> Self {
        Self::new()
    }
}

impl SigmoidCached {
    /// 创建一个新的 SigmoidCached 实例
    ///
    /// 自动初始化查找表。
    pub fn new() -> Self {
        let mut lookup = [0.0; LOOKUP_SIZE];
        let f = (SIGMOID_DOM_MAX - SIGMOID_DOM_MIN) / LOOKUP_SIZE as f64;
        let interval = LOOKUP_SIZE as f64 / (SIGMOID_DOM_MAX - SIGMOID_DOM_MIN);

        for i in 0..LOOKUP_SIZE {
            let x = SIGMOID_DOM_MIN + f * i as f64;
            lookup[i] = sigmoid(x);
        }

        SigmoidCached { lookup, interval }
    }
}

impl ActivationFunction for SigmoidCached {
    fn activate(&self, x: f64) -> f64 {
        // 边界处理
        if x < SIGMOID_DOM_MIN {
            return self.lookup[0];
        }
        if x >= SIGMOID_DOM_MAX {
            return self.lookup[LOOKUP_SIZE - 1];
        }

        // 计算查找表索引
        let j = ((x - SIGMOID_DOM_MIN) * self.interval + 0.5) as usize;

        // 边界保护
        if j >= LOOKUP_SIZE {
            return self.lookup[LOOKUP_SIZE - 1];
        }

        self.lookup[j]
    }

    fn derivative(&self, output: f64) -> f64 {
        // sigmoid 的导数: f'(x) = f(x) * (1 - f(x))
        output * (1.0 - output)
    }

    fn name(&self) -> &'static str {
        "Sigmoid (Cached)"
    }
}

/// 标准 Sigmoid 激活函数（无缓存）
///
/// 与 C 版本的 `genann_act_sigmoid` 行为一致。
/// 每次调用都直接计算，不使用查找表。
#[derive(Clone, Debug, Default)]
pub struct Sigmoid;

impl Sigmoid {
    /// 创建一个新的 Sigmoid 实例
    pub fn new() -> Self {
        Sigmoid
    }
}

impl ActivationFunction for Sigmoid {
    fn activate(&self, x: f64) -> f64 {
        sigmoid(x)
    }

    fn derivative(&self, output: f64) -> f64 {
        output * (1.0 - output)
    }

    fn name(&self) -> &'static str {
        "Sigmoid"
    }
}

/// 计算 sigmoid 函数的原始实现
fn sigmoid(x: f64) -> f64 {
    // 对极大/极小值进行快速处理，避免 exp 计算溢出
    if x < -45.0 {
        return 0.0;
    }
    if x > 45.0 {
        return 1.0;
    }
    1.0 / (1.0 + E.powf(-x))
}

// ============================================================================
// 线性激活函数
// ============================================================================

/// 线性激活函数
///
/// 与 C 版本的 `genann_act_linear` 行为一致。
///
/// 公式: f(x) = x
///
/// # 特性
/// - 输出范围: (-∞, +∞)
/// - 用途: 回归问题的输出层
#[derive(Clone, Debug, Default)]
pub struct Linear;

impl Linear {
    /// 创建一个新的 Linear 实例
    pub fn new() -> Self {
        Linear
    }
}

impl ActivationFunction for Linear {
    fn activate(&self, x: f64) -> f64 {
        x
    }

    fn derivative(&self, _output: f64) -> f64 {
        // 线性函数的导数是 1
        1.0
    }

    fn name(&self) -> &'static str {
        "Linear"
    }
}

// ============================================================================
// 阈值激活函数
// ============================================================================

/// 阈值激活函数
///
/// 与 C 版本的 `genann_act_threshold` 行为一致。
///
/// 公式: f(x) = if x > 0 { 1 } else { 0 }
///
/// # 特性
/// - 输出: 0 或 1
/// - 注意: 不可导，不能用于反向传播训练
///   但可以用于测试已训练好的网络
#[derive(Clone, Debug, Default)]
pub struct Threshold;

impl Threshold {
    /// 创建一个新的 Threshold 实例
    pub fn new() -> Self {
        Threshold
    }
}

impl ActivationFunction for Threshold {
    fn activate(&self, x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    fn derivative(&self, _output: f64) -> f64 {
        // 阈值函数在 x=0 处不可导
        // 这里返回 0 作为占位，但实际上不应该在训练中使用
        0.0
    }

    fn name(&self) -> &'static str {
        "Threshold"
    }
}

// ============================================================================
// 激活函数枚举
// ============================================================================

/// 预定义的激活函数枚举
///
/// 提供了一种方便的方式来选择常用的激活函数。
/// 与 C 版本的激活函数选择方式对应。
///
/// # 示例
///
/// ```
/// use genann_rs::{NeuralNetwork, Activation};
///
/// let mut nn = NeuralNetwork::new(2, 1, 2, 1).unwrap();
///
/// // 设置隐藏层使用 sigmoid 激活函数
/// nn.set_hidden_activation(Activation::SigmoidCached);
///
/// // 设置输出层使用线性激活函数（用于回归）
/// nn.set_output_activation(Activation::Linear);
/// ```
#[derive(Clone, Debug)]
pub enum Activation {
    /// 带缓存的 Sigmoid 激活函数（默认）
    /// 与 C 版本的 `genann_act_sigmoid_cached` 对应
    SigmoidCached,

    /// 标准 Sigmoid 激活函数（无缓存）
    /// 与 C 版本的 `genann_act_sigmoid` 对应
    Sigmoid,

    /// 线性激活函数
    /// 与 C 版本的 `genann_act_linear` 对应
    Linear,

    /// 阈值激活函数
    /// 与 C 版本的 `genann_act_threshold` 对应
    Threshold,
}

impl Default for Activation {
    fn default() -> Self {
        Activation::SigmoidCached
    }
}

// 为 Activation 枚举实现 ActivationFunction trait
// 这样就可以直接使用枚举作为激活函数
impl ActivationFunction for Activation {
    fn activate(&self, x: f64) -> f64 {
        match self {
            Activation::SigmoidCached => SigmoidCached::new().activate(x),
            Activation::Sigmoid => Sigmoid::new().activate(x),
            Activation::Linear => Linear::new().activate(x),
            Activation::Threshold => Threshold::new().activate(x),
        }
    }

    fn derivative(&self, output: f64) -> f64 {
        match self {
            Activation::SigmoidCached => SigmoidCached::new().derivative(output),
            Activation::Sigmoid => Sigmoid::new().derivative(output),
            Activation::Linear => Linear::new().derivative(output),
            Activation::Threshold => Threshold::new().derivative(output),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Activation::SigmoidCached => "Sigmoid (Cached)",
            Activation::Sigmoid => "Sigmoid",
            Activation::Linear => "Linear",
            Activation::Threshold => "Threshold",
        }
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let s = Sigmoid::new();

        // 测试边界情况
        assert!((s.activate(-50.0) - 0.0).abs() < 1e-10);
        assert!((s.activate(50.0) - 1.0).abs() < 1e-10);

        // 测试 0 点
        assert!((s.activate(0.0) - 0.5).abs() < 1e-10);

        // 测试导数
        assert!((s.derivative(0.5) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_cached() {
        let s = SigmoidCached::new();

        // 测试边界情况
        assert!((s.activate(-20.0) - 0.0).abs() < 1e-10);
        assert!((s.activate(20.0) - 1.0).abs() < 1e-10);

        // 测试 0 点（应该接近 0.5）
        assert!((s.activate(0.0) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_linear() {
        let l = Linear::new();

        assert!((l.activate(0.0) - 0.0).abs() < 1e-10);
        assert!((l.activate(5.0) - 5.0).abs() < 1e-10);
        assert!((l.activate(-3.0) - (-3.0)).abs() < 1e-10);

        // 线性函数的导数应该是 1
        assert!((l.derivative(0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_threshold() {
        let t = Threshold::new();

        assert!((t.activate(1.0) - 1.0).abs() < 1e-10);
        assert!((t.activate(0.5) - 1.0).abs() < 1e-10);
        assert!((t.activate(0.0) - 0.0).abs() < 1e-10);
        assert!((t.activate(-1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_activation_enum() {
        // 测试 SigmoidCached
        let a = Activation::SigmoidCached;
        assert!((a.activate(0.0) - 0.5).abs() < 0.01);

        // 测试 Linear
        let a = Activation::Linear;
        assert!((a.activate(5.0) - 5.0).abs() < 1e-10);

        // 测试 Threshold
        let a = Activation::Threshold;
        assert!((a.activate(1.0) - 1.0).abs() < 1e-10);
    }
}
