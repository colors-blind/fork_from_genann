//! 错误类型定义
//!
//! 定义了神经网络库中可能出现的所有错误类型。

use std::error::Error;
use std::fmt;
use std::io;
use std::num::ParseFloatError;

/// 神经网络操作的结果类型
pub type Result<T> = std::result::Result<T, NeuralNetworkError>;

/// 神经网络错误类型
///
/// 包含了所有可能的错误情况，使用 Rust 的 enum 进行清晰的分类。
#[derive(Debug)]
pub enum NeuralNetworkError {
    /// 无效的参数
    ///
    /// 当传入的参数不符合要求时返回此错误。
    /// 例如：输入数量为负数、隐藏层数量为负数等。
    InvalidParameter {
        /// 参数名称
        param: &'static str,
        /// 错误描述
        reason: String,
    },

    /// 输入数据长度不匹配
    ///
    /// 当输入数据的长度与神经网络期望的输入数量不匹配时返回。
    InputLengthMismatch {
        /// 期望的输入数量
        expected: usize,
        /// 实际的输入数量
        actual: usize,
    },

    /// 输出数据长度不匹配
    ///
    /// 当期望输出数据的长度与神经网络的输出数量不匹配时返回。
    OutputLengthMismatch {
        /// 期望的输出数量
        expected: usize,
        /// 实际的输出数量
        actual: usize,
    },

    /// I/O 错误
    ///
    /// 读写文件时发生的错误。
    Io(io::Error),

    /// 解析错误
    ///
    /// 从文件读取数据时解析失败。
    Parse(String),

    /// 计算错误
    ///
    /// 计算过程中出现的错误，例如 NaN 或无穷大。
    Computation(String),
}

impl fmt::Display for NeuralNetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeuralNetworkError::InvalidParameter { param, reason } => {
                write!(f, "Invalid parameter '{}': {}", param, reason)
            }
            NeuralNetworkError::InputLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "Input length mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            NeuralNetworkError::OutputLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "Output length mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            NeuralNetworkError::Io(err) => write!(f, "I/O error: {}", err),
            NeuralNetworkError::Parse(msg) => write!(f, "Parse error: {}", msg),
            NeuralNetworkError::Computation(msg) => write!(f, "Computation error: {}", msg),
        }
    }
}

impl Error for NeuralNetworkError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            NeuralNetworkError::Io(err) => Some(err),
            _ => None,
        }
    }
}

// 实现从其他错误类型到 NeuralNetworkError 的转换
// 这样可以使用 ? 操作符自动转换错误类型

impl From<io::Error> for NeuralNetworkError {
    fn from(err: io::Error) -> Self {
        NeuralNetworkError::Io(err)
    }
}

impl From<ParseFloatError> for NeuralNetworkError {
    fn from(err: ParseFloatError) -> Self {
        NeuralNetworkError::Parse(format!("Failed to parse float: {}", err))
    }
}

// 便捷的构造函数
impl NeuralNetworkError {
    /// 创建一个无效参数错误
    pub fn invalid_param(param: &'static str, reason: impl Into<String>) -> Self {
        NeuralNetworkError::InvalidParameter {
            param,
            reason: reason.into(),
        }
    }

    /// 创建一个输入长度不匹配错误
    pub fn input_mismatch(expected: usize, actual: usize) -> Self {
        NeuralNetworkError::InputLengthMismatch { expected, actual }
    }

    /// 创建一个输出长度不匹配错误
    pub fn output_mismatch(expected: usize, actual: usize) -> Self {
        NeuralNetworkError::OutputLengthMismatch { expected, actual }
    }

    /// 创建一个计算错误
    pub fn computation(msg: impl Into<String>) -> Self {
        NeuralNetworkError::Computation(msg.into())
    }
}
