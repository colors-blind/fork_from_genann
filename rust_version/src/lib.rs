//! Genann-rs - Rust 实现的极简前馈神经网络库
//!
//! 这是 GENANN (Minimal C Artificial Neural Network) 的 Rust 移植版本。
//! 按照 Rust 的最佳实践进行了重新设计，同时保持与 C 版本的行为一致。
//!
//! # 特性
//!
//! - 纯 Rust 实现，无外部依赖
//! - 内存安全，无需手动管理内存
//! - 灵活的激活函数系统
//! - 支持反向传播训练
//! - 可序列化/反序列化
//!
//! # 示例
//!
//! ```
//! use genann_rs::{NeuralNetwork, Activation};
//!
//! // 创建一个 2 输入、1 隐藏层（2 神经元）、1 输出的神经网络
//! let mut nn = NeuralNetwork::new(2, 1, 2, 1)
//!     .expect("Failed to create neural network");
//!
//! // XOR 训练数据
//! let inputs = [
//!     &[0.0, 0.0][..],
//!     &[0.0, 1.0][..],
//!     &[1.0, 0.0][..],
//!     &[1.0, 1.0][..],
//! ];
//! let outputs = [
//!     &[0.0][..],
//!     &[1.0][..],
//!     &[1.0][..],
//!     &[0.0][..],
//! ];
//!
//! // 训练网络
//! for _ in 0..500 {
//!     for i in 0..4 {
//!         nn.train(inputs[i], outputs[i], 3.0);
//!     }
//! }
//!
//! // 测试网络
//! let result = nn.run(&[0.0, 1.0]);
//! assert!((result[0] - 1.0).abs() < 0.1);
//! ```

pub mod activation;
pub mod error;
pub mod neural_network;
pub mod persist;

// 重新导出常用类型，方便使用
pub use activation::Activation;
pub use error::{NeuralNetworkError, Result};
pub use neural_network::NeuralNetwork;
pub use persist::{load, save};
