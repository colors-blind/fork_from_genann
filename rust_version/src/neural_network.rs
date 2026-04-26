//! 神经网络核心实现
//!
//! 这是库的核心模块，实现了前馈神经网络的所有功能。
//!
//! # 设计思路
//!
//! 与 C 版本的主要区别：
//! - C 版本使用手动内存管理，Rust 版本使用 `Vec<f64>` 自动管理
//! - C 版本使用函数指针作为激活函数，Rust 版本使用 trait
//! - C 版本返回 NULL 指针表示错误，Rust 版本使用 `Result`
//!
//! # 与 C 版本的兼容性
//!
//! 为了确保行为一致，以下内容与 C 版本保持相同：
//! - 权重的存储顺序
//! - 前向传播算法
//! - 反向传播算法
//! - 随机初始化权重的范围 (-0.5 到 0.5)
//! - 偏置的处理方式 (第一个权重 * -1.0)

use crate::activation::{Activation, ActivationFunction};
use crate::error::{NeuralNetworkError, Result};
use std::fmt;

/// 前馈神经网络
///
/// 包含了神经网络的所有状态和参数。
///
/// # 内存布局
///
/// 与 C 版本类似，使用连续的内存块存储：
/// - `weights`: 所有连接权重
/// - `outputs`: 所有神经元的输出（包含输入层）
/// - `deltas`: 所有神经元的误差增量（不包含输入层）
///
/// # 示例
///
/// ```
/// use genann_rs::NeuralNetwork;
///
/// // 创建一个 2 输入、1 隐藏层（2 神经元）、1 输出的神经网络
/// let mut nn = NeuralNetwork::new(2, 1, 2, 1).unwrap();
///
/// // 前向传播
/// let output = nn.run(&[0.5, 0.8]);
///
/// // 训练
/// nn.train(&[0.0, 0.0], &[0.0], 0.1);
/// ```
#[derive(Clone)]
pub struct NeuralNetwork {
    /// 输入层神经元数量
    input_count: usize,
    /// 隐藏层数量
    hidden_layer_count: usize,
    /// 每个隐藏层的神经元数量
    hidden_neuron_count: usize,
    /// 输出层神经元数量
    output_count: usize,

    /// 所有权重
    ///
    /// 存储顺序:
    /// 1. 输入层到第一隐藏层
    /// 2. 隐藏层之间
    /// 3. 最后隐藏层到输出层
    ///
    /// 每个神经元的权重布局: [偏置, w1, w2, ..., wn]
    weights: Vec<f64>,

    /// 所有神经元的输出
    ///
    /// 布局: [输入层输出, 隐藏层1输出, ..., 隐藏层N输出, 输出层输出]
    /// 注意: 输入层的"输出"就是输入值本身
    neuron_outputs: Vec<f64>,

    /// 所有神经元的误差增量
    ///
    /// 用于反向传播，不包含输入层
    deltas: Vec<f64>,

    /// 隐藏层激活函数
    hidden_activation: Activation,

    /// 输出层激活函数
    output_activation: Activation,
}

impl fmt::Debug for NeuralNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NeuralNetwork")
            .field("inputs", &self.input_count)
            .field("hidden_layers", &self.hidden_layer_count)
            .field("hidden", &self.hidden_neuron_count)
            .field("outputs", &self.output_count)
            .field("total_weights", &self.weights.len())
            .field("total_neurons", &self.neuron_outputs.len())
            .field(
                "hidden_activation",
                &self.hidden_activation.name(),
            )
            .field(
                "output_activation",
                &self.output_activation.name(),
            )
            .finish()
    }
}

// ============================================================================
// 构造函数和基础方法
// ============================================================================

impl NeuralNetwork {
    /// 创建一个新的神经网络
    ///
    /// # 参数
    /// * `inputs` - 输入层神经元数量 (必须 >= 1)
    /// * `hidden_layers` - 隐藏层数量 (可以为 0)
    /// * `hidden` - 每个隐藏层的神经元数量 (如果 hidden_layers > 0 则必须 >= 1)
    /// * `outputs` - 输出层神经元数量 (必须 >= 1)
    ///
    /// # 返回值
    /// 返回 `Result`，成功时包含 `NeuralNetwork` 实例
    ///
    /// # 错误
    /// - `NeuralNetworkError::InvalidParameter` - 参数无效
    ///
    /// # 示例
    ///
    /// ```
    /// use genann_rs::NeuralNetwork;
    ///
    /// // 创建一个简单的感知机（无隐藏层）
    /// let perceptron = NeuralNetwork::new(2, 0, 0, 1).unwrap();
    ///
    /// // 创建一个有隐藏层的神经网络
    /// let nn = NeuralNetwork::new(4, 2, 8, 3).unwrap();
    /// ```
    pub fn new(
        inputs: usize,
        hidden_layers: usize,
        hidden: usize,
        outputs: usize,
    ) -> Result<Self> {
        // 参数验证
        if inputs < 1 {
            return Err(NeuralNetworkError::invalid_param(
                "inputs",
                "must be at least 1",
            ));
        }
        if outputs < 1 {
            return Err(NeuralNetworkError::invalid_param(
                "outputs",
                "must be at least 1",
            ));
        }
        if hidden_layers > 0 && hidden < 1 {
            return Err(NeuralNetworkError::invalid_param(
                "hidden",
                "must be at least 1 when hidden_layers > 0",
            ));
        }

        // 计算权重数量（与 C 版本逻辑相同）
        let hidden_weights = if hidden_layers > 0 {
            // 输入层到第一隐藏层: (inputs + 1) * hidden
            let first_hidden = (inputs + 1) * hidden;
            // 隐藏层之间: (hidden_layers - 1) * (hidden + 1) * hidden
            let between_hidden = (hidden_layers - 1) * (hidden + 1) * hidden;
            first_hidden + between_hidden
        } else {
            0
        };

        let output_weights = if hidden_layers > 0 {
            // 有隐藏层: (hidden + 1) * outputs
            (hidden + 1) * outputs
        } else {
            // 无隐藏层: (inputs + 1) * outputs
            (inputs + 1) * outputs
        };

        let total_weights = hidden_weights + output_weights;
        let total_neurons = inputs + hidden * hidden_layers + outputs;

        // 创建向量
        let mut weights = vec![0.0; total_weights];
        let neuron_outputs = vec![0.0; total_neurons];
        let deltas = vec![0.0; total_neurons - inputs];

        // 随机初始化权重（与 C 版本相同的范围: -0.5 到 0.5）
        Self::randomize_weights(&mut weights);

        Ok(NeuralNetwork {
            input_count: inputs,
            hidden_layer_count: hidden_layers,
            hidden_neuron_count: hidden,
            output_count: outputs,
            weights,
            neuron_outputs,
            deltas,
            hidden_activation: Activation::SigmoidCached,
            output_activation: Activation::SigmoidCached,
        })
    }

    /// 随机初始化权重
    ///
    /// 权重范围: -0.5 到 0.5
    /// 与 C 版本的 `genann_randomize` 行为一致。
    fn randomize_weights(weights: &mut [f64]) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for w in weights {
            // 生成 [0, 1) 之间的随机数，然后减去 0.5 得到 [-0.5, 0.5)
            *w = rng.gen::<f64>() - 0.5;
        }
    }

    // ========================================================================
    // Getter 方法
    // ========================================================================

    /// 获取输入层神经元数量
    pub fn inputs(&self) -> usize {
        self.input_count
    }

    /// 获取隐藏层数量
    pub fn hidden_layers(&self) -> usize {
        self.hidden_layer_count
    }

    /// 获取每个隐藏层的神经元数量
    pub fn hidden(&self) -> usize {
        self.hidden_neuron_count
    }

    /// 获取输出层神经元数量
    pub fn outputs(&self) -> usize {
        self.output_count
    }

    /// 获取总权重数量
    pub fn total_weights(&self) -> usize {
        self.weights.len()
    }

    /// 获取总神经元数量（包含输入层）
    pub fn total_neurons(&self) -> usize {
        self.neuron_outputs.len()
    }

    /// 获取权重的引用
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// 获取权重的可变引用
    pub fn weights_mut(&mut self) -> &mut [f64] {
        &mut self.weights
    }

    // ========================================================================
    // 激活函数设置
    // ========================================================================

    /// 设置隐藏层激活函数
    pub fn set_hidden_activation(&mut self, activation: Activation) {
        self.hidden_activation = activation;
    }

    /// 设置输出层激活函数
    pub fn set_output_activation(&mut self, activation: Activation) {
        self.output_activation = activation;
    }

    // ========================================================================
    // 前向传播
    // ========================================================================

    /// 执行前向传播
    ///
    /// 计算神经网络对给定输入的输出。
    /// 与 C 版本的 `genann_run` 行为一致。
    ///
    /// # 参数
    /// * `inputs` - 输入数据切片，长度必须等于 `self.inputs()`
    ///
    /// # 返回值
    /// 输出层结果的引用
    ///
    /// # 错误
    /// - `NeuralNetworkError::InputLengthMismatch` - 输入长度不匹配
    ///
    /// # 示例
    ///
    /// ```
    /// use genann_rs::NeuralNetwork;
    ///
    /// let mut nn = NeuralNetwork::new(2, 1, 2, 1).unwrap();
    /// let output = nn.run(&[0.5, 0.8]).unwrap();
    /// assert_eq!(output.len(), 1);
    /// ```
    pub fn run(&mut self, inputs: &[f64]) -> Result<&[f64]> {
        // 验证输入长度
        if inputs.len() != self.input_count {
            return Err(NeuralNetworkError::input_mismatch(
                self.input_count,
                inputs.len(),
            ));
        }

        // 将输入复制到 outputs 数组的开头（与 C 版本相同）
        // 这样可以统一处理第一层
        self.neuron_outputs[..self.input_count].copy_from_slice(inputs);

        // 权重索引
        let mut w_idx = 0;

        // 输出索引（跳过输入层）
        let mut o_idx = self.input_count;

        // 输入索引（指向当前层的输入）
        let mut i_idx = 0;

        // 情况1: 没有隐藏层（感知机）
        if self.hidden_layer_count == 0 {
            let output_start = o_idx;

            for _ in 0..self.output_count {
                // 计算加权和
                let sum = self.compute_dot_product(&mut w_idx, i_idx, self.input_count);
                // 应用激活函数
                self.neuron_outputs[o_idx] = self.output_activation.activate(sum);
                o_idx += 1;
            }

            return Ok(&self.neuron_outputs[output_start..]);
        }

        // 情况2: 有隐藏层

        // 第一步: 计算第一隐藏层
        for _ in 0..self.hidden_neuron_count {
            let sum = self.compute_dot_product(&mut w_idx, i_idx, self.input_count);
            self.neuron_outputs[o_idx] = self.hidden_activation.activate(sum);
            o_idx += 1;
        }
        i_idx += self.input_count;

        // 第二步: 计算后续隐藏层
        for _ in 1..self.hidden_layer_count {
            for _ in 0..self.hidden_neuron_count {
                let sum = self.compute_dot_product(&mut w_idx, i_idx, self.hidden_neuron_count);
                self.neuron_outputs[o_idx] = self.hidden_activation.activate(sum);
                o_idx += 1;
            }
            i_idx += self.hidden_neuron_count;
        }

        // 第三步: 计算输出层
        let output_start = o_idx;
        for _ in 0..self.output_count {
            let sum = self.compute_dot_product(&mut w_idx, i_idx, self.hidden_neuron_count);
            self.neuron_outputs[o_idx] = self.output_activation.activate(sum);
            o_idx += 1;
        }

        Ok(&self.neuron_outputs[output_start..])
    }

    /// 计算加权和（点积）
    ///
    /// 与 C 版本的逻辑相同：
    /// sum = weights[0] * -1.0 + Σ(weights[1..n] * inputs[0..n-1])
    ///
    /// 注意: 第一个权重被当作偏置处理，乘以 -1.0
    ///
    /// # 参数
    /// * `w_idx` - 权重起始索引（会被更新）
    /// * `i_idx` - 输入起始索引
    /// * `n` - 输入数量
    ///
    /// # 返回值
    /// 加权和
    fn compute_dot_product(
        &self,
        w_idx: &mut usize,
        i_idx: usize,
        n: usize,
    ) -> f64 {
        // 第一个权重是偏置: bias = weights[w_idx] * -1.0
        let mut sum = self.weights[*w_idx] * -1.0;
        *w_idx += 1;

        // 计算加权和
        for k in 0..n {
            sum += self.weights[*w_idx] * self.neuron_outputs[i_idx + k];
            *w_idx += 1;
        }

        sum
    }

    // ========================================================================
    // 反向传播训练
    // ========================================================================

    /// 执行单次反向传播训练
    ///
    /// 与 C 版本的 `genann_train` 行为一致。
    ///
    /// # 参数
    /// * `inputs` - 输入数据
    /// * `desired_outputs` - 期望输出（标签）
    /// * `learning_rate` - 学习率
    ///
    /// # 错误
    /// - `NeuralNetworkError::InputLengthMismatch` - 输入长度不匹配
    /// - `NeuralNetworkError::OutputLengthMismatch` - 输出长度不匹配
    ///
    /// # 算法步骤
    ///
    /// 1. 前向传播计算输出
    /// 2. 计算输出层的误差增量 (delta)
    /// 3. 反向传播计算隐藏层的误差增量
    /// 4. 根据误差增量更新权重
    pub fn train(
        &mut self,
        inputs: &[f64],
        desired_outputs: &[f64],
        learning_rate: f64,
    ) -> Result<()> {
        // 验证输入输出长度
        if inputs.len() != self.input_count {
            return Err(NeuralNetworkError::input_mismatch(
                self.input_count,
                inputs.len(),
            ));
        }
        if desired_outputs.len() != self.output_count {
            return Err(NeuralNetworkError::output_mismatch(
                self.output_count,
                desired_outputs.len(),
            ));
        }

        // 第一步: 前向传播
        self.run(inputs)?;

        // 第二步: 计算输出层的 delta
        self.compute_output_deltas(desired_outputs);

        // 第三步: 计算隐藏层的 delta（如果有隐藏层）
        if self.hidden_layer_count > 0 {
            self.compute_hidden_deltas();
        }

        // 第四步: 更新权重
        self.update_weights(learning_rate);

        Ok(())
    }

    /// 计算输出层的误差增量
    fn compute_output_deltas(&mut self, desired_outputs: &[f64]) {
        // 输出层在 outputs 数组中的起始位置
        let output_start = self.input_count + self.hidden_neuron_count * self.hidden_layer_count;
        // 输出层 delta 在 deltas 数组中的起始位置
        let delta_start = self.hidden_neuron_count * self.hidden_layer_count;

        for i in 0..self.output_count {
            let output = self.neuron_outputs[output_start + i];
            let desired = desired_outputs[i];

            // 计算 delta
            // 这里使用 trait 的 derivative 方法
            let delta = self.output_activation.derivative(output) * (desired - output);

            self.deltas[delta_start + i] = delta;
        }
    }

    /// 计算隐藏层的误差增量
    fn compute_hidden_deltas(&mut self) {
        // 从最后一个隐藏层开始，向前传播
        for h in (0..self.hidden_layer_count).rev() {
            // 当前隐藏层的输出起始位置
            let output_start = self.input_count + h * self.hidden_neuron_count;
            // 当前隐藏层的 delta 起始位置
            let delta_start = h * self.hidden_neuron_count;

            // 下一层的 delta 起始位置
            let next_delta_start = (h + 1) * self.hidden_neuron_count;

            // 下一层的权重起始位置
            // 这里需要精确计算权重位置，与 C 版本保持一致
            let next_weight_start = if h == 0 {
                // 第一隐藏层的下一层权重起始位置
                (self.input_count + 1) * self.hidden_neuron_count
            } else {
                // 其他隐藏层的下一层权重起始位置
                (self.input_count + 1) * self.hidden_neuron_count + h * (self.hidden_neuron_count + 1) * self.hidden_neuron_count
            };

            // 下一层的神经元数量
            let next_neurons = if h == self.hidden_layer_count - 1 {
                // 最后一个隐藏层的下一层是输出层
                self.output_count
            } else {
                // 其他隐藏层的下一层也是隐藏层
                self.hidden_neuron_count
            };

            for j in 0..self.hidden_neuron_count {
                let output = self.neuron_outputs[output_start + j];

                // 计算后续层对当前神经元的误差贡献
                let mut delta = 0.0;
                for k in 0..next_neurons {
                    let forward_delta = if h == self.hidden_layer_count - 1 {
                        // 下一层是输出层
                        self.deltas[self.hidden_neuron_count * self.hidden_layer_count + k]
                    } else {
                        // 下一层是隐藏层
                        self.deltas[next_delta_start + k]
                    };

                    // 计算权重索引
                    // 每个下一层神经元有 (self.hidden_neuron_count + 1) 个权重
                    // j + 1 跳过偏置
                    let w_idx = next_weight_start + k * (self.hidden_neuron_count + 1) + (j + 1);
                    let forward_weight = self.weights[w_idx];

                    delta += forward_delta * forward_weight;
                }

                // 乘以激活函数的导数
                // sigmoid 导数: output * (1 - output)
                self.deltas[delta_start + j] = self.hidden_activation.derivative(output) * delta;
            }
        }
    }

    /// 更新权重
    fn update_weights(&mut self, learning_rate: f64) {
        // 权重索引
        let mut w_idx;

        // 第一步: 更新输出层权重（如果有隐藏层）
        if self.hidden_layer_count > 0 {
            // 输出层权重的起始位置
            let output_weight_start = (self.input_count + 1) * self.hidden_neuron_count
                + (self.hidden_layer_count - 1) * (self.hidden_neuron_count + 1) * self.hidden_neuron_count;
            w_idx = output_weight_start;

            // 输出层的输入（最后一个隐藏层的输出）
            let input_start = if self.hidden_layer_count > 0 {
                self.input_count + (self.hidden_layer_count - 1) * self.hidden_neuron_count
            } else {
                0
            };

            // 输出层 delta 起始位置
            let delta_start = self.hidden_neuron_count * self.hidden_layer_count;

            for j in 0..self.output_count {
                let delta = self.deltas[delta_start + j];

                // 更新偏置权重
                self.weights[w_idx] += delta * learning_rate * -1.0;
                w_idx += 1;

                // 更新其他权重
                let input_count = if self.hidden_layer_count > 0 {
                    self.hidden_neuron_count
                } else {
                    self.input_count
                };
                for k in 0..input_count {
                    self.weights[w_idx] += delta * learning_rate * self.neuron_outputs[input_start + k];
                    w_idx += 1;
                }
            }
        }

        // 第二步: 更新隐藏层权重
        // 从最后一个隐藏层开始，向前更新
        for h in (0..self.hidden_layer_count).rev() {
            // 当前隐藏层的权重起始位置
            let weight_start = if h == 0 {
                0
            } else {
                (self.input_count + 1) * self.hidden_neuron_count + (h - 1) * (self.hidden_neuron_count + 1) * self.hidden_neuron_count
            };
            w_idx = weight_start;

            // 当前隐藏层的 delta 起始位置
            let delta_start = h * self.hidden_neuron_count;

            // 当前隐藏层的输入起始位置
            let input_start = if h == 0 {
                0 // 第一隐藏层的输入是原始输入
            } else {
                self.input_count + (h - 1) * self.hidden_neuron_count
            };

            // 输入数量
            let input_count = if h == 0 {
                self.input_count
            } else {
                self.hidden_neuron_count
            };

            for j in 0..self.hidden_neuron_count {
                let delta = self.deltas[delta_start + j];

                // 更新偏置权重
                self.weights[w_idx] += delta * learning_rate * -1.0;
                w_idx += 1;

                // 更新其他权重
                for k in 0..input_count {
                    self.weights[w_idx] += delta * learning_rate * self.neuron_outputs[input_start + k];
                    w_idx += 1;
                }
            }
        }

        // 如果没有隐藏层，更新输入层到输出层的权重
        if self.hidden_layer_count == 0 {
            w_idx = 0;
            let delta_start = 0; // 没有隐藏层时，输出层 delta 从 0 开始

            for j in 0..self.output_count {
                let delta = self.deltas[delta_start + j];

                // 更新偏置权重
                self.weights[w_idx] += delta * learning_rate * -1.0;
                w_idx += 1;

                // 更新其他权重
                for k in 0..self.input_count {
                    self.weights[w_idx] += delta * learning_rate * self.neuron_outputs[k];
                    w_idx += 1;
                }
            }
        }
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;

    #[test]
    fn test_new_invalid_params() {
        // 输入数量为 0
        assert!(NeuralNetwork::new(0, 1, 2, 1).is_err());

        // 输出数量为 0
        assert!(NeuralNetwork::new(2, 1, 2, 0).is_err());

        // 有隐藏层但隐藏神经元数量为 0
        assert!(NeuralNetwork::new(2, 1, 0, 1).is_err());
    }

    #[test]
    fn test_new_valid_params() {
        // 无隐藏层
        let nn = NeuralNetwork::new(2, 0, 0, 1).unwrap();
        assert_eq!(nn.inputs(), 2);
        assert_eq!(nn.hidden_layers(), 0);
        assert_eq!(nn.outputs(), 1);

        // 有隐藏层
        let nn = NeuralNetwork::new(4, 2, 8, 3).unwrap();
        assert_eq!(nn.inputs(), 4);
        assert_eq!(nn.hidden_layers(), 2);
        assert_eq!(nn.hidden(), 8);
        assert_eq!(nn.outputs(), 3);
    }

    #[test]
    fn test_run_input_length_mismatch() {
        let mut nn = NeuralNetwork::new(2, 1, 2, 1).unwrap();

        // 输入长度不对
        assert!(nn.run(&[0.0]).is_err());
        assert!(nn.run(&[0.0, 0.0, 0.0]).is_err());

        // 输入长度正确
        assert!(nn.run(&[0.0, 0.0]).is_ok());
    }

    #[test]
    fn test_basic_forward() {
        // 测试一个简单的网络
        // 使用固定的权重来验证计算

        let mut nn = NeuralNetwork::new(1, 0, 0, 1).unwrap();

        // 设置权重: [bias, w1]
        // 让网络计算: sigmoid(-bias + w1 * input)
        // 设置 bias=0, w1=0，这样输出应该是 0.5
        nn.weights_mut().copy_from_slice(&[0.0, 0.0]);

        let output = nn.run(&[0.0]).unwrap();
        assert!((output[0] - 0.5).abs() < 0.01);

        // 设置 w1=1，输入=10，输出应该接近 1
        nn.weights_mut().copy_from_slice(&[0.0, 1.0]);
        let output = nn.run(&[10.0]).unwrap();
        assert!((output[0] - 1.0).abs() < 0.01);

        // 输入=-10，输出应该接近 0
        let output = nn.run(&[-10.0]).unwrap();
        assert!((output[0] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_xor_with_threshold() {
        // 测试 XOR 网络（使用阈值激活函数）
        // 这是 test.c 中的测试用例

        let mut nn = NeuralNetwork::new(2, 1, 2, 1).unwrap();

        // 设置隐藏层激活函数为阈值
        nn.set_hidden_activation(Activation::Threshold);
        nn.set_output_activation(Activation::Threshold);

        // 设置权重（与 test.c 相同）
        // 第一隐藏层:
        // 神经元 0: bias=0.5, w1=1, w2=1
        // 神经元 1: bias=1, w1=1, w2=1
        // 输出层:
        // bias=0.5, w1=1, w2=-1
        let weights = [
            0.5, 1.0, 1.0,  // 隐藏神经元 0
            1.0, 1.0, 1.0,  // 隐藏神经元 1
            0.5, 1.0, -1.0, // 输出神经元
        ];
        nn.weights_mut().copy_from_slice(&weights);

        // 测试 XOR
        let inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let expected = [0.0, 1.0, 1.0, 0.0];

        for i in 0..4 {
            let output = nn.run(&inputs[i]).unwrap();
            assert!((output[0] - expected[i]).abs() < 0.001);
        }
    }

    #[test]
    fn test_clone() {
        let nn = NeuralNetwork::new(2, 1, 2, 1).unwrap();
        let cloned = nn.clone();

        assert_eq!(nn.inputs(), cloned.inputs());
        assert_eq!(nn.hidden_layers(), cloned.hidden_layers());
        assert_eq!(nn.total_weights(), cloned.total_weights());
    }

    #[test]
    fn test_set_activation() {
        let mut nn = NeuralNetwork::new(2, 1, 2, 1).unwrap();

        nn.set_hidden_activation(Activation::Linear);
        nn.set_output_activation(Activation::Sigmoid);
    }
}
