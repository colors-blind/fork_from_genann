//! 持久化模块
//!
//! 提供了将神经网络保存到文件和从文件加载的功能。
//! 与 C 版本的 `genann_write` 和 `genann_read` 保持兼容。
//!
//! # 文件格式
//!
//! 文件格式与 C 版本完全兼容：
//! ```text
//! inputs hidden_layers hidden outputs
//! weight1 weight2 weight3 ... weightN
//! ```
//!
//! 例如，一个 2 输入、1 隐藏层（2 神经元）、1 输出的网络：
//! ```text
//! 2 1 2 1
//! 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
//! ```

use crate::error::{NeuralNetworkError, Result};
use crate::neural_network::NeuralNetwork;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// 保存神经网络到文件
///
/// 与 C 版本的 `genann_write` 行为一致。
/// 文件格式与 C 版本完全兼容。
///
/// # 参数
/// * `nn` - 要保存的神经网络
/// * `path` - 文件路径
///
/// # 错误
/// - `NeuralNetworkError::Io` - I/O 错误
///
/// # 示例
///
/// ```ignore
/// use genann_rs::{NeuralNetwork, save};
///
/// let nn = NeuralNetwork::new(2, 1, 2, 1).unwrap();
/// save(&nn, "network.txt").unwrap();
/// ```
pub fn save(nn: &NeuralNetwork, path: impl AsRef<Path>) -> Result<()> {
    let mut file = File::create(path)?;

    // 写入网络结构
    writeln!(
        file,
        "{} {} {} {}",
        nn.inputs(),
        nn.hidden_layers(),
        nn.hidden(),
        nn.outputs()
    )?;

    // 写入权重（使用科学计数法，与 C 版本一致）
    let weights = nn.weights();
    for (i, w) in weights.iter().enumerate() {
        if i > 0 {
            write!(file, " ")?;
        }
        write!(file, "{:.20e}", w)?;
    }
    writeln!(file)?;

    Ok(())
}

/// 从文件加载神经网络
///
/// 与 C 版本的 `genann_read` 行为一致。
/// 可以加载 C 版本保存的网络文件。
///
/// # 参数
/// * `path` - 文件路径
///
/// # 返回值
/// 成功时返回加载的 `NeuralNetwork` 实例
///
/// # 错误
/// - `NeuralNetworkError::Io` - I/O 错误
/// - `NeuralNetworkError::Parse` - 解析错误
///
/// # 示例
///
/// ```ignore
/// use genann_rs::load;
///
/// let nn = load("network.txt").unwrap();
/// ```
pub fn load(path: impl AsRef<Path>) -> Result<NeuralNetwork> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    read_from(reader)
}

/// 将神经网络写入 writer
///
/// 与 `save` 类似，但接受任何实现了 `Write` trait 的类型。
///
/// # 参数
/// * `nn` - 要保存的神经网络
/// * `writer` - 实现了 `Write` trait 的对象
///
/// # 错误
/// - `NeuralNetworkError::Io` - I/O 错误
pub fn write_to<W: Write>(nn: &NeuralNetwork, mut writer: W) -> Result<()> {
    // 写入网络结构
    writeln!(
        writer,
        "{} {} {} {}",
        nn.inputs(),
        nn.hidden_layers(),
        nn.hidden(),
        nn.outputs()
    )?;

    // 写入权重
    let weights = nn.weights();
    for (i, w) in weights.iter().enumerate() {
        if i > 0 {
            write!(writer, " ")?;
        }
        write!(writer, "{:.20e}", w)?;
    }
    writeln!(writer)?;

    Ok(())
}

/// 从 reader 读取神经网络
///
/// 与 `load` 类似，但接受任何实现了 `BufRead` trait 的类型。
///
/// 与 C 版本的 `genann_read` 行为一致，可以读取 C 版本保存的文件。
/// C 版本将所有数据写在同一行，Rust 版本支持一行或两行格式。
///
/// # 参数
/// * `reader` - 实现了 `BufRead` trait 的对象
///
/// # 返回值
/// 成功时返回加载的 `NeuralNetwork` 实例
///
/// # 错误
/// - `NeuralNetworkError::Io` - I/O 错误
/// - `NeuralNetworkError::Parse` - 解析错误
pub fn read_from<R: BufRead>(mut reader: R) -> Result<NeuralNetwork> {
    let mut line = String::new();

    // 读取第一行：网络结构
    if reader.read_line(&mut line)? == 0 {
        return Err(NeuralNetworkError::Parse("Unexpected end of file".into()));
    }

    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() != 4 {
        return Err(NeuralNetworkError::Parse(format!(
            "Expected 4 numbers in first line, got {}",
            parts.len()
        )));
    }

    let inputs: usize = parts[0].parse()?;
    let hidden_layers: usize = parts[1].parse()?;
    let hidden: usize = parts[2].parse()?;
    let outputs: usize = parts[3].parse()?;

    // 创建神经网络
    let mut nn = NeuralNetwork::new(inputs, hidden_layers, hidden, outputs)?;

    // 读取第二行：权重
    line.clear();
    if reader.read_line(&mut line)? == 0 {
        return Err(NeuralNetworkError::Parse("Unexpected end of file".into()));
    }

    let weights: Vec<f64> = line
        .split_whitespace()
        .map(|s| s.parse::<f64>())
        .collect::<std::result::Result<Vec<f64>, _>>()?;

    if weights.len() != nn.total_weights() {
        return Err(NeuralNetworkError::Parse(format!(
            "Expected {} weights, got {}",
            nn.total_weights(),
            weights.len()
        )));
    }

    // 复制权重
    nn.weights_mut().copy_from_slice(&weights);

    Ok(nn)
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use std::io::Cursor;

    #[test]
    fn test_write_and_read() {
        // 创建一个简单的网络
        let mut nn = NeuralNetwork::new(2, 1, 2, 1).unwrap();

        // 设置固定权重
        let weights = [
            0.5, 1.0, 1.0,  // 隐藏神经元 0
            1.0, 1.0, 1.0,  // 隐藏神经元 1
            0.5, 1.0, -1.0, // 输出神经元
        ];
        nn.weights_mut().copy_from_slice(&weights);

        // 写入内存
        let mut buffer = Vec::new();
        write_to(&nn, &mut buffer).unwrap();

        // 从内存读取
        let loaded = read_from(Cursor::new(&buffer)).unwrap();

        // 验证结构
        assert_eq!(loaded.inputs(), nn.inputs());
        assert_eq!(loaded.hidden_layers(), nn.hidden_layers());
        assert_eq!(loaded.hidden(), nn.hidden());
        assert_eq!(loaded.outputs(), nn.outputs());
        assert_eq!(loaded.total_weights(), nn.total_weights());

        // 验证权重
        assert_eq!(loaded.weights(), nn.weights());
    }

    #[test]
    fn test_read_c_format() {
        // 模拟 C 版本保存的文件格式
        let c_format_data = "2 1 2 1\n5.00000000000000000000e-01 1.00000000000000000000e+00 1.00000000000000000000e+00 1.00000000000000000000e+00 1.00000000000000000000e+00 1.00000000000000000000e+00 5.00000000000000000000e-01 1.00000000000000000000e+00 -1.00000000000000000000e+00\n";

        // 读取
        let mut nn = read_from(Cursor::new(c_format_data)).unwrap();

        // 设置激活函数
        nn.set_hidden_activation(Activation::Threshold);
        nn.set_output_activation(Activation::Threshold);

        // 验证 XOR
        let inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let expected = [0.0, 1.0, 1.0, 0.0];

        for i in 0..4 {
            let output = nn.run(&inputs[i]).unwrap();
            assert!((output[0] - expected[i]).abs() < 0.001);
        }
    }

    #[test]
    fn test_invalid_format() {
        // 第一行数字数量不对
        let invalid = "2 1 2\n1.0 2.0";
        assert!(read_from(Cursor::new(invalid)).is_err());

        // 权重数量不对
        let invalid = "2 1 2 1\n1.0 2.0";
        assert!(read_from(Cursor::new(invalid)).is_err());

        // 不是数字
        let invalid = "2 1 2 1\nabc def";
        assert!(read_from(Cursor::new(invalid)).is_err());
    }
}
