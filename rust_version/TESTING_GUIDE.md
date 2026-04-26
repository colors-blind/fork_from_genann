# C 版本和 Rust 版本一致性测试指南

本文档说明如何验证 C 版本和 Rust 版本的神经网络库行为一致。

## 目录

- [测试原则](#测试原则)
- [测试方法](#测试方法)
  - [方法 1: 使用固定权重的文件交换（推荐）](#方法-1-使用固定权重的文件交换推荐)
  - [方法 2: 使用测试数据生成器](#方法-2-使用测试数据生成器)
  - [方法 3: 手动测试](#方法-3-手动测试)
- [具体测试场景](#具体测试场景)
- [预期结果比较](#预期结果比较)

## 测试原则

为了确保 C 版本和 Rust 版本行为一致，需要验证以下几点：

1. **权重存储顺序一致**：相同的网络结构，权重在内存中的顺序必须完全一致
2. **前向传播计算一致**：相同的输入和权重，输出必须完全一致
3. **反向传播计算一致**：相同的训练数据，权重更新必须一致
4. **文件格式兼容**：C 版本保存的文件必须能被 Rust 版本读取，反之亦然

**关键注意事项**：
- 由于随机数生成器在 C 和 Rust 中不同，**不能直接比较随机初始化的网络**
- 必须使用**固定权重**进行测试
- 使用阈值激活函数可以避免浮点数精度问题

## 测试方法

### 方法 1: 使用固定权重的文件交换（推荐）

这是最简单可靠的测试方法。

#### 步骤 1: 在 C 版本中创建固定权重的网络

创建一个 C 程序 `test_c.c`，使用固定权重：

```c
#include <stdio.h>
#include <stdlib.h>
#include "genann.h"

int main() {
    // 创建网络: 2 输入, 1 隐藏层, 2 隐藏神经元, 1 输出
    genann *ann = genann_init(2, 1, 2, 1);

    // 设置固定权重（与 test.c 中的 XOR 测试相同）
    // 第一隐藏层:
    // 神经元 0: bias=0.5, w1=1, w2=1
    // 神经元 1: bias=1, w1=1, w2=1
    // 输出层:
    // bias=0.5, w1=1, w2=-1
    ann->weight[0] = 0.5;  // 隐藏 0 bias
    ann->weight[1] = 1.0;  // 隐藏 0 w1
    ann->weight[2] = 1.0;  // 隐藏 0 w2
    ann->weight[3] = 1.0;  // 隐藏 1 bias
    ann->weight[4] = 1.0;  // 隐藏 1 w1
    ann->weight[5] = 1.0;  // 隐藏 1 w2
    ann->weight[6] = 0.5;  // 输出 bias
    ann->weight[7] = 1.0;  // 输出 w1
    ann->weight[8] = -1.0; // 输出 w2

    // 保存到文件
    FILE *f = fopen("test_network.txt", "w");
    genann_write(ann, f);
    fclose(f);

    // 使用阈值激活函数测试 XOR
    printf("C 版本 XOR 测试 (阈值激活函数):\n");
    
    // 设置为阈值激活函数（通过直接设置函数指针）
    // 注意: 为了简化，我们直接在测试中使用阈值逻辑
    // 或者使用 genann.h 中的阈值激活函数
    
    double inputs[4][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    // 使用权重手动计算（模拟阈值激活）
    for (int i = 0; i < 4; ++i) {
        // 第一隐藏层
        double h0 = -ann->weight[0] + ann->weight[1] * inputs[i][0] + ann->weight[2] * inputs[i][1];
        double h1 = -ann->weight[3] + ann->weight[4] * inputs[i][0] + ann->weight[5] * inputs[i][1];
        
        // 阈值激活
        double h0_out = h0 > 0.0 ? 1.0 : 0.0;
        double h1_out = h1 > 0.0 ? 1.0 : 0.0;
        
        // 输出层
        double o = -ann->weight[6] + ann->weight[7] * h0_out + ann->weight[8] * h1_out;
        double o_out = o > 0.0 ? 1.0 : 0.0;
        
        printf("  输入 [%g, %g] -> 输出 %g\n", inputs[i][0], inputs[i][1], o_out);
    }

    printf("\n网络已保存到 test_network.txt\n");

    genann_free(ann);
    return 0;
}
```

编译并运行：
```bash
gcc -o test_c test_c.c genann.c -lm
./test_c
```

#### 步骤 2: 在 Rust 版本中加载并测试

创建一个 Rust 程序 `tests/verify_xor.rs`（或者直接使用 `cargo test`）：

```rust
use genann_rs::{NeuralNetwork, Activation, load};
use std::path::Path;

fn main() {
    println!("Rust 版本 XOR 测试 (阈值激活函数):");
    
    // 从文件加载网络
    let path = Path::new("../test_network.txt");
    let mut nn = load(path).expect("Failed to load network");
    
    // 设置阈值激活函数
    nn.set_hidden_activation(Activation::Threshold);
    nn.set_output_activation(Activation::Threshold);
    
    // 测试 XOR
    let inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ];
    
    let expected = [0.0, 1.0, 1.0, 0.0];
    
    for i in 0..4 {
        let output = nn.run(&inputs[i]).unwrap();
        println!(
            "  输入 [{:?}, {:?}] -> 输出 {} (期望 {})",
            inputs[i][0], inputs[i][1], output[0], expected[i]
        );
        
        assert!((output[0] - expected[i]).abs() < 0.001, 
            "Mismatch at input {:?}: expected {}, got {}", 
            inputs[i], expected[i], output[0]);
    }
    
    println!("\n✅ 所有测试通过！");
}
```

或者直接使用单元测试（已在代码中）：
```bash
cd rust_version
cargo test
```

### 方法 2: 使用测试数据生成器

创建一个程序，生成大量测试数据并保存到文件，然后在两个版本中分别运行。

#### C 版本数据生成器 `generate_test_data.c`：

```c
#include <stdio.h>
#include <stdlib.h>
#include "genann.h"

int main() {
    // 创建网络
    genann *ann = genann_init(4, 2, 8, 3);
    
    // 设置固定权重（简单模式：递增）
    for (int i = 0; i < ann->total_weights; ++i) {
        ann->weight[i] = (i % 100 - 50) / 10.0;  // -5.0 到 4.9
    }
    
    // 保存网络
    FILE *f = fopen("test_data_network.txt", "w");
    genann_write(ann, f);
    fclose(f);
    
    // 生成测试输入和输出
    f = fopen("test_data_inputs.txt", "w");
    for (int i = 0; i < 100; ++i) {
        double in[4] = {
            (i % 10) / 10.0,
            ((i + 1) % 10) / 10.0,
            ((i + 2) % 10) / 10.0,
            ((i + 3) % 10) / 10.0,
        };
        double const *out = genann_run(ann, in);
        fprintf(f, "%g %g %g %g | %g %g %g\n", 
            in[0], in[1], in[2], in[3],
            out[0], out[1], out[2]);
    }
    fclose(f);
    
    printf("测试数据已生成\n");
    genann_free(ann);
    return 0;
}
```

#### Rust 版本验证器：

```rust
use genann_rs::{load, NeuralNetwork};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    // 加载网络
    let mut nn = load("../test_data_network.txt").unwrap();
    
    // 读取并验证测试数据
    let file = File::open("../test_data_inputs.txt").unwrap();
    let reader = BufReader::new(file);
    
    let mut passed = 0;
    let mut failed = 0;
    
    for (line_num, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split(" | ").collect();
        
        // 解析输入
        let inputs: Vec<f64> = parts[0]
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        
        // 解析期望输出
        let expected: Vec<f64> = parts[1]
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        
        // 运行网络
        let output = nn.run(&inputs).unwrap();
        
        // 比较（允许微小的浮点误差）
        let epsilon = 1e-10;
        let mut match_result = true;
        for i in 0..output.len() {
            if (output[i] - expected[i]).abs() > epsilon {
                match_result = false;
                break;
            }
        }
        
        if match_result {
            passed += 1;
        } else {
            failed += 1;
            println!("测试 {} 失败:", line_num);
            println!("  输入: {:?}", inputs);
            println!("  期望: {:?}", expected);
            println!("  实际: {:?}", output);
        }
    }
    
    println!("\n结果: 通过 {}, 失败 {}", passed, failed);
}
```

### 方法 3: 手动测试

对于简单的网络，可以手动计算并验证。

#### 示例：简单感知机测试

**网络结构**: 2 输入, 0 隐藏层, 1 输出

**权重**: [0.0, 2.0, 3.0]
- bias = 0.0
- w1 = 2.0
- w2 = 3.0

**计算**:
```
sum = -bias + w1 * x1 + w2 * x2
    = -0 + 2*x1 + 3*x2
    
output = sigmoid(sum)
```

**测试用例**:
| 输入 | sum | sigmoid(sum) |
|------|-----|--------------|
| [0, 0] | 0 | 0.5 |
| [1, 0] | 2 | ~0.88 |
| [0, 1] | 3 | ~0.95 |

## 具体测试场景

### 场景 1: 无隐藏层的网络（感知机）

**目标**: 验证最简单的网络结构

**C 版本测试代码**:
```c
genann *ann = genann_init(2, 0, 0, 1);
ann->weight[0] = 0.0;  // bias
ann->weight[1] = 1.0;  // w1
ann->weight[2] = 0.0;  // w2

double input[] = {1.0, 0.0};
double const *out = genann_run(ann, input);
// 期望: sigmoid(-0 + 1*1 + 0*0) = sigmoid(1) ≈ 0.731
```

### 场景 2: 多隐藏层的网络

**目标**: 验证多层网络的计算

**网络结构**: 2 输入, 2 隐藏层, 2 隐藏神经元, 1 输出

**验证点**:
- 权重索引计算正确
- 每层的输入来源正确
- 前向传播顺序正确

### 场景 3: 文件兼容性

**目标**: 验证 C 和 Rust 版本可以互相读取对方保存的文件

**测试步骤**:
1. C 版本保存网络 → Rust 版本读取 → 验证输出一致
2. Rust 版本保存网络 → C 版本读取 → 验证输出一致

### 场景 4: 反向传播训练（高级）

**注意**: 反向传播的验证比较复杂，因为：
- 权重更新依赖于学习率
- 浮点数精度可能有差异
- 需要精确控制所有变量

**建议的测试方法**:
1. 使用非常简单的网络（1输入, 0隐藏, 1输出）
2. 设置固定的初始权重
3. 使用固定的输入和期望输出
4. 使用固定的学习率
5. 执行一次训练，然后比较更新后的权重

## 预期结果比较

### 浮点数比较

由于浮点数精度问题，比较时应该使用容差：

```rust
// 允许的误差范围
let epsilon = 1e-10;

// 比较方式
assert!((actual - expected).abs() < epsilon);
```

### 使用阈值激活函数简化测试

使用阈值激活函数可以完全避免浮点数精度问题：

```rust
// 阈值激活函数: f(x) = if x > 0 { 1 } else { 0 }
// 输出只能是 0 或 1，可以精确比较
```

## 运行测试的命令

### 编译 C 版本

```bash
# 在项目根目录
make

# 或者手动编译测试程序
gcc -o test_xor test_c.c genann.c -lm
./test_xor
```

### 编译并测试 Rust 版本

```bash
cd rust_version

# 编译
cargo build

# 运行单元测试
cargo test

# 运行特定测试
cargo test test_xor_with_threshold

# 详细输出
cargo test -- --nocapture
```

### 集成测试示例

创建一个集成测试 `rust_version/tests/integration.rs`：

```rust
//! 集成测试：验证与 C 版本的一致性

use genann_rs::{NeuralNetwork, Activation, load};
use std::io::Cursor;

#[test]
fn test_xor_network() {
    // 这是 C 版本 test.c 中的测试用例
    let c_format_data = "2 1 2 1\n5.00000000000000000000e-01 1.00000000000000000000e+00 1.00000000000000000000e+00 1.00000000000000000000e+00 1.00000000000000000000e+00 1.00000000000000000000e+00 5.00000000000000000000e-01 1.00000000000000000000e+00 -1.00000000000000000000e+00\n";
    
    let mut nn = load(Cursor::new(c_format_data)).unwrap();
    
    nn.set_hidden_activation(Activation::Threshold);
    nn.set_output_activation(Activation::Threshold);
    
    let inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let expected = [0.0, 1.0, 1.0, 0.0];
    
    for i in 0..4 {
        let output = nn.run(&inputs[i]).unwrap();
        assert!((output[0] - expected[i]).abs() < 0.001);
    }
}

#[test]
fn test_save_load_roundtrip() {
    // 创建网络
    let mut nn = NeuralNetwork::new(4, 2, 8, 3).unwrap();
    
    // 设置固定权重
    for (i, w) in nn.weights_mut().iter_mut().enumerate() {
        *w = (i % 100 - 50) as f64 / 10.0;
    }
    
    // 保存
    let mut buffer = Vec::new();
    genann_rs::write_to(&nn, &mut buffer).unwrap();
    
    // 加载
    let loaded = genann_rs::read_from(Cursor::new(&buffer)).unwrap();
    
    // 比较结构
    assert_eq!(loaded.inputs(), nn.inputs());
    assert_eq!(loaded.hidden_layers(), nn.hidden_layers());
    assert_eq!(loaded.hidden(), nn.hidden());
    assert_eq!(loaded.outputs(), nn.outputs());
    assert_eq!(loaded.weights(), nn.weights());
}
```

## 测试清单

在验证 C 和 Rust 版本一致性时，请检查以下各项：

- [ ] 相同网络结构的权重数量一致
- [ ] 权重的存储顺序一致
- [ ] 前向传播计算结果一致（使用固定权重）
- [ ] 文件格式完全兼容
- [ ] 阈值激活函数测试通过（无浮点误差）
- [ ] sigmoid 激活函数测试通过（允许微小浮点误差）

如果所有测试都通过，说明两个版本的行为一致！
