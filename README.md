# Genann - 极简 C 语言神经网络库

[![Build Status](https://travis-ci.org/codeplea/genann.svg?branch=master)](https://travis-ci.org/codeplea/genann)

<img alt="Genann logo" src="https://codeplea.com/public/content/genann_logo.png" align="right" />

## 说明

本项目从 `https://github.com/codeplea/genann` 复制而来，仅供学习使用。

## 简介

Genann 是一个精简、经过充分测试的前馈人工神经网络（ANN）C 语言库。它的主要特点是简单、快速、可靠且易于修改。Genann 通过只提供必要的函数来实现这一点，没有多余的功能。

## 特性

- **纯 C99 实现，无外部依赖**。
- 只包含在一个源文件和一个头文件中。
- 简单易用。
- 快速且线程安全。
- 易于扩展。
- 实现了反向传播训练。
- **支持 SIMD 指令集优化**（AVX2/SSE2），可加速前向传播计算。
- *兼容其他训练方法*（经典优化、遗传算法等）。
- 包含示例代码和测试套件。
- 使用 zlib 许可证发布 - 几乎可用于任何用途。

## 编译

Genann 自包含在两个文件中：`genann.c` 和 `genann.h`。要使用 Genann，只需将这两个文件添加到您的项目中。

### 编译选项

项目支持多种编译选项：

- **默认编译**：使用 `-march=native` 自动检测 CPU 特性
  ```bash
  make
  ```

- **启用 SIMD 优化**：
  - 如果你的 CPU 支持 AVX2（大多数现代 Intel CPU），可以通过添加 `-mavx2` 来启用 AVX2 优化：
    ```bash
    CFLAGS="-Wall -Wshadow -O3 -g -mavx2" make
    ```
  - 如果只支持 SSE2：
    ```bash
    CFLAGS="-Wall -Wshadow -O3 -g -msse2" make
    ```

- **不同激活函数**：
  ```bash
  make sigmoid    # 使用 sigmoid 激活函数
  make threshold  # 使用阈值激活函数
  make linear     # 使用线性激活函数
  ```

### SIMD 优化说明

Genann 的 `genann_run` 函数（前向传播）已使用 SIMD 指令集优化：

| 指令集 | 一次处理 double 数 | 典型 CPU 支持 |
|--------|-------------------|--------------|
| AVX2 | 4 个 | Intel Haswell 及更新版本 |
| SSE2 | 2 个 | 大多数 x86-64 CPU |
| 纯 C | 1 个 | 所有平台 |

优化通过条件编译自动启用：
- 编译时添加 `-mavx2` → 使用 AVX2 优化
- 编译时添加 `-msse2` → 使用 SSE2 优化
- 无特殊选项 → 使用纯 C 实现（保证可移植性）

## 示例代码

源代码中包含四个示例程序：

- [`example1.c`](./example1.c) - 使用反向传播训练神经网络学习 XOR 函数。
- [`example2.c`](./example2.c) - 使用随机搜索训练神经网络学习 XOR 函数。
- [`example3.c`](./example3.c) - 从文件加载并运行神经网络。
- [`example4.c`](./example4.c) - 使用反向传播在 [IRIS 数据集](https://archive.ics.uci.edu/ml/datasets/Iris)上训练神经网络。

## 快速示例

我们创建一个神经网络，它有 2 个输入、1 层 3 个隐藏神经元和 2 个输出。其结构如下：

![NN 示例结构](./doc/e1.png)

然后我们使用反向传播在一组标记数据上训练它，并让它对测试数据点进行预测：

```C
#include "genann.h"

/* 此处省略加载训练和测试数据的代码 */
double **training_data_input, **training_data_output, **test_data_input;

/* 创建新网络：
 * 2 个输入，
 * 1 个隐藏层，每层 3 个神经元，
 * 2 个输出。 */
genann *ann = genann_init(2, 1, 3, 2);

/* 在训练集上学习。 */
int i, j;
for (i = 0; i < 300; ++i) {
    for (j = 0; j < 100; ++j)
        genann_train(ann, training_data_input[j], training_data_output[j], 0.1);
}

/* 运行网络，查看它的预测结果。 */
double const *prediction = genann_run(ann, test_data_input[0]);
printf("第一个测试数据点的输出是: %f, %f\n", prediction[0], prediction[1]);

genann_free(ann);
```

此示例仅用于展示 API 用法，并未展示良好的机器学习技术。在实际应用中，您可能希望以随机顺序在测试数据上学习，并且需要监控学习过程以防止过拟合。

## 使用方法

### 创建和释放神经网络

```C
genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs);
genann *genann_copy(genann const *ann);
void genann_free(genann *ann);
```

使用 `genann_init()` 函数创建新的神经网络。其参数是输入数量、隐藏层数、每个隐藏层的神经元数量以及输出数量。它返回一个 `genann` 结构体指针。

调用 `genann_copy()` 将创建现有 `genann` 结构体的深拷贝。

使用完 `genann_init()` 返回的神经网络后，请调用 `genann_free()`。

### 训练神经网络

```C
void genann_train(genann const *ann, double const *inputs,
        double const *desired_outputs, double learning_rate);
```

`genann_train()` 将使用标准反向传播执行一次更新。调用时需要传入输入数组、期望输出数组和学习率。有关使用反向传播学习的示例，请参见 *example1.c*。

Genann 的主要设计目标之一是将所有网络权重存储在一个连续的内存块中。这使得使用直接搜索数值优化算法（如[爬山法](https://en.wikipedia.org/wiki/Hill_climbing)、[遗传算法](https://en.wikipedia.org/wiki/Genetic_algorithm)、[模拟退火](https://en.wikipedia.org/wiki/Simulated_annealing)等）训练网络权重变得简单高效。

这些方法可以通过直接搜索 ANN 的权重来使用。每个 `genann` 结构体都包含成员 `int total_weights;` 和 `double *weight;`。`*weight` 指向一个大小为 `total_weights` 的数组，其中包含 ANN 使用的所有权重。有关使用随机爬山搜索进行训练的示例，请参见 *example2.c*。

### 保存和加载神经网络

```C
genann *genann_read(FILE *in);
void genann_write(genann const *ann, FILE *out);
```

Genann 提供了 `genann_read()` 和 `genann_write()` 函数，用于以基于文本的格式加载或保存神经网络。

### 评估（前向传播）

```C
double const *genann_run(genann const *ann, double const *inputs);
```

在训练好的神经网络上调用 `genann_run()` 来对给定的输入集执行前向传播。`genann_run()` 将返回指向预测输出数组的指针（长度为 `ann->outputs`）。

**SIMD 优化提示**：`genann_run` 是性能关键函数。对于大型网络或批量推理，建议使用 AVX2 编译以获得最佳性能。

## 提示

- 所有函数都以 `genann_` 开头。
- 代码很简单，请深入研究并修改它。

## 额外资源

[comp.ai.neural-nets FAQ](http://www.faqs.org/faqs/ai-faq/neural-nets/part1/) 是介绍人工神经网络的优秀资源。

如果您需要更小的神经网络库，请查看优秀的单隐藏层库 [tinn](https://github.com/glouw/tinn)。

如果您正在寻找一个更重量级、更有主见的 C 语言神经网络库，我推荐 [FANN 库](http://leenissen.dk/fann/wp/)。另一个好的库是 Peter van Rossum 的 [Lightweight Neural Network](http://lwneuralnet.sourceforge.net/)，尽管它的名字叫轻量级，但它比 Genann 更重、功能更多。

## 学习资源

有关神经网络的学习资源，请参见 [`ref.md`](./ref.md)，其中包含：
- 入门教程推荐
- 核心概念详解
- 数学基础资源
- 快速参考表
- 学习路径建议

## 许可证

本项目使用 zlib/libpng 许可证。详见 [LICENSE](./LICENSE) 文件。

原始代码版权归 Lewis Van Winkle (2015-2018) 所有。
