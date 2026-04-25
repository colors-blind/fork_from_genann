# 神经网络学习资源参考

这是一份为神经网络初学者准备的学习资源清单，帮助你理解 Genann 库的工作原理。

---

## 📚 入门教程

### 神经网络基础
- **3Blue1Brown - 神经网络系列视频**（强烈推荐）
  - 网址：https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3
  - 说明：用直观的可视化方式讲解神经网络的工作原理，包括前向传播、反向传播等核心概念。
  
- **Michael Nielsen - Neural Networks and Deep Learning**（免费在线书）
  - 网址：http://neuralnetworksanddeeplearning.com/
  - 说明：一本深入浅出的神经网络入门书，包含详细的数学推导和代码示例。

- **吴恩达机器学习课程**
  - 网址：https://www.coursera.org/learn/machine-learning
  - 说明：经典的机器学习入门课程，其中包含神经网络的详细讲解。

---

## 🔬 核心概念详解

### 前向传播 (Forward Propagation)
前向传播是神经网络进行预测的过程。数据从输入层经过隐藏层，最终到达输出层。

- **维基百科 - 前向传播**
  - 网址：https://en.wikipedia.org/wiki/Feedforward_neural_network
  - 说明：讲解前馈神经网络的基本结构和工作原理。

- **直观理解前向传播**
  - 网址：https://towardsdatascience.com/forward-propagation-in-neural-networks-simplified-math-and-code-version-bbcfef6f9253
  - 说明：用简单的数学和代码示例解释前向传播。

### 反向传播 (Backpropagation)
反向传播是训练神经网络的核心算法，通过计算梯度来更新权重。

- **反向传播算法详解**
  - 网址：https://www.3blue1brown.com/lessons/backpropagation
  - 说明：3Blue1Brown 的经典视频，用可视化方式讲解反向传播。

- **反向传播的数学原理**
  - 网址：http://neuralnetworksanddeeplearning.com/chap2.html
  - 说明：Michael Nielsen 书中关于反向传播的详细数学推导。

### 激活函数 (Activation Functions)
激活函数决定了神经元是否被激活，是神经网络能够学习非线性关系的关键。

- **常用激活函数详解**
  - 网址：https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
  - 说明：介绍 sigmoid、ReLU、tanh 等常用激活函数及其优缺点。

- **Sigmoid 函数**
  - 网址：https://en.wikipedia.org/wiki/Sigmoid_function
  - 说明：维基百科对 sigmoid 函数的详细介绍。

---

## 💻 实践项目

### Genann 相关
- **Genann 原项目地址**
  - 网址：https://github.com/codeplea/genann
  - 说明：本项目的原始代码仓库。

### 其他 C 语言神经网络库
- **FANN (Fast Artificial Neural Network)**
  - 网址：http://leenissen.dk/fann/wp/
  - 说明：一个功能更强大的 C 语言神经网络库。

- **Tinn**
  - 网址：https://github.com/glouw/tinn
  - 说明：一个极简的单隐藏层神经网络库。

---

## 📖 进阶学习

### 深度学习
- **Deep Learning Book**（深度学习圣经）
  - 网址：https://www.deeplearningbook.org/
  - 说明：由 Ian Goodfellow 等编写的深度学习权威教材。

### 梯度下降优化
- **梯度下降算法可视化**
  - 网址：https://ruder.io/optimizing-gradient-descent/
  - 说明：详细介绍各种梯度下降优化算法（SGD、Adam 等）。

---

## 🧮 数学基础

学习神经网络需要一些数学基础，以下是推荐资源：

### 线性代数
- **Essence of Linear Algebra**（3Blue1Brown）
  - 网址：https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
  - 说明：直观理解线性代数的核心概念。

### 微积分
- **Khan Academy - 微积分**
  - 网址：https://www.khanacademy.org/math/calculus-home
  - 说明：免费的微积分入门课程。

---

## 🔗 快速参考

| 概念 | 说明 |
|------|------|
| **神经元** | 神经网络的基本计算单元，接收输入并产生输出 |
| **权重** | 连接神经元之间的强度，表示一个神经元对另一个的影响程度 |
| **偏置** | 神经元的激活阈值，类似于线性回归中的截距 |
| **激活函数** | 引入非线性的函数，让神经网络能够学习复杂模式 |
| **前向传播** | 输入数据从输入流向输出的计算过程 |
| **反向传播** | 计算梯度并更新权重的训练过程 |
| **梯度下降** | 通过最小化损失函数来优化权重的算法 |
| **学习率** | 控制每次权重更新幅度的超参数 |

---

## 🎯 学习路径建议

1. **第1步**：观看 3Blue1Brown 的神经网络视频，建立直观理解
2. **第2步**：阅读 Michael Nielsen 的在线书第1-2章，理解数学原理
3. **第3步**：查看 Genann 的源码注释，对照理解代码实现
4. **第4步**：运行 example1.c 到 example4.c，动手实践
5. **第5步**：尝试修改代码（如更换激活函数、调整网络结构）

祝你学习愉快！ 🚀
