/*
 * GENANN - 极简C语言人工神经网络库
 *
 * Copyright (c) 2015-2018 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * 本软件按"原样"提供，不提供任何明示或暗示的保证。
 * 在任何情况下，作者都不对因使用本软件而产生的任何损害负责。
 *
 * 允许任何人将本软件用于任何目的，包括商业应用，
 * 并且可以自由修改和重新分发，但须遵守以下限制：
 *
 * 1. 不得歪曲本软件的来源；您不得声称自己编写了原始软件。
 *    如果您在产品中使用本软件，我们将不胜感激，但不要求
 *    在产品文档中注明。
 * 2. 修改后的源版本必须明确标记，并且不得歪曲为原始软件。
 * 3. 此通知不得从任何源分发中删除或更改。
 *
 */


#ifndef GENANN_H
#define GENANN_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GENANN_RANDOM
/* 我们使用以下函数生成0到1之间的均匀随机数。
 * 如果您有更好的函数，可以重新定义此宏。 */
#define GENANN_RANDOM() (((double)rand())/RAND_MAX)
#endif

/* 前向声明神经网络结构体 */
struct genann;

/* 激活函数指针类型定义
 * 参数:
 *   ann - 神经网络指针
 *   a - 输入值
 * 返回值:
 *   激活后的输出值
 */
typedef double (*genann_actfun)(const struct genann *ann, double a);

/* 神经网络结构体定义
 * 这是一个前馈神经网络，包含输入层、隐藏层和输出层
 */
typedef struct genann {
    /* 网络结构参数 */
    int inputs;           /* 输入层神经元数量 */
    int hidden_layers;    /* 隐藏层数量 */
    int hidden;           /* 每个隐藏层的神经元数量 */
    int outputs;          /* 输出层神经元数量 */

    /* 激活函数 */
    genann_actfun activation_hidden;  /* 隐藏层激活函数，默认: genann_act_sigmoid_cached */
    genann_actfun activation_output;  /* 输出层激活函数，默认: genann_act_sigmoid_cached */

    /* 内存管理参数 */
    int total_weights;    /* 所有权重的总数，也是权重缓冲区的大小 */
    int total_neurons;    /* 神经元总数（包含输入层），也是输出缓冲区的大小 */

    /* 数据缓冲区指针
     * 注意：这些指针都指向同一块连续内存的不同位置
     */
    double *weight;       /* 所有连接权重数组（大小: total_weights） */
    double *output;       /* 存储输入数组和每个神经元的输出（大小: total_neurons） */
    double *delta;        /* 存储每个隐藏层和输出层神经元的误差增量（大小: total_neurons - inputs） */

} genann;

/*
 * 函数功能: 创建并初始化一个新的神经网络
 * 参数:
 *   inputs - 输入层神经元数量
 *   hidden_layers - 隐藏层数量（可以为0，表示没有隐藏层）
 *   hidden - 每个隐藏层的神经元数量
 *   outputs - 输出层神经元数量
 * 返回值:
 *   成功返回神经网络指针，失败返回NULL
 * 说明:
 *   所有内存（包括权重、输出、delta缓冲区）都在一次分配中完成
 */
genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs);

/*
 * 函数功能: 从文件中读取并创建神经网络
 * 参数:
 *   in - 已打开的文件指针（使用genann_write保存的文件）
 * 返回值:
 *   成功返回神经网络指针，失败返回NULL
 */
genann *genann_read(FILE *in);

/*
 * 函数功能: 随机初始化神经网络的所有权重
 * 参数:
 *   ann - 神经网络指针
 * 说明:
 *   权重范围: -0.5 到 0.5
 *   此函数在genann_init中自动调用
 */
void genann_randomize(genann *ann);

/*
 * 函数功能: 创建神经网络的深拷贝
 * 参数:
 *   ann - 原神经网络指针
 * 返回值:
 *   成功返回新的神经网络副本指针，失败返回NULL
 */
genann *genann_copy(genann const *ann);

/*
 * 函数功能: 释放神经网络使用的所有内存
 * 参数:
 *   ann - 神经网络指针
 */
void genann_free(genann *ann);

/*
 * 函数功能: 执行前向传播算法，计算神经网络的输出
 * 参数:
 *   ann - 神经网络指针
 *   inputs - 输入数据数组（大小: ann->inputs）
 * 返回值:
 *   输出层结果数组指针（大小: ann->outputs）
 * 说明:
 *   这是推理函数，用于使用训练好的网络进行预测
 */
double const *genann_run(genann const *ann, double const *inputs);

/*
 * 函数功能: 执行单次反向传播训练
 * 参数:
 *   ann - 神经网络指针
 *   inputs - 输入数据数组
 *   desired_outputs - 期望输出数组（标签）
 *   learning_rate - 学习率（决定每次更新的步长）
 * 说明:
 *   使用梯度下降算法更新网络权重
 */
void genann_train(genann const *ann, double const *inputs, double const *desired_outputs, double learning_rate);

/*
 * 函数功能: 将神经网络保存到文件
 * 参数:
 *   ann - 神经网络指针
 *   out - 已打开的文件指针（用于写入）
 * 说明:
 *   保存格式: 先保存网络结构，然后保存所有权重
 */
void genann_write(genann const *ann, FILE *out);

/*
 * 函数功能: 初始化sigmoid函数的查找表
 * 参数:
 *   ann - 神经网络指针
 * 说明:
 *   用于加速sigmoid函数的计算，通过预先计算并缓存结果
 */
void genann_init_sigmoid_lookup(const genann *ann);

/*
 * 函数功能: 标准sigmoid激活函数
 * 参数:
 *   ann - 神经网络指针（未使用）
 *   a - 输入值
 * 返回值:
 *   sigmoid(a) = 1.0 / (1 + exp(-a))
 * 说明:
 *   这是神经网络中最常用的激活函数之一，输出范围(0, 1)
 */
double genann_act_sigmoid(const genann *ann, double a);

/*
 * 函数功能: 使用查找表的sigmoid激活函数（更快）
 * 参数:
 *   ann - 神经网络指针（未使用）
 *   a - 输入值
 * 返回值:
 *   近似的sigmoid值
 * 说明:
 *   通过预先计算的查找表来加速计算，牺牲少量精度换取速度
 */
double genann_act_sigmoid_cached(const genann *ann, double a);

/*
 * 函数功能: 阈值激活函数
 * 参数:
 *   ann - 神经网络指针（未使用）
 *   a - 输入值
 * 返回值:
 *   1 如果 a > 0，否则返回 0
 * 说明:
 *   这是一个二值激活函数，用于感知机模型
 */
double genann_act_threshold(const genann *ann, double a);

/*
 * 函数功能: 线性激活函数
 * 参数:
 *   ann - 神经网络指针（未使用）
 *   a - 输入值
 * 返回值:
 *   输入值 a
 * 说明:
 *   常用于回归问题的输出层，或需要线性输出的场景
 */
double genann_act_linear(const genann *ann, double a);


#ifdef __cplusplus
}
#endif

#endif /*GENANN_H*/
