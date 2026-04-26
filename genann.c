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

#include "genann.h"

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * SIMD 优化支持
 * 使用编译时定义的宏来检测支持的指令集
 * 
 * 支持的SIMD级别:
 *   - GENANN_USE_AVX2: 使用AVX2指令集（一次处理4个double）
 *   - GENANN_USE_SSE2: 使用SSE2指令集（一次处理2个double）
 *   - 默认: 纯C实现（无SIMD优化）
 * 
 * 编译时可以通过添加 -mavx2 或 -msse2 来启用相应的SIMD优化
 * 例如: gcc -O3 -mavx2 -c genann.c
 */

/* 检测编译器是否支持AVX2 */
#if defined(__AVX2__) || defined(GENANN_USE_AVX2)
#include <immintrin.h>
#define GENANN_SIMD_AVX2 1
#define GENANN_SIMD_WIDTH 4  /* AVX2一次处理4个double (256位) */
#define GENANN_SIMD_ALIGN 32 /* AVX2需要32字节对齐 */

/* 检测编译器是否支持SSE2 */
#elif defined(__SSE2__) || defined(GENANN_USE_SSE2)
#include <emmintrin.h>
#define GENANN_SIMD_SSE2 1
#define GENANN_SIMD_WIDTH 2  /* SSE2一次处理2个double (128位) */
#define GENANN_SIMD_ALIGN 16 /* SSE2需要16字节对齐 */

#else
/* 纯C实现，无SIMD优化 */
#define GENANN_SIMD_WIDTH 1
#define GENANN_SIMD_ALIGN 8
#endif

/* 激活函数的间接调用宏定义
 * 如果用户在编译时定义了genann_act宏，则使用统一的激活函数
 * 否则使用独立的隐藏层和输出层激活函数
 */
#ifndef genann_act
#define genann_act_hidden genann_act_hidden_indirect
#define genann_act_output genann_act_output_indirect
#else
#define genann_act_hidden genann_act
#define genann_act_output genann_act
#endif

/* sigmoid函数查找表的大小
 * 用于加速sigmoid计算，通过预先计算并缓存结果
 */
#define LOOKUP_SIZE 4096

/*
 * 函数功能: 隐藏层激活函数的间接调用包装器
 * 参数:
 *   ann - 神经网络指针
 *   a - 输入值
 * 返回值:
 *   激活后的输出值
 * 说明:
 *   通过函数指针调用实际的隐藏层激活函数
 *   这样设计允许在运行时动态切换激活函数
 */
double genann_act_hidden_indirect(const struct genann *ann, double a) {
    return ann->activation_hidden(ann, a);
}

/*
 * 函数功能: 输出层激活函数的间接调用包装器
 * 参数:
 *   ann - 神经网络指针
 *   a - 输入值
 * 返回值:
 *   激活后的输出值
 * 说明:
 *   通过函数指针调用实际的输出层激活函数
 */
double genann_act_output_indirect(const struct genann *ann, double a) {
    return ann->activation_output(ann, a);
}

/* sigmoid函数查找表的定义域范围
 * sigmoid(x) 在 x < -15 时近似为0，在 x > 15 时近似为1
 * 因此只需要缓存这个区间内的值
 */
const double sigmoid_dom_min = -15.0;
const double sigmoid_dom_max = 15.0;

/* 查找表的间隔因子（用于快速索引计算） */
double interval;

/* sigmoid函数查找表，全局静态数组 */
double lookup[LOOKUP_SIZE];

/* 编译器特定的宏定义 */
#ifdef __GNUC__
/* 分支预测宏，告诉编译器某个条件很可能为真 */
#define likely(x)       __builtin_expect(!!(x), 1)
/* 分支预测宏，告诉编译器某个条件很可能为假 */
#define unlikely(x)     __builtin_expect(!!(x), 0)
/* 标记未使用的变量，避免编译器警告 */
#define unused          __attribute__((unused))
#else
/* 非GCC编译器的备用定义 */
#define likely(x)       x
#define unlikely(x)     x
#define unused
#pragma warning(disable : 4996) /* 禁用VS编译器关于fscanf的警告 */
#endif


/*
 * 函数功能: 标准sigmoid激活函数
 * 参数:
 *   ann - 神经网络指针（未使用，保持函数签名一致）
 *   a - 输入值
 * 返回值:
 *   sigmoid(a) = 1.0 / (1 + exp(-a))
 * 数学特性:
 *   - 输出范围: (0, 1)
 *   - 导数: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
 *   - 单调递增函数
 * 说明:
 *   sigmoid函数是神经网络中最经典的激活函数之一
 *   它将任意实数映射到(0,1)区间，可以解释为概率
 *   但存在梯度消失问题（当x很大或很小时，导数趋近于0）
 */
double genann_act_sigmoid(const genann *ann unused, double a) {
    /* 对极大/极小值进行快速处理，避免exp计算溢出 */
    if (a < -45.0) return 0;      /* exp(45) 太大，返回0 */
    if (a > 45.0) return 1;       /* exp(-45) 太小，返回1 */
    return 1.0 / (1 + exp(-a));   /* 标准sigmoid公式 */
}

/*
 * 函数功能: 初始化sigmoid函数查找表
 * 参数:
 *   ann - 神经网络指针
 * 说明:
 *   预先计算sigmoid_dom_min到sigmoid_dom_max范围内的sigmoid值
 *   并存入查找表，以空间换时间，加速后续计算
 */
void genann_init_sigmoid_lookup(const genann *ann) {
    /* 计算查找表中每个元素对应的x轴步长 */
    const double f = (sigmoid_dom_max - sigmoid_dom_min) / LOOKUP_SIZE;
    int i;

    /* 计算间隔因子，用于快速将x值映射到查找表索引 */
    interval = LOOKUP_SIZE / (sigmoid_dom_max - sigmoid_dom_min);
    
    /* 填充查找表 */
    for (i = 0; i < LOOKUP_SIZE; ++i) {
        lookup[i] = genann_act_sigmoid(ann, sigmoid_dom_min + f * i);
    }
}

/*
 * 函数功能: 使用查找表的sigmoid激活函数（更快）
 * 参数:
 *   ann - 神经网络指针（未使用）
 *   a - 输入值
 * 返回值:
 *   近似的sigmoid值
 * 说明:
 *   通过查找表进行近似计算，比直接计算快很多
 *   适用于对精度要求不高但对速度要求很高的场景
 *   这是genann的默认激活函数
 */
double genann_act_sigmoid_cached(const genann *ann unused, double a) {
    /* 断言：输入值不是NaN */
    assert(!isnan(a));

    /* 边界处理：超出查找表范围的值使用边界值 */
    if (a < sigmoid_dom_min) return lookup[0];
    if (a >= sigmoid_dom_max) return lookup[LOOKUP_SIZE - 1];

    /* 计算查找表索引
     * 公式: index = (a - min) * interval
     * +0.5 是为了四舍五入到最近的整数
     */
    size_t j = (size_t)((a-sigmoid_dom_min)*interval+0.5);

    /* 浮点数精度保护：确保索引不会越界 */
    if (unlikely(j >= LOOKUP_SIZE)) return lookup[LOOKUP_SIZE - 1];

    return lookup[j];
}

/*
 * 函数功能: 线性激活函数
 * 参数:
 *   ann - 神经网络指针（未使用）
 *   a - 输入值
 * 返回值:
 *   输入值 a
 * 说明:
 *   f(x) = x
 *   常用于:
 *     1. 回归问题的输出层（需要预测连续值）
 *     2. 某些特殊网络结构（如自编码器的某些层）
 *   注意：如果网络中只有线性激活函数，那么整个网络等价于一个线性模型
 */
double genann_act_linear(const struct genann *ann unused, double a) {
    return a;
}

/*
 * 函数功能: 阈值激活函数
 * 参数:
 *   ann - 神经网络指针（未使用）
 *   a - 输入值
 * 返回值:
 *   1 如果 a > 0，否则返回 0
 * 说明:
 *   这是最简单的激活函数，模拟生物神经元的"全或无"特性
 *   历史上用于感知机模型
 *   缺点：不可导，无法使用标准反向传播算法训练
 *   但可以用于测试已经训练好的网络（如test.c中的示例）
 */
double genann_act_threshold(const struct genann *ann unused, double a) {
    return a > 0;
}

/*
 * ============================================================================
 * SIMD 优化的加权和计算函数
 * ============================================================================
 * 
 * 神经网络前向传播的核心计算是：
 *   sum = bias + Σ(weight[k] * input[k])
 * 
 * 这是一个典型的点积运算，可以用SIMD指令集并行加速。
 * 
 * 优化说明:
 * - AVX2: 一次处理4个double（256位寄存器）
 * - SSE2: 一次处理2个double（128位寄存器）
 * - 纯C: 作为后备方案，保证代码可移植性
 * 
 * 注意: 代码中第一个权重被当作偏置处理:
 *   sum = w[0] * -1.0 + Σ(w[k+1] * i[k])
 */

#if defined(GENANN_SIMD_AVX2)
/*
 * 函数功能: 使用AVX2计算加权和（一次处理4个double）
 * 参数:
 *   w_ptr - 权重数组指针的指针（用于更新外部指针）
 *   i - 输入数组指针
 *   n - 输入数量
 * 返回值:
 *   加权和
 * 说明:
 *   w_ptr 是双重指针，函数会更新它指向的位置
 */
static inline double genann_dot_product_avx2(double const **w_ptr, double const *i, int n) {
    double const *w = *w_ptr;
    
    /* 第一个权重是偏置: sum = w[0] * -1.0 */
    double sum = *w++ * -1.0;
    
    /* 计算完整的SIMD向量数量 */
    const int vec_count = n / GENANN_SIMD_WIDTH;
    const int remain = n % GENANN_SIMD_WIDTH;
    
    /* 初始化SIMD累加器为0 */
    __m256d sum_vec = _mm256_setzero_pd();
    
    int k;
    /* 使用AVX2并行计算 */
    for (k = 0; k < vec_count; ++k) {
        /* 从内存加载4个double到SIMD寄存器 */
        __m256d w_vec = _mm256_loadu_pd(w);      /* 未对齐加载 */
        __m256d i_vec = _mm256_loadu_pd(i);
        
        /* 向量乘法: w_vec[i] * i_vec[i] */
        __m256d prod = _mm256_mul_pd(w_vec, i_vec);
        
        /* 累加到sum_vec */
        sum_vec = _mm256_add_pd(sum_vec, prod);
        
        /* 移动指针 */
        w += GENANN_SIMD_WIDTH;
        i += GENANN_SIMD_WIDTH;
    }
    
    /* 将SIMD寄存器中的4个double相加
     * 使用水平加法: sum_vec = [a, b, c, d] -> [a+b, c+d, a+b, c+d] -> [a+b+c+d, ...]
     */
    __m256d hsum = _mm256_hadd_pd(sum_vec, sum_vec);
    __m128d sum_low = _mm256_castpd256_pd128(hsum);
    __m128d sum_high = _mm256_extractf128_pd(hsum, 1);
    __m128d result = _mm_add_pd(sum_low, sum_high);
    
    /* 将结果存入标量变量 */
    double simd_sum;
    _mm_store_sd(&simd_sum, result);
    sum += simd_sum;
    
    /* 处理剩余的元素（不足4个的部分） */
    for (k = 0; k < remain; ++k) {
        sum += *w++ * *i++;
    }
    
    /* 更新外部指针 */
    *w_ptr = w;
    
    return sum;
}
#endif

#if defined(GENANN_SIMD_SSE2)
/*
 * 函数功能: 使用SSE2计算加权和（一次处理2个double）
 * 参数:
 *   w_ptr - 权重数组指针的指针（用于更新外部指针）
 *   i - 输入数组指针
 *   n - 输入数量
 * 返回值:
 *   加权和
 * 说明:
 *   w_ptr 是双重指针，函数会更新它指向的位置
 */
static inline double genann_dot_product_sse2(double const **w_ptr, double const *i, int n) {
    double const *w = *w_ptr;
    
    /* 第一个权重是偏置 */
    double sum = *w++ * -1.0;
    
    const int vec_count = n / GENANN_SIMD_WIDTH;
    const int remain = n % GENANN_SIMD_WIDTH;
    
    /* 初始化SSE累加器 */
    __m128d sum_vec = _mm_setzero_pd();
    
    int k;
    for (k = 0; k < vec_count; ++k) {
        /* 加载2个double */
        __m128d w_vec = _mm_loadu_pd(w);
        __m128d i_vec = _mm_loadu_pd(i);
        
        /* 乘法 */
        __m128d prod = _mm_mul_pd(w_vec, i_vec);
        
        /* 累加 */
        sum_vec = _mm_add_pd(sum_vec, prod);
        
        w += GENANN_SIMD_WIDTH;
        i += GENANN_SIMD_WIDTH;
    }
    
    /* 水平相加: [a, b] -> a + b */
    /* 使用 SSE2 兼容的方式（_mm_hadd_pd 是 SSE3 指令） */
    __m128d high = _mm_unpackhi_pd(sum_vec, sum_vec);
    __m128d sum2 = _mm_add_sd(sum_vec, high);
    double simd_sum;
    _mm_store_sd(&simd_sum, sum2);
    sum += simd_sum;
    
    /* 处理剩余元素 */
    for (k = 0; k < remain; ++k) {
        sum += *w++ * *i++;
    }
    
    /* 更新外部指针 */
    *w_ptr = w;
    
    return sum;
}
#endif

/*
 * 函数功能: 纯C实现的加权和计算（作为后备方案）
 * 参数:
 *   w_ptr - 权重数组指针的指针（用于更新外部指针）
 *   i - 输入数组指针
 *   n - 输入数量
 * 返回值:
 *   加权和
 * 说明:
 *   当编译器不支持AVX2或SSE2时使用此版本
 *   也可以通过不定义GENANN_SIMD_AVX2和GENANN_SIMD_SSE2来强制使用此版本
 */
static inline double genann_dot_product_scalar(double const **w_ptr, double const *i, int n) {
    double const *w = *w_ptr;
    
    /* 第一个权重是偏置 */
    double sum = *w++ * -1.0;
    
    int k;
    for (k = 0; k < n; ++k) {
        sum += *w++ * *i++;
    }
    
    /* 更新外部指针 */
    *w_ptr = w;
    
    return sum;
}

/*
 * 函数功能: 统一的加权和计算接口
 * 参数:
 *   w_ptr - 权重数组指针的指针（用于更新外部指针）
 *   i - 输入数组指针
 *   n - 输入数量
 * 返回值:
 *   加权和
 * 说明:
 *   根据编译时定义的宏自动选择最优的实现版本
 *   优先级: AVX2 > SSE2 > 纯C
 *   函数会自动更新w_ptr指向的权重指针位置
 */
static inline double genann_dot_product(double const **w_ptr, double const *i, int n) {
#if defined(GENANN_SIMD_AVX2)
    return genann_dot_product_avx2(w_ptr, i, n);
#elif defined(GENANN_SIMD_SSE2)
    return genann_dot_product_sse2(w_ptr, i, n);
#else
    return genann_dot_product_scalar(w_ptr, i, n);
#endif
}

/*
 * 函数功能: 创建并初始化一个新的神经网络
 * 参数:
 *   inputs - 输入层神经元数量
 *   hidden_layers - 隐藏层数量（可以为0，表示没有隐藏层）
 *   hidden - 每个隐藏层的神经元数量
 *   outputs - 输出层神经元数量
 * 返回值:
 *   成功返回神经网络指针，失败返回NULL
 * 
 * 内存布局说明:
 *   genann结构体 + 权重数组 + 输出数组 + delta数组
 *   所有内存一次性分配，提高缓存局部性和分配效率
 * 
 * 网络结构示例 (2输入, 1隐藏层, 2隐藏神经元, 1输出):
 *   输入层:  [i1, i2]
 *                \  /
 *   隐藏层:    [h1, h2]
 *                  \  /
 *   输出层:       [o1]
 * 
 * 权重存储顺序:
 *   1. 输入层到第一隐藏层的权重（包含偏置）
 *   2. 隐藏层之间的权重（如果有多个隐藏层）
 *   3. 最后隐藏层到输出层的权重
 */
genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs) {
    /* 参数有效性检查 */
    if (hidden_layers < 0) return 0;
    if (inputs < 1) return 0;
    if (outputs < 1) return 0;
    if (hidden_layers > 0 && hidden < 1) return 0;

    /* 计算隐藏层权重总数
     * 公式说明:
     * - (inputs+1) * hidden: 输入层到第一隐藏层（+1是因为偏置）
     * - (hidden_layers-1) * (hidden+1) * hidden: 隐藏层之间的连接
     */
    const int hidden_weights = hidden_layers ? (inputs+1) * hidden + (hidden_layers-1) * (hidden+1) * hidden : 0;
    
    /* 计算输出层权重总数
     * 如果有隐藏层: (hidden+1) * outputs（+1是偏置）
     * 如果没有隐藏层: (inputs+1) * outputs
     */
    const int output_weights = (hidden_layers ? (hidden+1) : (inputs+1)) * outputs;
    
    /* 总权重数 = 隐藏层权重 + 输出层权重 */
    const int total_weights = (hidden_weights + output_weights);

    /* 计算总神经元数（包含输入层） */
    const int total_neurons = (inputs + hidden * hidden_layers + outputs);

    /* 计算需要分配的总内存大小
     * 包含: genann结构体 + 权重数组 + 输出数组 + delta数组
     * 注意: delta数组不包含输入层（输入层没有误差）
     */
    const int size = sizeof(genann) + sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
    genann *ret = malloc(size);
    if (!ret) return 0;

    /* 初始化网络结构参数 */
    ret->inputs = inputs;
    ret->hidden_layers = hidden_layers;
    ret->hidden = hidden;
    ret->outputs = outputs;

    ret->total_weights = total_weights;
    ret->total_neurons = total_neurons;

    /* 设置各数据缓冲区的指针
     * 内存布局:
     * [genann结构体][权重数组][输出数组][delta数组]
     */
    ret->weight = (double*)((char*)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;

    /* 随机初始化所有权重 */
    genann_randomize(ret);

    /* 设置默认激活函数（使用查找表的sigmoid） */
    ret->activation_hidden = genann_act_sigmoid_cached;
    ret->activation_output = genann_act_sigmoid_cached;

    /* 初始化sigmoid查找表 */
    genann_init_sigmoid_lookup(ret);

    return ret;
}


/*
 * 函数功能: 从文件中读取并创建神经网络
 * 参数:
 *   in - 已打开的文件指针（使用genann_write保存的文件）
 * 返回值:
 *   成功返回神经网络指针，失败返回NULL
 * 文件格式:
 *   第一行: inputs hidden_layers hidden outputs
 *   后续: 所有权重值（科学计数法）
 */
genann *genann_read(FILE *in) {
    int inputs, hidden_layers, hidden, outputs;
    int rc;

    /* 读取网络结构参数 */
    errno = 0;
    rc = fscanf(in, "%d %d %d %d", &inputs, &hidden_layers, &hidden, &outputs);
    if (rc < 4 || errno != 0) {
        perror("fscanf");
        return NULL;
    }

    /* 创建神经网络 */
    genann *ann = genann_init(inputs, hidden_layers, hidden, outputs);

    /* 读取所有权重值 */
    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        errno = 0;
        /* %le 读取double类型的科学计数法表示 */
        rc = fscanf(in, " %le", ann->weight + i);
        if (rc < 1 || errno != 0) {
            perror("fscanf");
            genann_free(ann);
            return NULL;
        }
    }

    return ann;
}


/*
 * 函数功能: 创建神经网络的深拷贝
 * 参数:
 *   ann - 原神经网络指针
 * 返回值:
 *   成功返回新的神经网络副本指针，失败返回NULL
 * 说明:
 *   复制包括: 结构体、所有权重、输出、delta
 *   注意需要重新设置指针，因为新网络在不同的内存位置
 */
genann *genann_copy(genann const *ann) {
    /* 计算需要分配的内存大小（与原网络相同） */
    const int size = sizeof(genann) + sizeof(double) * (ann->total_weights + ann->total_neurons + (ann->total_neurons - ann->inputs));
    genann *ret = malloc(size);
    if (!ret) return 0;

    /* 一次性复制所有内容 */
    memcpy(ret, ann, size);

    /* 重新设置指针（因为新内存的地址不同） */
    ret->weight = (double*)((char*)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;

    return ret;
}


/*
 * 函数功能: 随机初始化神经网络的所有权重
 * 参数:
 *   ann - 神经网络指针
 * 说明:
 *   权重范围: -0.5 到 0.5
 *   使用GENANN_RANDOM宏生成随机数，可在编译时替换
 *   此函数在genann_init中自动调用
 */
void genann_randomize(genann *ann) {
    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        double r = GENANN_RANDOM();
        /* 将[0,1]范围映射到[-0.5, 0.5] */
        ann->weight[i] = r - 0.5;
    }
}


/*
 * 函数功能: 释放神经网络使用的所有内存
 * 参数:
 *   ann - 神经网络指针
 * 说明:
 *   由于weight、output、delta都指向同一块内存的不同位置
 *   所以只需要free(ann)一次即可释放所有内存
 */
void genann_free(genann *ann) {
    /* 所有缓冲区都是在同一次malloc中分配的，所以只需一次free */
    free(ann);
}


/*
 * 函数功能: 执行前向传播算法，计算神经网络的输出
 * 参数:
 *   ann - 神经网络指针
 *   inputs - 输入数据数组（大小: ann->inputs）
 * 返回值:
 *   输出层结果数组指针（大小: ann->outputs）
 * 
 * 前向传播算法说明:
 *   对于每层的每个神经元j:
 *   1. 计算加权和: sum = bias + Σ(weight_ij * input_i)
 *      - 注意: 第一个权重是偏置(bias)，乘以-1.0
 *   2. 应用激活函数: output_j = activation(sum)
 * 
 * 网络计算流程:
 *   输入层 -> [隐藏层...] -> 输出层
 * 
 * 设计特点:
 *   1. 将输入数据复制到output数组的开头，使第一层不需要特殊处理
 *   2. 使用指针算术进行高效遍历
 *   3. 可以处理有隐藏层和无隐藏层的情况
 */
double const *genann_run(genann const *ann, double const *inputs) {
    /* 权重指针，从第一个权重开始 */
    double const *w = ann->weight;
    /* 输出指针，跳过输入层，指向第一个隐藏层或输出层 */
    double *o = ann->output + ann->inputs;
    /* 输入指针，指向当前层的输入（上一层的输出） */
    double const *i = ann->output;

    /* 将输入数据复制到output数组的开头
     * 这样处理后，输入层的"输出"就是输入数据本身
     * 使得第一层的计算逻辑与其他层保持一致
     */
    memcpy(ann->output, inputs, sizeof(double) * ann->inputs);

    int h, j;

    /* 情况1: 没有隐藏层（单层感知机）
     * 直接从输入层计算到输出层
     */
    if (!ann->hidden_layers) {
        /* 保存输出层起始位置，用于返回 */
        double *ret = o;
        /* 遍历每个输出神经元 */
        for (j = 0; j < ann->outputs; ++j) {
            /* 使用优化的点积计算（支持SIMD）
             * genann_dot_product会自动更新权重指针w
             */
            double sum = genann_dot_product(&w, i, ann->inputs);
            /* 应用输出层激活函数 */
            *o++ = genann_act_output(ann, sum);
        }

        return ret;
    }

    /* 情况2: 有隐藏层的网络 */

    /* 第一步: 计算第一隐藏层
     * 输入: ann->output（刚刚复制的输入数据）
     * 输出: ann->output + ann->inputs
     */
    for (j = 0; j < ann->hidden; ++j) {
        /* 使用优化的点积计算 */
        double sum = genann_dot_product(&w, i, ann->inputs);
        *o++ = genann_act_hidden(ann, sum);
    }

    /* 移动输入指针，现在i指向第一隐藏层的输出 */
    i += ann->inputs;

    /* 第二步: 计算后续隐藏层（如果有多个隐藏层）
     * 从第2个隐藏层开始，到最后一个隐藏层
     */
    for (h = 1; h < ann->hidden_layers; ++h) {
        /* 遍历当前隐藏层的每个神经元 */
        for (j = 0; j < ann->hidden; ++j) {
            /* 使用优化的点积计算 */
            double sum = genann_dot_product(&w, i, ann->hidden);
            *o++ = genann_act_hidden(ann, sum);
        }

        /* 移动输入指针到下一层 */
        i += ann->hidden;
    }

    /* 保存输出层起始位置，用于返回 */
    double const *ret = o;

    /* 第三步: 计算输出层
     * 输入: 最后一个隐藏层的输出
     * 输出: 最终预测结果
     */
    for (j = 0; j < ann->outputs; ++j) {
        /* 使用优化的点积计算 */
        double sum = genann_dot_product(&w, i, ann->hidden);
        *o++ = genann_act_output(ann, sum);
    }

    /* 调试断言：确保正确使用了所有的权重和输出 */
    assert(w - ann->weight == ann->total_weights);
    assert(o - ann->output == ann->total_neurons);

    return ret;
}


/*
 * 函数功能: 执行单次反向传播训练
 * 参数:
 *   ann - 神经网络指针
 *   inputs - 输入数据数组
 *   desired_outputs - 期望输出数组（标签）
 *   learning_rate - 学习率（决定每次更新的步长）
 * 
 * 反向传播算法原理:
 *   反向传播是一种用于训练神经网络的监督学习算法
 *   核心思想: 计算输出误差，然后从输出层向输入层反向传播误差，
 *             最后根据梯度下降法更新权重
 * 
 * 算法步骤:
 *   1. 前向传播: 计算网络输出
 *   2. 计算输出层误差增量 (delta)
 *   3. 反向传播: 从输出层向输入层计算各隐藏层的误差增量
 *   4. 更新权重: 根据误差增量和学习率更新所有连接权重
 * 
 * 数学公式:
 *   对于输出层神经元j:
 *     delta_j = (desired_j - output_j) * activation_derivative(output_j)
 *     
 *   对于隐藏层神经元j:
 *     delta_j = output_j * (1 - output_j) * Σ(delta_k * weight_jk)
 *               （activation_derivative * 后续层的加权误差）
 *     
 *   权重更新:
 *     weight_new = weight_old + learning_rate * delta_j * input_i
 * 
 * 说明:
 *   本实现假设激活函数是sigmoid，其导数为: f'(x) = f(x) * (1 - f(x))
 *   对于线性激活函数，导数为1，所以delta = desired - output
 */
void genann_train(genann const *ann, double const *inputs, double const *desired_outputs, double learning_rate) {
    /* 第一步: 前向传播，计算网络输出
     * 这会填充 ann->output 数组
     */
    genann_run(ann, inputs);

    int h, j, k;

    /* 第二步: 计算输出层的误差增量 (delta)
     * 
     * 输出层delta计算:
     * - 线性激活: delta = desired_output - actual_output
     * - sigmoid激活: delta = (desired - actual) * actual * (1 - actual)
     *   因为sigmoid的导数是 f(x)*(1-f(x))
     */
    {
        /* o: 指向输出层的第一个输出 */
        double const *o = ann->output + ann->inputs + ann->hidden * ann->hidden_layers;
        /* d: 指向输出层的第一个delta */
        double *d = ann->delta + ann->hidden * ann->hidden_layers;
        /* t: 指向期望输出（标签） */
        double const *t = desired_outputs;

        /* 根据激活函数类型选择不同的delta计算方式 */
        if (genann_act_output == genann_act_linear ||
                ann->activation_output == genann_act_linear) {
            /* 线性激活函数: 导数为1 */
            for (j = 0; j < ann->outputs; ++j) {
                *d++ = *t++ - *o++;
            }
        } else {
            /* sigmoid激活函数: delta = (t - o) * o * (1 - o) */
            for (j = 0; j < ann->outputs; ++j) {
                *d++ = (*t - *o) * *o * (1.0 - *o);
                ++o; ++t;
            }
        }
    }


    /* 第三步: 反向传播，计算隐藏层的误差增量
     * 从最后一个隐藏层开始，向前传播到第一个隐藏层
     * 
     * 隐藏层delta计算原理:
     * 隐藏层神经元j的误差 = 所有后续层神经元k的误差 * weight_jk 的加权和
     * 然后乘以当前神经元输出的导数
     * 
     * 公式:
     *   delta_j = output_j * (1 - output_j) * Σ(delta_k * weight_jk)
     *   其中:
     *     - output_j * (1 - output_j) 是sigmoid的导数
     *     - weight_jk 是神经元j到神经元k的权重
     */
    for (h = ann->hidden_layers - 1; h >= 0; --h) {

        /* o: 当前隐藏层的第一个输出 */
        double const *o = ann->output + ann->inputs + (h * ann->hidden);
        /* d: 当前隐藏层的第一个delta */
        double *d = ann->delta + (h * ann->hidden);

        /* dd: 下一层（更靠近输出层）的第一个delta
         * 用于计算当前层的误差
         */
        double const * const dd = ann->delta + ((h+1) * ann->hidden);

        /* ww: 下一层使用的权重起始位置
         * 用于获取连接当前层和下一层的权重
         */
        double const * const ww = ann->weight + ((ann->inputs+1) * ann->hidden) + ((ann->hidden+1) * ann->hidden * (h));

        /* 遍历当前隐藏层的每个神经元 */
        for (j = 0; j < ann->hidden; ++j) {

            double delta = 0;

            /* 计算后续层（可能是隐藏层或输出层）对当前神经元的误差贡献
             * 
             * 对于最后一个隐藏层: 后续层是输出层，数量为ann->outputs
             * 对于其他隐藏层: 后续层是隐藏层，数量为ann->hidden
             */
            for (k = 0; k < (h == ann->hidden_layers-1 ? ann->outputs : ann->hidden); ++k) {
                const double forward_delta = dd[k];
                /* 计算权重索引
                 * 每个后续层神经元有 (hidden + 1) 个权重（+1是偏置）
                 * (j + 1) 跳过偏置权重，因为偏置不参与反向传播
                 */
                const int windex = k * (ann->hidden + 1) + (j + 1);
                const double forward_weight = ww[windex];
                /* 累加误差贡献 */
                delta += forward_delta * forward_weight;
            }

            /* 乘以当前神经元输出的导数（sigmoid导数）
             * delta_j = output_j * (1 - output_j) * accumulated_error
             */
            *d = *o * (1.0-*o) * delta;
            ++d; ++o;
        }
    }


    /* 第四步: 更新输出层的权重
     * 
     * 权重更新公式:
     *   w_new = w_old + learning_rate * delta * input
     *   
     * 对于偏置权重:
     *   偏置的"输入"始终为1，所以:
     *   bias_new = bias_old + learning_rate * delta * (-1.0)
     *   （乘以-1.0是因为代码中偏置权重是 weight * -1.0 的方式使用的）
     */
    {
        /* d: 输出层的delta */
        double const *d = ann->delta + ann->hidden * ann->hidden_layers;

        /* w: 输出层权重的起始位置 */
        double *w = ann->weight + (ann->hidden_layers
                ? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * ann->hidden * (ann->hidden_layers-1))
                : (0));

        /* i: 输出层的输入（最后一个隐藏层的输出，或者输入层） */
        double const * const i = ann->output + (ann->hidden_layers
                ? (ann->inputs + (ann->hidden) * (ann->hidden_layers-1))
                : 0);

        /* 遍历每个输出神经元 */
        for (j = 0; j < ann->outputs; ++j) {
            /* 更新偏置权重 */
            *w++ += *d * learning_rate * -1.0;
            /* 更新其他权重 */
            for (k = 1; k < (ann->hidden_layers ? ann->hidden : ann->inputs) + 1; ++k) {
                *w++ += *d * learning_rate * i[k-1];
            }

            ++d;
        }

        /* 断言：确保更新了所有输出层权重 */
        assert(w - ann->weight == ann->total_weights);
    }


    /* 第五步: 更新隐藏层的权重
     * 从最后一个隐藏层开始，向前更新到第一个隐藏层
     * 
     * 更新逻辑与输出层类似，但输入来自更前一层
     */
    for (h = ann->hidden_layers - 1; h >= 0; --h) {

        /* d: 当前隐藏层的delta */
        double const *d = ann->delta + (h * ann->hidden);

        /* i: 当前隐藏层的输入（上一层的输出）
         *   - 第一个隐藏层: 输入是原始输入
         *   - 其他隐藏层: 输入是前一个隐藏层的输出
         */
        double const *i = ann->output + (h
                ? (ann->inputs + ann->hidden * (h-1))
                : 0);

        /* w: 当前隐藏层权重的起始位置 */
        double *w = ann->weight + (h
                ? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * (ann->hidden) * (h-1))
                : 0);

        /* 遍历当前隐藏层的每个神经元 */
        for (j = 0; j < ann->hidden; ++j) {
            /* 更新偏置权重 */
            *w++ += *d * learning_rate * -1.0;
            /* 更新其他权重 */
            for (k = 1; k < (h == 0 ? ann->inputs : ann->hidden) + 1; ++k) {
                *w++ += *d * learning_rate * i[k-1];
            }
            ++d;
        }

    }

}


/*
 * 函数功能: 将神经网络保存到文件
 * 参数:
 *   ann - 神经网络指针
 *   out - 已打开的文件指针（用于写入）
 * 文件格式:
 *   第一部分: inputs hidden_layers hidden outputs
 *   第二部分: 所有权重值（科学计数法，20位精度）
 * 说明:
 *   可以使用genann_read从文件中恢复网络
 */
void genann_write(genann const *ann, FILE *out) {
    /* 写入网络结构参数 */
    fprintf(out, "%d %d %d %d", ann->inputs, ann->hidden_layers, ann->hidden, ann->outputs);

    /* 写入所有权重值，使用科学计数法保证精度 */
    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        fprintf(out, " %.20e", ann->weight[i]);
    }
}
