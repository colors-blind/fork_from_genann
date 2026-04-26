#include <stdio.h>
#include <stdlib.h>
#include "genann.h"

int main() {
    printf("=== C 版本一致性测试 ===\n\n");

    // 测试 1: XOR 网络（与 test.c 中的相同）
    printf("测试 1: XOR 网络（使用阈值激活函数）\n");
    genann *ann = genann_init(2, 1, 2, 1);
    ann->activation_hidden = genann_act_threshold;
    ann->activation_output = genann_act_threshold;

    // 设置与 test.c 中相同的权重
    ann->weight[0] = 0.5;  // 隐藏 0 bias
    ann->weight[1] = 1.0;  // 隐藏 0 w1
    ann->weight[2] = 1.0;  // 隐藏 0 w2
    ann->weight[3] = 1.0;  // 隐藏 1 bias
    ann->weight[4] = 1.0;  // 隐藏 1 w1
    ann->weight[5] = 1.0;  // 隐藏 1 w2
    ann->weight[6] = 0.5;  // 输出 bias
    ann->weight[7] = 1.0;  // 输出 w1
    ann->weight[8] = -1.0; // 输出 w2

    printf("  网络结构: 2 输入, 1 隐藏层(2 神经元), 1 输出\n");
    printf("  总权重数: %d\n", ann->total_weights);

    // 测试 XOR
    double input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double expected[4] = {0, 1, 1, 0};
    int passed = 1;

    printf("\n  测试结果:\n");
    for (int i = 0; i < 4; i++) {
        double output = *genann_run(ann, input[i]);
        printf("    输入 [%.0f, %.0f] -> 输出 %.0f (期望 %.0f)", 
               input[i][0], input[i][1], output, expected[i]);
        if (output == expected[i]) {
            printf(" ✓\n");
        } else {
            printf(" ✗\n");
            passed = 0;
        }
    }

    // 保存网络到文件
    printf("\n  保存网络到 xor_network.txt...\n");
    FILE *f = fopen("xor_network.txt", "w");
    genann_write(ann, f);
    fclose(f);

    genann_free(ann);

    // 测试 2: 简单感知机
    printf("\n测试 2: 简单感知机（无隐藏层）\n");
    genann *ann2 = genann_init(2, 0, 0, 1);
    
    // 设置权重: bias=0, w1=2, w2=3
    ann2->weight[0] = 0.0;  // bias
    ann2->weight[1] = 2.0;  // w1
    ann2->weight[2] = 3.0;  // w2

    printf("  网络结构: 2 输入, 0 隐藏层, 1 输出\n");
    printf("  权重: [bias=%.1f, w1=%.1f, w2=%.1f]\n", 
           ann2->weight[0], ann2->weight[1], ann2->weight[2]);

    // 计算: output = sigmoid(-bias + w1*x1 + w2*x2)
    // 输入 [1, 0]: sum = 0 + 2*1 + 3*0 = 2, sigmoid(2) ≈ 0.8808
    // 输入 [0, 1]: sum = 0 + 2*0 + 3*1 = 3, sigmoid(3) ≈ 0.9526
    // 输入 [0, 0]: sum = 0, sigmoid(0) = 0.5

    double test_inputs[3][2] = {{1, 0}, {0, 1}, {0, 0}};
    printf("\n  测试结果 (sigmoid 激活):\n");
    for (int i = 0; i < 3; i++) {
        double output = *genann_run(ann2, test_inputs[i]);
        printf("    输入 [%.0f, %.0f] -> 输出 %.6f\n", 
               test_inputs[i][0], test_inputs[i][1], output);
    }

    // 保存到文件
    printf("\n  保存网络到 perceptron_network.txt...\n");
    f = fopen("perceptron_network.txt", "w");
    genann_write(ann2, f);
    fclose(f);

    genann_free(ann2);

    printf("\n=== C 版本测试完成 ===\n");
    if (passed) {
        printf("所有测试通过 ✓\n");
        return 0;
    } else {
        printf("部分测试失败 ✗\n");
        return 1;
    }
}