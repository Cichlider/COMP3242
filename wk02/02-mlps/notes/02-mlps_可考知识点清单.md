# Week 2 可考知识点清单

## 1. 二分类与 Logistic Regression
- 线性分类边界：`a^T x + b = 0`。
- 概率输出：`\sigma(z)=1/(1+e^{-z})`，其中 `z=a^T x+b`。
- 交叉熵损失与最大似然等价关系（会写出目标函数）。

## 2. 正则化与概率校准
- L2 正则项对目标函数的影响。
- 温度缩放对 softmax/sigmoid 输出分布的影响方向。

## 3. XOR 与模型表达能力
- 为什么线性分类器不能解决 XOR。
- 通过隐藏层实现非线性可分的思路。

## 4. MLP 结构与参数计算
- 两层 MLP 前向形式：`y=c^T\phi(Ax+b)+d`。
- 给定层宽时参数总数计算（权重+偏置）。
- 激活函数对可表达性与梯度传播的影响（sigmoid/tanh/ReLU）。

## 5. 多分类与多标签
- softmax + CE（多分类）与 sigmoid + BCE（多标签）的建模差异。
- logits、概率、标签编码（one-hot 或 class index）对应关系。

## 6. 梯度与反向传播
- 梯度定义与负梯度更新规则。
- 链式法则在多层网络中的展开顺序。
- 手推关键梯度：`dL/dc, dL/dd, dL/dA, dL/db`（会写矩阵维度）。

## 7. 训练循环与代码题高频点
- 训练步骤顺序：`zero_grad -> forward -> loss -> backward -> step`。
- `CrossEntropyLoss` 输入 logits、标签为整型类索引。
- 常见错误：shape 不一致、标签类型错误、忘记清梯度。
