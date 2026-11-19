> **Language:** [English](README.md) · 中文

# CDFA —— 定制化动态滤波器增强

## CDFA 是什么

CDFA 是一种面向时间序列预测的数据增强方法。它按数据相关的权重分布，重新分配一个训练窗口自身的一阶差分，扰动强度由单个全局幅度控制。闭式形式为

```
d' = d + 1/2 * alpha * Delta(phi .* d)
```

其中 `d` 是窗口的一阶差分，`phi` 由局部波动率构造、并缩放到单位均值，`alpha` 是唯一的全局幅度（所有实验取 `0.3`）。扰动作用在增量上而不是水平值上，因此窗口的水平以及它与后继值之间的对齐关系都保持不变。每个窗口只改写一次，训练样本库的大小不变。

## 它用来做什么

- 在标注历史有限时，对训练窗口做受控扰动；只需一个标量就能设定扰动强度，不需要采集新观测。
- 作为与学习器无关的预处理步骤。实验中使用距离加权的 kNN 和一个用 NumPy 写的小型单层 LSTM，覆盖单步滚动预测与直接多步预测。
- 在股票面板上与已发表的数据增强方法做对照，包括窗口扭曲、幅度扭曲，以及频域的掩码与混合。

实验覆盖八个单变量 Darts 序列、2012–2025 年间固定的 20 只标普 500 成分股（280 个「股票-年份」单元），以及公开的长周期预测面板。结果以 MAE 报告，并给出配对改进、bootstrap 置信区间、胜率和 Wilcoxon 检验。

## 仓库结构

```
.
|-- README.md
|-- Datasets/                          外部数据集（只读输入）
|   |-- Darts/                          八条单变量序列
|   |-- Stocks/                         标普 500 复权收盘价面板
|   `-- ETTs/                           公开预测面板
`-- paper 'Magnitude... CDFA' codes/    复现包：代码 + 结果
    |-- README.md
    |-- main.py, cdfa.py, ...           源代码
    |-- augmentations.py, compare_augmentations.py
    `-- outputs/                        表格、逐单元结果、运行日志
```

仓库里的其他文件夹与结果复现无关。

## 复现包

`paper 'Magnitude... CDFA' codes/` 存放投稿论文 **"Magnitude-Controlled Increment Redistribution for Time-Series by Adapting Advanced Filtering Techniques"** 所对应的源代码、数值结果和运行日志。论文中的每张表都能由这些脚本重新生成；`outputs/` 下的 CSV 就是报告数字所用的逐单元结果与汇总统计。该文件夹内的 README 列出了文件、命令和评估协议。

## 数据

`Datasets/` 是唯一的数据来源。`datasets.py` 从文件自身位置解析这些路径，因此只要保持上述相对结构，整个项目可以移动。数据文件只作为输入，不会被写入。

## 运行环境

Python 3.13 或更高版本，需要 NumPy、pandas、SciPy 和 scikit-learn。Darts 表格还会写 xlsx 文件，可用时使用 openpyxl。不需要预训练权重、API 密钥或外部服务。
