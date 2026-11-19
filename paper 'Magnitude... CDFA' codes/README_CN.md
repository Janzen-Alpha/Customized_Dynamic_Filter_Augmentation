> **Language:** [English](README.md) · 中文

# 复现包

本文件夹存放投稿论文 **"Magnitude-Controlled Increment Redistribution for Time-Series by Adapting Advanced Filtering Techniques"** 所对应的源代码、数值结果和运行日志。

CDFA 按数据相关的权重分布，重新分配一个训练窗口自身的一阶差分，扰动强度由单个全局幅度控制。闭式形式为

```
d' = d + 1/2 * alpha * Delta(phi .* d)
```

实验有两个入口：`main.py` 运行三组数据集，`compare_augmentations.py` 在股票面板上运行与已发表增强方法的对比。论文中的每张表都能由下面的脚本重新生成；`outputs/` 下的 CSV 和日志就是报告数字所用的结果。

## 目录结构

```
paper 'Magnitude... CDFA' codes/
|-- README.md
|-- main.py                   入口；按数据集族分发
|-- cdfa.py                   方法：权重构造与重分配规则
|-- models.py                 学习器：单步 kNN、直接多步 kNN、NumPy LSTM
|-- datasets.py               三族数据的加载与单元切分
|-- evaluate.py               滚动评估协议与并行驱动
|-- tables.py                 配对统计与三张汇总表
|-- darts_pipeline.py         Darts 流程（8:2 划分，直接预测）
|-- augmentations.py          对照方法：窗口/幅度扭曲、FrAug
|-- compare_augmentations.py  股票面板上与已发表增强方法的对比
`-- outputs/                  表格、逐单元结果、运行日志
```

数据集在上一级，即仓库根目录的 `Datasets/`。`datasets.py` 从文件自身位置解析路径，因此只要保持相对结构，整个项目可以移动。

## 运行

从仓库根目录运行。`main.py` 把 `--out` 相对仓库根解析，所以要显式指定该路径，才能把结果写进本文件夹的 `outputs/`。若省略，默认写到仓库根的 `outputs/`。

```bash
python "paper 'Magnitude... CDFA' codes/main.py" darts  --out "paper 'Magnitude... CDFA' codes/outputs"
python "paper 'Magnitude... CDFA' codes/main.py" stocks --out "paper 'Magnitude... CDFA' codes/outputs"
python "paper 'Magnitude... CDFA' codes/main.py" panels --out "paper 'Magnitude... CDFA' codes/outputs"
python "paper 'Magnitude... CDFA' codes/main.py" all    --out "paper 'Magnitude... CDFA' codes/outputs"

python "paper 'Magnitude... CDFA' codes/compare_augmentations.py" --workers 12
```

`main.py` 接受 `darts`、`stocks`、`panels` 或 `all`，另有 `--workers N`（并行线程数，默认 1）和 `--out DIR`。控制台输出会写入 `outputs/log_<family>.txt`。

`compare_augmentations.py` 默认写到本文件夹的 `outputs/`，接受 `--workers N`（默认 12）、`--out DIR` 和 `--from-cells`。12 个线程跑完约需八分钟；`--from-cells` 跳过评估，几秒内从 `cells_augmentations.csv` 重建表格和日志。

## 三组数据集

**darts** —— Darts 集合中的八条单变量序列，按 8:2 划分。回溯长度和邻居数在训练尾部选定，预测为直接多步。评估四个模型：kNN、CDFA+kNN、LSTM、CDFA+LSTM；其中 LSTM 是用 NumPy 写的小型单层网络。误差以原始单位报告，同时给出相对朴素预测的算术均值和几何均值。

**stocks** —— 标普 500 复权收盘价面板，固定取 20 只成分股，时间跨度 2012–2025。一个评估单元是一个日历年，共 280 个「股票-年份」单元。

**panels** —— 公开的长周期预测数据集。一个评估单元是一个日历月；低频面板则用一个日历年。通道按均值水平系统抽样，并始终保留基准目标列。

股票和面板共用同一滚动协议：训练窗口 300 个观测，学习器输入长度 15，距离加权 kNN 取 5 个邻居，预测裁剪到前一个观测的 `[0.67, 1.5]` 倍，窗口按若干观测向前滚动。权重向量在每个单元开始时计算一次并保持固定，不按数据集调参。

已纳入的面板为 `Fred-MD`、`weather`、`ETTm2`、`Air-Quality` 和 `ETTh2`。更多面板（`electricity`、`solar`、`M4`）已在 `datasets.py` 中描述但注释掉了；把文件放进 `Datasets/ETTs/`，取消注释，并把名字加入 `PANEL_ORDER` 即可。

## 与已发表增强方法的对比

`compare_augmentations.py` 在同一批 280 个股票面板单元上评估九种增强设置：不增强、CDFA、窗口扭曲（两个切片比例）、幅度扭曲（三种强度），以及 FrAug 的频域掩码与混合。每种设置都把每个库窗口改写一次，因此各设置的样本库大小相同。`augmentations.py` 存放四种对照变换。

其中六行进入论文表格，另外三行是强度检查。CSV 还给出 *dose* 一列，即窗口绝对改变量除以其自身标准差，用来直接读出每种变换的强度。在该面板上 CDFA 的 dose 为 0.11，对照方法在 0.10 到 0.36 之间。

## 输出

| 文件 | 内容 |
|---|---|
| `table2_equity_panel.csv` | 股票面板上每个方法的 MAE、改进及 95% CI、胜率、Wilcoxon p |
| `table3_public_panels.csv` | 每个数据集上 CDFA 的改进及 CI、胜率，并列两个对照滤波器 |
| `table4_head_to_head.csv` | 每个数据集上 CDFA 相对各对照滤波器的领先百分点及胜出比例 |
| `knn_lstm_cdfa_mae_table.csv/.xlsx` | Darts 表格：原始单位 MAE 与缩放均值 |
| `table_augmentations.csv` | 与已发表增强方法的对比：MAE、改进及 95% CI、胜率、CDFA 胜率、p 值与 dose；六行入表，三行强度检查 |
| `cells_stocks.csv`, `cells_panels.csv` | 表 2–4 背后的逐单元 MAE |
| `cells_augmentations.csv` | 2520 行：九种增强设置的逐单元 MAE、dose 和起点数 |
| `log_*.txt` | 每次运行的控制台输出，含 `log_augmentations.txt` |

统计在同一评估单元上配对。方法的改进定义为各单元 `(baseline - method) / baseline` 的均值；区间是对单元做 5000 次 bootstrap（种子 42）；胜率是改进为正的单元占比；p 值来自配对 Wilcoxon 符号秩检验。对照值数值为零的单元无法计算比率，予以剔除。

## 运行环境

Python 3.13 或更高版本，需要 NumPy、pandas、SciPy 和 scikit-learn。Darts 表格还会写 xlsx 文件，可用时使用 openpyxl。不需要预训练权重或外部服务。
