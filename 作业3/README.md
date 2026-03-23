# 大数据原理与技术 — 作业3（聚类算法对比实验）

## 作业要求（来自 `作业3要求.jpg`）

- 在鸢尾花数据集上实现 `K-means` 和 `DBSCAN` 聚类算法
- 调整超参数（如簇数、邻域半径等），观察聚类结果变化
- 用准确率、轮廓系数和 Calinski-Harabasz 指数评价性能
- 禁止直接调用现成工具包中的完整聚类接口，需要自己实现核心逻辑
- 提交：代码 + 简单报告（2 页以上）
- 压缩包命名格式：`学号+姓名+作业3.zip`
- 邮箱：`BDSysu2026Spring@163.com`

## 环境（Anaconda `ml`）

本作业默认使用你机器上的 conda 环境：`ml`。

### 依赖安装（仅需一次）

```bash
conda run -n ml python -m pip install -r requirements.txt
```

## 目录结构

- `src/load_data.py`：加载 Iris 数据集
- `src/kmeans.py`：自实现 K-means
- `src/dbscan.py`：自实现 DBSCAN
- `src/metrics.py`：自实现聚类准确率、轮廓系数、Calinski-Harabasz 指数
- `src/visualize.py`：绘制聚类散点图
- `src/run_all.py`：一键运行全部实验并导出结果
- `tests/`：最小单元测试
- `outputs/`：实验图像和指标汇总结果
- `report.md`：提交报告

## 运行

### 1) 运行测试

```bash
conda run -n ml python -m unittest discover -s tests -v
```

### 2) 运行全部实验

```bash
conda run -n ml python src/run_all.py
```

运行完成后会在 `outputs/` 下生成：

- `kmeans_k2.png` ~ `kmeans_k5.png`
- `dbscan_eps0_3_min3.png` 等参数组合图
- `metrics_summary.json`
- `metrics_summary.md`

## 实验设置

### K-means 参数

- `k = 2, 3, 4, 5`
- `max_iter = 200`
- `tol = 1e-6`

### DBSCAN 参数

- `(eps=0.3, min_samples=3)`
- `(eps=0.5, min_samples=4)`
- `(eps=0.6, min_samples=5)`
- `(eps=0.8, min_samples=5)`

## 说明

- 本作业只使用 `sklearn.datasets.load_iris` 加载数据，不使用 `sklearn.cluster.KMeans` 或 `sklearn.cluster.DBSCAN`
- 三个评价指标均在 `src/metrics.py` 中自行实现
- 图像默认展示 Iris 的前两个特征（花萼长度、花萼宽度）上的聚类结果，便于直观对比

## 打包提交

建议把以下内容放进压缩包：

- `src/`
- `tests/`
- `outputs/`
- `README.md`
- `report.md`

然后按要求命名：`学号+姓名+作业3.zip`
