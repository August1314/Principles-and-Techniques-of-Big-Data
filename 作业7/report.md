# 作业7：混合推荐系统设计

**作者**：梁力航  
**学号**：23336128  
**日期**：2026年4月

---

## 1. 实验目的

本实验基于 MovieLens 100K 数据集，完成四种推荐方案的实现与对比：

1. 用户协同过滤（User-CF）
2. 物品协同过滤（Item-CF）
3. 基于内容的推荐（Content-Based）
4. 混合推荐（Hybrid，作为加分项）

核心评价目标是对比 `Recall@10`，并结合多样性、覆盖率和新颖度分析不同方法的取舍。混合推荐部分只做基于真实结果的补充分析，不扩展到未验证的结论。

---

## 2. 数据与实验设置

### 2.1 数据集

使用 MovieLens 100K 数据集。

| 数据项 | 数量 |
|--------|------|
| 用户数 | 943 |
| 电影数 | 1682 |
| 评分数 | 100000 |
| 电影类型 | 19 |

### 2.2 数据划分

- 训练集：80000 条评分
- 测试集：20000 条评分
- 评估用户数：920

### 2.3 评价口径

- `Recall@10` 以测试集中评分 `>= 4` 的物品作为正样本
- 多样性、覆盖率、新颖度均按**全部 920 个评估用户**上的方法级平均值计算
- 这意味着报告中的多样性不是单个用户样本值，而是全体评估用户的真实汇总结果

---

## 3. 方法实现

### 3.1 用户协同过滤（User-CF）

通过用户之间的余弦相似度，寻找与目标用户兴趣相近的邻居，再对未评分物品进行加权预测。

### 3.2 物品协同过滤（Item-CF）

先计算物品之间的相似度，再根据用户历史评分过的物品推断未见物品的推荐分数。

### 3.3 基于内容的推荐（Content-Based）

利用电影的类型标签构建用户画像，并据此计算未评分电影与用户偏好的匹配程度。

### 3.4 混合推荐（Hybrid）

采用加权融合策略：

$$S_{u,i} = \alpha \cdot S_{CF}(u,i) + (1-\alpha) \cdot S_{CB}(u,i)$$

当前结果对应的混合权重为 `alpha = 0.6`。从 `results.json` 的权重分析看，不同权重会带来明显的召回率和多样性差异，因此这里只保留结果驱动的描述。

---

## 4. 评估指标

### 4.1 召回率（Recall@K）

$$\text{Recall@K} = \frac{|\text{推荐列表} \cap \text{用户喜欢的物品}|}{|\text{用户喜欢的物品}|}$$

### 4.2 多样性（Diversity）

衡量推荐列表中物品之间的差异程度。报告中的该指标为全体评估用户的平均值。

### 4.3 覆盖率（Coverage）

衡量方法最终能覆盖到的物品范围。

### 4.4 新颖度（Novelty）

衡量推荐结果对低流行度物品的偏好程度。

---

## 5. 实验结果

### 5.1 Recall@10 对比

| 方法 | Recall@10 |
|------|-----------|
| User-CF | 0.0094 |
| Item-CF | 0.0002 |
| Content-Based | 0.0192 |
| **Hybrid** | **0.0314** |

### 5.2 多样性、覆盖率和新颖度

| 方法 | Diversity | Coverage | Novelty |
|------|-----------|----------|---------|
| User-CF | 0.7511 | 0.5297 | 0.8594 |
| Item-CF | 0.6929 | 0.2176 | 0.9966 |
| Content-Based | 0.1360 | 0.4180 | 0.8790 |
| Hybrid | 0.3404 | 0.4073 | 0.8121 |

### 5.3 可视化结果

- `figures/comparison.png`：方法间综合指标对比
- `figures/recall_at_k.png`：不同 `K` 下的 `Recall@K` 变化
- `figures/weight_analysis.png`：混合权重与指标变化趋势
- `figures/radar_chart.png`：Recall、Diversity、Coverage、Novelty 的雷达图

---

## 6. 结果分析

### 6.1 召回率分析

1. **Hybrid 的 Recall@10 最高**，达到 `0.0314`，高于 Content-Based 的 `0.0192`、User-CF 的 `0.0094` 和 Item-CF 的 `0.0002`。这说明在当前数据划分和实现参数下，混合策略在召回表现上更占优。

2. **Content-Based 的召回率高于 User-CF 和 Item-CF**。这更像是当前特征表达和数据稀疏性下的结果，不宜直接推广为一般性结论。

3. **Item-CF 的召回率最弱**，说明在这个实验设置中，基于物品相似度的信号没有形成有效的召回能力。

### 6.2 多样性、覆盖率和新颖度分析

1. **多样性最高的是 User-CF（0.7511）**，其次是 Item-CF（0.6929）。Content-Based 的多样性最低（0.1360），说明它更容易围绕少数相似内容反复推荐。

2. **覆盖率最高的是 User-CF（0.5297）**。Hybrid 的覆盖率为 `0.4073`，低于 User-CF 和 Content-Based，但高于 Item-CF。说明混合策略并没有简单地把覆盖面拉到最大，而是在多个指标之间做了折中。

3. **新颖度最高的是 Item-CF（0.9966）**，这意味着它更倾向推荐低流行度物品。但高新颖度并不自动带来更高召回率，当前结果已经体现出这一点。

4. 这些指标均基于 **920 个评估用户的整体平均值**，不是单个用户样本，因此可以直接用于方法层面的比较。

### 6.3 混合权重趋势

`results.json` 中的权重分析显示，不同 `alpha` 会带来明显差异：

- `alpha = 0.2` 时 Recall@10 最高，达到 `0.0450`
- `alpha = 0.6` 时 Recall@10 为 `0.0314`，多样性和覆盖率处于相对更均衡的位置
- `alpha = 1.0` 时多样性最高，但召回率明显下降

因此，当前混合结果更适合被理解为一种折中配置，而不是对所有指标都最优的方案。报告没有额外实验来证明“冷启动被解决”或“推荐质量全面提升”，这类表述不应超出当前数据支持范围。

---

## 7. 实验结论

1. 在本次实验的真实结果下，**Hybrid 的 `Recall@10` 最好**，但它并不是所有指标都最优的方案。

2. **User-CF 在多样性和覆盖率上更强**，Item-CF 在新颖度上最高，Content-Based 在召回率上高于 User-CF 和 Item-CF。各方法优势不同，适合按目标选择。

3. 本实验只能支持“在当前划分与参数设置下的比较结论”，**不能直接推出混合推荐已经解决冷启动问题**，也不能说明它在所有场景下都能提升推荐质量。

---

## 8. 文件结构

```text
作业7/
├── data/
│   └── ml-100k/
├── figures/
│   ├── comparison.png
│   ├── recall_at_k.png
│   ├── weight_analysis.png
│   └── radar_chart.png
├── src/
│   └── hybrid_recommender.py
├── results.json
├── 作业要求.jpg
└── report.md
```

---

## 9. 运行说明

在项目根目录执行：

```bash
cd "/Users/lianglihang/Downloads/Principles-and-Techniques-of-Big-Data/作业7"
uv run --with numpy,pandas,scikit-learn,matplotlib python src/hybrid_recommender.py
```

运行后会生成或更新：

- `results.json`
- `figures/comparison.png`
- `figures/recall_at_k.png`
- `figures/weight_analysis.png`
- `figures/radar_chart.png`

---

## 参考文献

1. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30-37.
2. Lops, P., De Gemmis, M., & Semeraro, G. (2011). Content-based recommender systems: State of the art and trends. In *Recommender systems handbook* (pp. 73-105). Springer.
3. Su, X., & Khoshgoftaar, T. M. (2009). A survey of collaborative filtering techniques. *Advances in artificial intelligence*, 2009.
