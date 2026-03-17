# 作业2：构建小型领域知识图谱（医疗问答）

本作业基于公开的中文医疗问答数据集，使用 **spaCy** 做分词/实体抽取，并用规则抽取部分关系，最终写入 **Neo4j** 进行可视化。

## 目录结构

- `raw/`：原始数据（CSV，GBK/GB18030 编码）
- `data/`：清洗后的数据（JSONL）
- `src/`：数据处理、抽取、写库脚本
- `outputs/`：导出给 Neo4j 的节点/边 CSV 与统计结果

## 环境（Anaconda `ml`）

在 `ml` 环境中安装依赖：

```bash
conda run -n ml python -m pip install -r requirements.txt
conda run -n ml python -m spacy download zh_core_web_sm
```

## 运行流程

1) 清洗数据（从 `raw/cmd_sample.csv` 生成 `data/qa.jsonl`）：

```bash
conda run -n ml python src/prepare_dataset.py
```

2) 抽取实体与关系，生成 Neo4j 导入文件：

```bash
conda run -n ml python src/extract_kg.py
```

生成：
- `outputs/nodes.csv`
- `outputs/edges.csv`
- `outputs/stats.json`

3) 启动 Neo4j（2 选 1）

方式 A：Docker（推荐）

```bash
docker run --rm \
  --name neo4j-assignment2 \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

方式 B：Neo4j Desktop
- 新建 DBMS，设置 Bolt 端口 `7687`，并记住用户名/密码

4) 写入 Neo4j

```bash
conda run -n ml python src/load_neo4j.py --uri bolt://localhost:7687 --user neo4j --password password
```

5) 在 Neo4j Browser 可视化（`http://localhost:7474`）

示例查询（复制到 Browser 执行）：

```cypher
MATCH (d:Disease)-[r]->(e) RETURN d,r,e LIMIT 50;
```

```cypher
MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
RETURN d.name AS disease, count(*) AS n
ORDER BY n DESC LIMIT 10;
```

## 数据来源说明

本项目默认使用 `Toyhom/Chinese-medical-dialogue-data` 的样例 CSV（内科问答片段）。见报告中的“数据来源”部分写法（附仓库链接）。

