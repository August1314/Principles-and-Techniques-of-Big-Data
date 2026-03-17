from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import spacy
from spacy.matcher import PhraseMatcher
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
IN_JSONL = ROOT / "data" / "qa.jsonl"
NODES_CSV = ROOT / "outputs" / "nodes.csv"
EDGES_CSV = ROOT / "outputs" / "edges.csv"
STATS_JSON = ROOT / "outputs" / "stats.json"


BASE_DISEASE_LEXICON = [
    "高血压",
    "糖尿病",
    "冠心病",
    "脑梗",
    "脑梗塞",
    "高血脂",
    "胃炎",
    "胃溃疡",
    "脂肪肝",
    "哮喘",
    "肺炎",
    "感冒",
    "咽炎",
]

SYMPTOM_SUFFIXES = ("痛", "疼", "发热", "发烧", "咳嗽", "乏力", "头晕", "恶心", "呕吐", "腹泻", "失眠")

TEST_KEYWORDS = ("血常规", "尿常规", "B超", "彩超", "CT", "核磁", "心电图", "胃镜", "肠镜", "血糖", "血压", "血脂")

DRUG_VERBS = ("吃", "服用", "口服", "用", "使用", "配合", "加用")


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    name: str


@dataclass(frozen=True)
class Edge:
    source: str
    rel: str
    target: str
    evidence: str


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_id(prefix: str, name: str) -> str:
    name = re.sub(r"\s+", "", name)
    return f"{prefix}:{name}"


def build_phrase_matcher(nlp) -> PhraseMatcher:
    matcher = PhraseMatcher(nlp.vocab, attr="TEXT")
    patterns = [nlp.make_doc(x) for x in BASE_DISEASE_LEXICON]
    matcher.add("DISEASE", patterns)
    return matcher


def harvest_disease_lexicon(rows: list[dict]) -> list[str]:
    # 从标题里自动挖掘“XX病/XX炎/XX症/XX癌/XX结石”等候选疾病词，补充词表
    cands: set[str] = set(BASE_DISEASE_LEXICON)
    pat = re.compile(r"([一-龥]{2,10}?(?:病|炎|症|癌|瘤|结石))")
    for r in rows:
        title = normalize_text(str(r.get("title", "")))
        for m in pat.finditer(title):
            w = m.group(1)
            if 2 <= len(w) <= 10:
                cands.add(w)
    return sorted(cands, key=len, reverse=True)


def extract_disease(nlp, matcher: PhraseMatcher, title: str, ask: str) -> str | None:
    doc = nlp(normalize_text(title + " " + ask))
    matches = matcher(doc)
    if not matches:
        return None
    # 取最先出现的疾病词
    matches = sorted(matches, key=lambda x: x[1])
    _, start, end = matches[0]
    return doc[start:end].text


def split_items(text: str) -> list[str]:
    text = re.sub(r"[；;。.!！？?]", "，", text)
    parts = re.split(r"[，,、/ ]+", text)
    return [p.strip() for p in parts if p.strip()]


def extract_symptoms(text: str) -> list[str]:
    # 规则：抓取“症状/表现/不适”为锚点后的一小段
    out: list[str] = []
    m = re.search(r"(症状|表现|不适)[:：]?(.*)", text)
    if m:
        tail = m.group(2)[:80]
        cands = split_items(tail)
        for c in cands:
            if any(c.endswith(suf) for suf in SYMPTOM_SUFFIXES) and 1 < len(c) <= 10:
                out.append(c)

    # 兜底：直接在全文里抓取常见症状后缀（问答语料常写“头痛/胸痛/咳嗽/发热…”）
    suffix_alt = "|".join(re.escape(s) for s in SYMPTOM_SUFFIXES)
    pat = re.compile(rf"([一-龥]{{1,6}}(?:{suffix_alt}))")
    for m2 in pat.finditer(text):
        w = m2.group(1)
        if 2 <= len(w) <= 10:
            out.append(w)

    return list(dict.fromkeys(out))


def extract_tests(answer: str) -> list[str]:
    out: list[str] = []
    for kw in TEST_KEYWORDS:
        if kw in answer:
            out.append(kw)
    return out


def extract_drugs(answer: str) -> list[str]:
    # 只做轻量抽取：动词后面 2~6 个字的“药名/药物”
    out: list[str] = []
    for v in DRUG_VERBS:
        for m in re.finditer(re.escape(v) + r"([一-龥]{2,6}(片|胶囊|颗粒|口服液|针|喷雾|药|滴眼液)?)", answer):
            name = m.group(1)
            if len(name) >= 2:
                out.append(name)
    return list(dict.fromkeys(out))


def main() -> None:
    if not IN_JSONL.exists():
        raise FileNotFoundError(f"请先运行 prepare_dataset：{IN_JSONL} 不存在")

    ROOT.joinpath("outputs").mkdir(parents=True, exist_ok=True)

    nlp = spacy.load("zh_core_web_sm")
    rows = list(iter_jsonl(IN_JSONL))
    # 自动扩充疾病词表，再构建 PhraseMatcher
    disease_lexicon = harvest_disease_lexicon(rows)
    matcher = PhraseMatcher(nlp.vocab, attr="TEXT")
    matcher.add("DISEASE", [nlp.make_doc(x) for x in disease_lexicon])

    nodes: dict[str, Node] = {}
    edges: list[Edge] = []
    stats = Counter()
    rel_stats = Counter()
    skipped_no_disease = 0

    def upsert(node: Node) -> None:
        nodes[node.id] = node

    for row in tqdm(rows, desc="extract"):
        department = normalize_text(row.get("department", ""))
        title = normalize_text(row.get("title", ""))
        ask = normalize_text(row.get("ask", ""))
        answer = normalize_text(row.get("answer", ""))

        disease = extract_disease(nlp, matcher, title, ask)
        if not disease:
            skipped_no_disease += 1
            continue

        disease_id = make_id("Disease", disease)
        upsert(Node(id=disease_id, label="Disease", name=disease))

        if department:
            dep_id = make_id("Department", department)
            upsert(Node(id=dep_id, label="Department", name=department))
            edges.append(Edge(source=disease_id, rel="BELONGS_TO", target=dep_id, evidence=title[:40]))
            rel_stats["BELONGS_TO"] += 1

        for s in extract_symptoms(ask + " " + answer):
            sid = make_id("Symptom", s)
            upsert(Node(id=sid, label="Symptom", name=s))
            edges.append(Edge(source=disease_id, rel="HAS_SYMPTOM", target=sid, evidence=s))
            rel_stats["HAS_SYMPTOM"] += 1

        for t in extract_tests(answer):
            tid = make_id("Test", t)
            upsert(Node(id=tid, label="Test", name=t))
            edges.append(Edge(source=disease_id, rel="NEED_TEST", target=tid, evidence=t))
            rel_stats["NEED_TEST"] += 1

        for d in extract_drugs(answer):
            did = make_id("Drug", d)
            upsert(Node(id=did, label="Drug", name=d))
            edges.append(Edge(source=disease_id, rel="RECOMMEND_DRUG", target=did, evidence=d))
            rel_stats["RECOMMEND_DRUG"] += 1

        stats["rows_used"] += 1

    with NODES_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "label", "name"])
        w.writeheader()
        for n in nodes.values():
            w.writerow({"id": n.id, "label": n.label, "name": n.name})

    with EDGES_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "rel", "target", "evidence"])
        w.writeheader()
        for e in edges:
            w.writerow({"source": e.source, "rel": e.rel, "target": e.target, "evidence": e.evidence})

    summary = {
        "rows_total": len(rows),
        "rows_used": stats["rows_used"],
        "skipped_no_disease": skipped_no_disease,
        "nodes": len(nodes),
        "edges": len(edges),
        "rel_stats": dict(rel_stats),
        "node_label_stats": dict(Counter(n.label for n in nodes.values())),
        "note": "实体抽取：spaCy 中文分词 + 领域词表（疾病）+ 规则；关系抽取：正则规则（症状/检查/用药/科室）。",
    }
    STATS_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
