#!/usr/bin/env python3
"""
作业7：混合推荐系统设计（完整版）
- 协同过滤（用户/物品相似度）
- 基于内容的推荐（电影标签）
- 召回率对比
- 加权混合策略
- 完整评估与可视化

作者：梁力航 23336128
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml-100k"
RESULTS_PATH = PROJECT_ROOT / "results.json"
FIGURES_DIR = PROJECT_ROOT / "figures"


def load_movielens(data_dir: Path = DATA_DIR):
    """加载 MovieLens 100K 数据。"""
    data_dir = Path(data_dir)
    ratings = pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    movies = pd.read_csv(
        data_dir / "u.item",
        sep="|",
        encoding="latin-1",
        names=["movie_id", "title", "date", "vdate", "url"] + [f"g{i}" for i in range(19)],
        usecols=range(24),
    )

    genres = []
    with open(data_dir / "u.genre", "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 2 and parts[0]:
                genres.append(parts[0])

    for i, genre in enumerate(genres):
        movies.rename(columns={f"g{i}": genre}, inplace=True)

    n_users = int(ratings["user_id"].max())
    n_items = int(ratings["item_id"].max())
    rating_matrix = np.zeros((n_users, n_items), dtype=np.float32)

    user_idx = ratings["user_id"].to_numpy(dtype=np.int64) - 1
    item_idx = ratings["item_id"].to_numpy(dtype=np.int64) - 1
    rating_matrix[user_idx, item_idx] = ratings["rating"].to_numpy(dtype=np.float32)

    movie_features = movies[genres].to_numpy(dtype=np.float32, copy=True)

    print(f"数据加载完成：{len(ratings)}评分, {n_users}用户, {n_items}电影, {len(genres)}类型")

    return ratings, movies, rating_matrix, movie_features, genres


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def build_cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """基于矩阵行向量构建余弦相似度矩阵。"""
    normalized = normalize_rows(matrix.astype(np.float32, copy=False))
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, 0.0)
    return similarity.astype(np.float32, copy=False)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """计算余弦相似度。"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _sorted_items_by_score(scores: dict[int, float]) -> tuple[tuple[int, float], ...]:
    if not scores:
        return ()
    return tuple(sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])))


def _min_max_normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = np.array(list(scores.values()), dtype=np.float32)
    min_v = float(values.min())
    max_v = float(values.max())
    if np.isclose(min_v, max_v):
        return {item_id: 1.0 for item_id in scores}
    scale = max_v - min_v
    return {item_id: float((score - min_v) / scale) for item_id, score in scores.items()}


def diversity(predictions, movie_features):
    """计算单个推荐列表的多样性。"""
    if len(predictions) < 2:
        return 0.0

    items = [i for i, _ in predictions]
    dissims = []

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            fi, fj = movie_features[items[i]], movie_features[items[j]]
            if np.linalg.norm(fi) > 0 and np.linalg.norm(fj) > 0:
                dissims.append(1 - cosine_sim(fi, fj))

    return float(np.mean(dissims)) if dissims else 0.0


def mean_diversity(recommendations, movie_features):
    """计算方法层面的平均多样性。"""
    if not recommendations:
        return 0.0
    values = [diversity(recs, movie_features) for recs in recommendations]
    return float(np.mean(values)) if values else 0.0


def coverage(recommendations, n_items):
    """计算覆盖率。"""
    if n_items <= 0:
        return 0.0
    all_items = set()
    for recs in recommendations:
        all_items.update([i for i, _ in recs])
    return len(all_items) / n_items


def novelty(recommendations, item_popularity, max_popularity):
    """计算新颖度，越接近 1 表示越偏向低流行度物品。"""
    if not recommendations or max_popularity <= 0:
        return 0.0

    novelties = []
    for recs in recommendations:
        if not recs:
            continue
        item_novelties = [1 - item_popularity.get(i, 0) / max_popularity for i, _ in recs]
        novelties.append(float(np.mean(item_novelties)))
    return float(np.mean(novelties)) if novelties else 0.0


class RecommenderSystem:
    """封装预计算相似度和缓存推荐结果。"""

    def __init__(self, train_matrix: np.ndarray, movie_features: np.ndarray, item_popularity: dict[int, int]):
        self.train_matrix = train_matrix.astype(np.float32, copy=False)
        self.movie_features = movie_features.astype(np.float32, copy=False)
        self.user_sim = build_cosine_similarity_matrix(self.train_matrix)
        self.item_sim = build_cosine_similarity_matrix(self.train_matrix.T)
        self.movie_features_norm = normalize_rows(self.movie_features)
        self.item_popularity = item_popularity
        self.max_popularity = max(item_popularity.values()) if item_popularity else 0

    @lru_cache(maxsize=None)
    def user_cf(self, user_id: int, top_k: int = 30) -> tuple[tuple[int, float], ...]:
        """用户协同过滤推荐。"""
        user_ratings = self.train_matrix[user_id]
        sims = self.user_sim[user_id]
        neighbor_ids = np.flatnonzero(sims > 0)
        if neighbor_ids.size == 0:
            return ()

        if neighbor_ids.size > top_k:
            local_scores = sims[neighbor_ids]
            top_idx = np.argpartition(local_scores, -top_k)[-top_k:]
            neighbor_ids = neighbor_ids[top_idx]

        weights = sims[neighbor_ids]
        neighbor_ratings = self.train_matrix[neighbor_ids]
        rated_mask = neighbor_ratings > 0
        numerator = weights @ neighbor_ratings
        denominator = (np.abs(weights)[:, np.newaxis] * rated_mask).sum(axis=0)

        scores = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float32),
            where=denominator > 0,
        )
        scores[user_ratings > 0] = 0.0

        ranked = np.flatnonzero(scores > 0)
        if ranked.size == 0:
            return ()

        ranked = ranked[np.argsort(scores[ranked])[::-1]]
        return tuple((int(item_id), float(scores[item_id])) for item_id in ranked)

    @lru_cache(maxsize=None)
    def item_cf(self, user_id: int, top_k: int = 30) -> tuple[tuple[int, float], ...]:
        """物品协同过滤推荐。"""
        user_ratings = self.train_matrix[user_id]
        rated_items = np.flatnonzero(user_ratings > 0)
        if rated_items.size == 0:
            return ()

        item_scores = self.item_sim[:, rated_items]
        positive_scores = np.clip(item_scores, 0.0, None)
        numerator = positive_scores @ user_ratings[rated_items]
        denominator = positive_scores.sum(axis=1)

        scores = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float32),
            where=denominator > 0,
        )
        scores[user_ratings > 0] = 0.0

        ranked = np.flatnonzero(scores > 0)
        if ranked.size == 0:
            return ()

        ranked = ranked[np.argsort(scores[ranked])[::-1]]
        return tuple((int(item_id), float(scores[item_id])) for item_id in ranked)

    @lru_cache(maxsize=None)
    def content_based(self, user_id: int) -> tuple[tuple[int, float], ...]:
        """基于内容的推荐。"""
        user_ratings = self.train_matrix[user_id]
        rated_mask = user_ratings > 0
        if not np.any(rated_mask):
            return ()

        rated_weights = user_ratings[rated_mask]
        user_profile = (self.movie_features[rated_mask] * rated_weights[:, np.newaxis]).sum(axis=0)
        profile_norm = np.linalg.norm(user_profile)
        if profile_norm == 0:
            return ()
        user_profile = user_profile / profile_norm

        scores = self.movie_features_norm @ user_profile
        scores[user_ratings > 0] = 0.0

        ranked = np.flatnonzero(scores > 0)
        if ranked.size == 0:
            return ()

        ranked = ranked[np.argsort(scores[ranked])[::-1]]
        return tuple((int(item_id), float(scores[item_id])) for item_id in ranked)

    @lru_cache(maxsize=None)
    def hybrid(self, user_id: int, weight_cf: float = 0.6) -> tuple[tuple[int, float], ...]:
        """加权混合推荐，使用真实的 CF 与内容分数融合。"""
        cf_scores = dict(self.user_cf(user_id, top_k=50))
        cb_scores = dict(self.content_based(user_id))
        if not cf_scores and not cb_scores:
            return ()

        cf_norm = _min_max_normalize(cf_scores)
        cb_norm = _min_max_normalize(cb_scores)

        all_items = sorted(set(cf_norm) | set(cb_norm))
        hybrid_scores = []
        for item_id in all_items:
            score = weight_cf * cf_norm.get(item_id, 0.0) + (1 - weight_cf) * cb_norm.get(item_id, 0.0)
            if score > 0:
                hybrid_scores.append((item_id, score))

        hybrid_scores.sort(key=lambda kv: (-kv[1], kv[0]))
        return tuple((int(item_id), float(score)) for item_id, score in hybrid_scores)


def build_user_liked(test_data: pd.DataFrame, threshold: float = 4.0):
    """按测试集构建用户喜欢物品集合。"""
    user_liked = defaultdict(set)
    user_ids = test_data["user_id"].to_numpy(dtype=np.int64) - 1
    item_ids = test_data["item_id"].to_numpy(dtype=np.int64) - 1
    ratings = test_data["rating"].to_numpy(dtype=np.float32)

    for uid, iid, rating in zip(user_ids, item_ids, ratings, strict=False):
        if rating >= threshold:
            user_liked[int(uid)].add(int(iid))
    return user_liked


def evaluate_method(
    test_data: pd.DataFrame,
    recommend_func: Callable[[int], tuple[tuple[int, float], ...]],
    movie_features: np.ndarray,
    item_popularity: dict[int, int],
    n_items: int,
    n_rec: int = 10,
):
    """计算方法层面的 Recall / Diversity / Coverage / Novelty。"""
    user_liked = build_user_liked(test_data)
    users = sorted(user_liked.keys())

    recalls = []
    recommendation_lists = []
    for user_id in users:
        liked = user_liked[user_id]
        recs = list(recommend_func(user_id))[:n_rec]
        recommendation_lists.append(recs)
        if liked:
            hits = len(liked & {item_id for item_id, _ in recs})
            recalls.append(hits / len(liked))

    if not recalls:
        return {
            "recall": 0.0,
            "diversity": 0.0,
            "coverage": 0.0,
            "novelty": 0.0,
            "recommendations": [],
            "users": users,
        }

    return {
        "recall": float(np.mean(recalls)),
        "diversity": mean_diversity(recommendation_lists, movie_features),
        "coverage": coverage(recommendation_lists, n_items),
        "novelty": novelty(recommendation_lists, item_popularity, max(item_popularity.values()) if item_popularity else 0),
        "recommendations": recommendation_lists,
        "users": users,
    }


def evaluate_at_k(
    test_data: pd.DataFrame,
    recommend_funcs: dict[str, Callable[[int], tuple[tuple[int, float], ...]]],
    k_values=(5, 10, 20, 50),
):
    """评估不同 K 值的召回率。"""
    user_liked = build_user_liked(test_data)
    users = sorted(user_liked.keys())

    rankings = {method: {} for method in recommend_funcs}
    for method, func in recommend_funcs.items():
        for user_id in users:
            rankings[method][user_id] = func(user_id)

    results = {method: {} for method in recommend_funcs}
    for k in k_values:
        print(f"  评估 K={k}...")
        for method in recommend_funcs:
            recalls = []
            for user_id in users:
                liked = user_liked[user_id]
                recs = rankings[method][user_id][:k]
                hits = len(liked & {item_id for item_id, _ in recs})
                recalls.append(hits / len(liked) if liked else 0.0)
            results[method][k] = float(np.mean(recalls)) if recalls else 0.0

    return results


def evaluate_weights(
    test_data: pd.DataFrame,
    recommender: RecommenderSystem,
    weights=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
):
    """评估不同混合权重的效果。"""
    user_liked = build_user_liked(test_data)
    users = sorted(user_liked.keys())

    results = {}
    for weight in weights:
        print(f"  评估权重 CF={weight:.1f}, CB={1-weight:.1f}...")
        recalls = []
        diversities = []
        coverage_lists = []

        for user_id in users:
            liked = user_liked[user_id]
            recs = list(recommender.hybrid(user_id, weight_cf=weight))[:10]
            coverage_lists.append(recs)
            hits = len(liked & {item_id for item_id, _ in recs})
            recalls.append(hits / len(liked) if liked else 0.0)
            diversities.append(diversity(recs, recommender.movie_features))

        results[weight] = {
            "recall": float(np.mean(recalls)) if recalls else 0.0,
            "diversity": float(np.mean(diversities)) if diversities else 0.0,
            "coverage": coverage(coverage_lists, recommender.movie_features.shape[0]),
            "novelty": novelty(coverage_lists, recommender.item_popularity, recommender.max_popularity),
        }

    return results


def show_recommendations(movies: pd.DataFrame, recs, name: str):
    """打印示例推荐。"""
    print(f"\n  {name} Top-5:")
    for i, (item_id, score) in enumerate(recs[:5], 1):
        title = movies[movies["movie_id"] == item_id + 1]["title"].values
        movie_title = title[0][:35] if len(title) > 0 else f"Movie {item_id}"
        print(f"    {i}. {movie_title:<37} (分数: {score:.3f})")


def main():
    print("=" * 60)
    print("作业7：混合推荐系统设计（完整版）")
    print("作者：梁力航 23336128")
    print("=" * 60)

    # 1. 加载数据
    print("\n【步骤1】加载数据")
    ratings, movies, rating_matrix, movie_features, genres = load_movielens()
    n_users, n_items = rating_matrix.shape

    # 2. 划分数据
    print("\n【步骤2】划分训练/测试集")
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

    train_matrix = np.zeros((n_users, n_items), dtype=np.float32)
    train_user_idx = train_data["user_id"].to_numpy(dtype=np.int64) - 1
    train_item_idx = train_data["item_id"].to_numpy(dtype=np.int64) - 1
    train_matrix[train_user_idx, train_item_idx] = train_data["rating"].to_numpy(dtype=np.float32)

    print(f"  训练集: {len(train_data)}, 测试集: {len(test_data)}")

    # 3. 计算物品流行度（用于新颖度）
    item_popularity = defaultdict(int)
    for item_id in train_data["item_id"].to_numpy(dtype=np.int64):
        item_popularity[int(item_id) - 1] += 1

    recommender = RecommenderSystem(train_matrix, movie_features, dict(item_popularity))

    # 4. 主要评估（Recall@10）
    print("\n【步骤3】主要评估 (Recall@10)")
    recommend_funcs = {
        "user_cf": lambda uid: recommender.user_cf(uid),
        "item_cf": lambda uid: recommender.item_cf(uid),
        "content": lambda uid: recommender.content_based(uid),
        "hybrid": lambda uid: recommender.hybrid(uid, weight_cf=0.6),
    }

    method_results = {}
    for method_name, func in recommend_funcs.items():
        print(f"  {method_name.replace('_', '-').title()}...")
        method_results[method_name] = evaluate_method(
            test_data,
            func,
            movie_features,
            dict(item_popularity),
            n_items,
            n_rec=10,
        )

    user_cf_recall = method_results["user_cf"]["recall"]
    item_cf_recall = method_results["item_cf"]["recall"]
    cb_recall = method_results["content"]["recall"]
    hybrid_recall = method_results["hybrid"]["recall"]

    print(f"\n  User-CF 召回率: {user_cf_recall:.4f}")
    print(f"  Item-CF 召回率: {item_cf_recall:.4f}")
    print(f"  Content-Based 召回率: {cb_recall:.4f}")
    print(f"  Hybrid 召回率: {hybrid_recall:.4f}")

    # 5. 多样性、覆盖率、新颖度分析
    print("\n【步骤4】多样性、覆盖率、新颖度分析")
    user_cf_div = method_results["user_cf"]["diversity"]
    item_cf_div = method_results["item_cf"]["diversity"]
    cb_div = method_results["content"]["diversity"]
    hybrid_div = method_results["hybrid"]["diversity"]

    user_cf_cov = method_results["user_cf"]["coverage"]
    item_cf_cov = method_results["item_cf"]["coverage"]
    cb_cov = method_results["content"]["coverage"]
    hybrid_cov = method_results["hybrid"]["coverage"]

    user_cf_nov = method_results["user_cf"]["novelty"]
    item_cf_nov = method_results["item_cf"]["novelty"]
    cb_nov = method_results["content"]["novelty"]
    hybrid_nov = method_results["hybrid"]["novelty"]

    print(f"  User-CF 多样性: {user_cf_div:.4f}，覆盖率: {user_cf_cov:.4f}，新颖度: {user_cf_nov:.4f}")
    print(f"  Item-CF 多样性: {item_cf_div:.4f}，覆盖率: {item_cf_cov:.4f}，新颖度: {item_cf_nov:.4f}")
    print(f"  Content-Based 多样性: {cb_div:.4f}，覆盖率: {cb_cov:.4f}，新颖度: {cb_nov:.4f}")
    print(f"  Hybrid 多样性: {hybrid_div:.4f}，覆盖率: {hybrid_cov:.4f}，新颖度: {hybrid_nov:.4f}")

    # 6. 不同K值的评估
    print("\n【步骤5】不同K值的召回率评估")
    k_results = evaluate_at_k(
        test_data,
        recommend_funcs,
        k_values=(5, 10, 20, 50),
    )

    for method in ["user_cf", "item_cf", "content", "hybrid"]:
        print(f"  {method}:")
        for k, recall in k_results[method].items():
            print(f"    K={k}: {recall:.4f}")

    # 7. 不同权重的评估
    print("\n【步骤6】不同混合权重的评估")
    weight_results = evaluate_weights(test_data, recommender)

    for w, metrics in weight_results.items():
        print(
            f"  CF权重={w:.1f}: 召回率={metrics['recall']:.4f}, "
            f"多样性={metrics['diversity']:.4f}, 覆盖率={metrics['coverage']:.4f}, 新颖度={metrics['novelty']:.4f}"
        )

    # 8. 示例推荐
    print("\n【步骤7】示例推荐 (用户1)")
    sample_user = 0
    show_recommendations(movies, recommender.user_cf(sample_user), "User-CF")
    show_recommendations(movies, recommender.item_cf(sample_user), "Item-CF")
    show_recommendations(movies, recommender.content_based(sample_user), "Content-Based")
    show_recommendations(movies, recommender.hybrid(sample_user, weight_cf=0.6), "Hybrid")

    # 9. 保存结果
    print("\n【步骤8】保存结果")
    results = {
        "recall": {
            "user_cf": float(round(user_cf_recall, 4)),
            "item_cf": float(round(item_cf_recall, 4)),
            "content_based": float(round(cb_recall, 4)),
            "hybrid": float(round(hybrid_recall, 4)),
        },
        "diversity": {
            "user_cf": float(round(user_cf_div, 4)),
            "item_cf": float(round(item_cf_div, 4)),
            "content_based": float(round(cb_div, 4)),
            "hybrid": float(round(hybrid_div, 4)),
        },
        "coverage": {
            "user_cf": float(round(user_cf_cov, 4)),
            "item_cf": float(round(item_cf_cov, 4)),
            "content_based": float(round(cb_cov, 4)),
            "hybrid": float(round(hybrid_cov, 4)),
        },
        "novelty": {
            "user_cf": float(round(user_cf_nov, 4)),
            "item_cf": float(round(item_cf_nov, 4)),
            "content_based": float(round(cb_nov, 4)),
            "hybrid": float(round(hybrid_nov, 4)),
        },
        "recall_at_k": {
            k: {m: float(v) for m, v in vals.items()}
            for k, vals in k_results.items()
        },
        "weight_analysis": {
            str(w): {k: float(v) for k, v in metrics.items()}
            for w, metrics in weight_results.items()
        },
        "meta": {
            "n_users": int(n_users),
            "n_items": int(n_items),
            "n_train": int(len(train_data)),
            "n_test": int(len(test_data)),
            "n_eval_users": int(len(method_results["user_cf"]["users"])),
            "n_recommendations": 10,
        },
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  已保存: {RESULTS_PATH}")

    # 10. 生成可视化
    print("\n【步骤9】生成可视化")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from math import pi

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        # 图1: 召回率与多样性对比
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        methods = ["User-CF", "Item-CF", "Content", "Hybrid"]
        recalls = [user_cf_recall, item_cf_recall, cb_recall, hybrid_recall]
        diversities = [user_cf_div, item_cf_div, cb_div, hybrid_div]

        colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

        axes[0].bar(methods, recalls, color=colors)
        axes[0].set_ylabel("Recall@10")
        axes[0].set_title("Recall Comparison")
        axes[0].set_ylim(0, max(recalls) * 1.5 if max(recalls) > 0 else 1)
        for i, v in enumerate(recalls):
            axes[0].text(i, v + 0.001, f"{v:.4f}", ha="center")

        axes[1].bar(methods, diversities, color=colors)
        axes[1].set_ylabel("Diversity")
        axes[1].set_title("Diversity Comparison")
        axes[1].set_ylim(0, max(diversities) * 1.3 if max(diversities) > 0 else 1)
        for i, v in enumerate(diversities):
            axes[1].text(i, v + 0.02, f"{v:.4f}", ha="center")

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "comparison.png", dpi=150)
        print(f"  已保存: {FIGURES_DIR / 'comparison.png'}")
        plt.close()

        # 图2: 不同K值的召回率曲线
        fig, ax = plt.subplots(figsize=(10, 6))

        k_values = [5, 10, 20, 50]
        for method, color, label in [
            ("user_cf", "#3498db", "User-CF"),
            ("item_cf", "#e74c3c", "Item-CF"),
            ("content", "#2ecc71", "Content-Based"),
            ("hybrid", "#9b59b6", "Hybrid"),
        ]:
            method_recalls = [k_results[method][k] for k in k_values]
            ax.plot(k_values, method_recalls, marker="o", linewidth=2, color=color, label=label)

        ax.set_xlabel("K (Number of Recommendations)")
        ax.set_ylabel("Recall@K")
        ax.set_title("Recall@K for Different K Values")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "recall_at_k.png", dpi=150)
        print(f"  已保存: {FIGURES_DIR / 'recall_at_k.png'}")
        plt.close()

        # 图3: 混合权重分析
        fig, ax1 = plt.subplots(figsize=(10, 6))

        weights = sorted(weight_results.keys())
        weight_recalls = [weight_results[w]["recall"] for w in weights]
        weight_diversities = [weight_results[w]["diversity"] for w in weights]

        ax1.set_xlabel("CF Weight")
        ax1.set_ylabel("Recall@10", color="#3498db")
        ax1.plot(weights, weight_recalls, "o-", color="#3498db", linewidth=2, label="Recall")
        ax1.tick_params(axis="y", labelcolor="#3498db")
        ax1.set_ylim(0, max(weight_recalls) * 1.3 if max(weight_recalls) > 0 else 1)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Diversity", color="#e74c3c")
        ax2.plot(weights, weight_diversities, "s-", color="#e74c3c", linewidth=2, label="Diversity")
        ax2.tick_params(axis="y", labelcolor="#e74c3c")
        ax2.set_ylim(0, max(weight_diversities) * 1.3 if max(weight_diversities) > 0 else 1)

        plt.title("Hybrid Recommender: Weight Analysis")
        fig.tight_layout()
        plt.savefig(FIGURES_DIR / "weight_analysis.png", dpi=150)
        print(f"  已保存: {FIGURES_DIR / 'weight_analysis.png'}")
        plt.close()

        # 图4: 雷达图对比
        categories = ["Recall", "Diversity", "Novelty", "Coverage"]
        metric_map = {
            "User-CF": {
                "Recall": user_cf_recall,
                "Diversity": user_cf_div,
                "Novelty": user_cf_nov,
                "Coverage": user_cf_cov,
            },
            "Item-CF": {
                "Recall": item_cf_recall,
                "Diversity": item_cf_div,
                "Novelty": item_cf_nov,
                "Coverage": item_cf_cov,
            },
            "Content-Based": {
                "Recall": cb_recall,
                "Diversity": cb_div,
                "Novelty": cb_nov,
                "Coverage": cb_cov,
            },
            "Hybrid": {
                "Recall": hybrid_recall,
                "Diversity": hybrid_div,
                "Novelty": hybrid_nov,
                "Coverage": hybrid_cov,
            },
        }
        metric_max = {
            metric: max(values[metric] for values in metric_map.values()) or 1.0
            for metric in categories
        }

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]

        for label, color in [
            ("User-CF", "#3498db"),
            ("Item-CF", "#e74c3c"),
            ("Content-Based", "#2ecc71"),
            ("Hybrid", "#9b59b6"),
        ]:
            values = [metric_map[label][metric] / metric_max[metric] if metric_max[metric] > 0 else 0.0 for metric in categories]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, color=color, label=label)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax.set_title("Multi-dimensional Comparison", y=1.08)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "radar_chart.png", dpi=150)
        print(f"  已保存: {FIGURES_DIR / 'radar_chart.png'}")
        plt.close()

    except Exception as e:
        print(f"  图表生成失败: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("【实验完成】")
    print("=" * 60)
    print("\n结果摘要:")
    print(f"  User-CF:    召回率 {user_cf_recall:.4f}, 多样性 {user_cf_div:.4f}, 覆盖率 {user_cf_cov:.4f}, 新颖度 {user_cf_nov:.4f}")
    print(f"  Item-CF:    召回率 {item_cf_recall:.4f}, 多样性 {item_cf_div:.4f}, 覆盖率 {item_cf_cov:.4f}, 新颖度 {item_cf_nov:.4f}")
    print(f"  Content:    召回率 {cb_recall:.4f}, 多样性 {cb_div:.4f}, 覆盖率 {cb_cov:.4f}, 新颖度 {cb_nov:.4f}")
    print(f"  Hybrid:     召回率 {hybrid_recall:.4f}, 多样性 {hybrid_div:.4f}, 覆盖率 {hybrid_cov:.4f}, 新颖度 {hybrid_nov:.4f}")

    return results


if __name__ == "__main__":
    main()
