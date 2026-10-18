"""
End-to-end reproducible benchmark runner for Rankwise.
Evaluates:
- Baseline LightGBM LambdaMART NDCG@10 on held-out unbiased random traffic [A]
- Two-Tower + DCN-v2 NDCG@10 on held-out unbiased random traffic [B] (with 95% bootstrap CI)
- Counterfactual SNIPS policy-value relative estimation error [E]% vs unbiased ground truth
- Expected Calibration Error (ECE) with isotonic regression [C]
- Serving benchmark: p50, p95, p99 latency [D] ms and throughput [Q] RPS on [HARDWARE] (1,000 requests)

All metrics are saved strictly to benchmarks/results/rankwise_metrics.json.
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import os
import time
import json
import random
import platform
from pathlib import Path

# Add src to sys.path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from fastapi.testclient import TestClient
import onnxruntime as ort

from rankwise.data.dataset import load_kuairand_pure, compute_item_popularity
from rankwise.retrieval.two_tower import UserTower, ItemTower, SampledSoftmaxWithLogQ, export_user_tower_onnx
from rankwise.retrieval.indexer import HNSWIndexer
from rankwise.ranking.dcn_v2 import DCNv2MultiTask
from rankwise.ranking.baseline import LambdaMARTBaseline
from rankwise.ranking.calibration import RankerCalibrator
from rankwise.eval.metrics import compute_group_ndcg, compute_ndcg_at_k, compute_ece, bootstrap_ci
from rankwise.eval.counterfactual import ExposurePropensityModel, evaluate_counterfactual_policy
from rankwise.serving.app import app, configure_app


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_hardware_info() -> str:
    processor = platform.processor() or platform.machine() or "x86_64"
    if "Intel" in processor or "AMD" in processor:
        cpu_name = processor
    else:
        cpu_name = f"{platform.machine()} ({os.cpu_count()} vCPUs)"
    
    device_name = "CUDA GPU" if torch.cuda.is_available() else f"CPU ({cpu_name})"
    return device_name


def run_benchmarks(seed: int = 42):
    set_seed(seed)
    print(f"=== Rankwise End-to-End Reproducible Benchmark (seed={seed}) ===")
    
    data_dir = root_dir / "data_raw" / "extracted" / "KuaiRand-Pure" / "data"
    results_dir = root_dir / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Dataset Acquisition & Temporal Slices
    # ---------------------------------------------------------
    train_csv = data_dir / "log_standard_4_08_to_4_21_pure.csv"
    val_csv = data_dir / "log_standard_4_22_to_5_08_pure.csv"
    rand_csv = data_dir / "log_random_4_22_to_5_08_pure.csv"
    
    print("[*] Loading KuaiRand temporal splits...")
    df_train_raw = pd.read_csv(train_csv, nrows=100000)
    df_val_raw = pd.read_csv(val_csv, nrows=25000)
    df_rand_raw = pd.read_csv(rand_csv, nrows=35000)
    
    # Filter common users and items
    train_users = set(df_train_raw["user_id"].unique())
    train_items = set(df_train_raw["video_id"].unique())
    
    user_to_id = {u: i + 1 for i, u in enumerate(sorted(train_users))}
    item_to_id = {v: i + 1 for i, v in enumerate(sorted(train_items))}
    
    def process_df(df):
        d = df[df["user_id"].isin(user_to_id) & df["video_id"].isin(item_to_id)].copy()
        d["u_idx"] = d["user_id"].map(user_to_id)
        d["i_idx"] = d["video_id"].map(item_to_id)
        return d
        
    df_train = process_df(df_train_raw)
    df_val = process_df(df_val_raw)
    df_rand = process_df(df_rand_raw)
    
    n_users = len(user_to_id)
    n_items = len(item_to_id)
    print(f"[✓] Processed: {len(df_train):,} train, {len(df_val):,} val, {len(df_rand):,} random traffic interactions across {n_users:,} users, {n_items:,} items.")
    
    # Compute empirical item popularity Q(i)
    pop_dict = compute_item_popularity(df_train, item_col="video_id")
    item_log_q_array = np.zeros(n_items + 1, dtype=np.float32)
    for orig_item, prob in pop_dict.items():
        if orig_item in item_to_id:
            item_log_q_array[item_to_id[orig_item]] = np.log(prob)
    
    # ---------------------------------------------------------
    # 2. Stage 1: Two-Tower Retriever Training (Sampled Softmax + LogQ)
    # ---------------------------------------------------------
    print("[*] Training Stage 1 Two-Tower Retriever with logQ debiasing...")
    embed_dim = 64
    user_tower = UserTower(num_users=n_users, embed_dim=embed_dim)
    item_tower = ItemTower(num_items=n_items, embed_dim=embed_dim)
    loss_fn = SampledSoftmaxWithLogQ(temperature=0.07)
    
    optimizer = torch.optim.Adam(
        list(user_tower.parameters()) + list(item_tower.parameters()),
        lr=0.005,
        weight_decay=1e-5,
    )
    
    df_clicks = df_train[df_train["is_click"] == 1]
    u_train_tensor = torch.tensor(df_clicks["u_idx"].to_numpy(), dtype=torch.long)
    i_train_tensor = torch.tensor(df_clicks["i_idx"].to_numpy(), dtype=torch.long)
    
    dataset = TensorDataset(u_train_tensor, i_train_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
    
    user_tower.train()
    item_tower.train()
    for epoch in range(3):
        total_loss = 0.0
        for batch_u, batch_i in loader:
            optimizer.zero_grad()
            u_emb = user_tower(batch_u)
            i_emb = item_tower(batch_i)
            batch_log_q = torch.tensor(item_log_q_array[batch_i.numpy()], dtype=torch.float32)
            loss = loss_fn(u_emb, i_emb, batch_log_q)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Two-Tower Epoch {epoch + 1}/3 - Loss: {total_loss / len(loader):.4f}")
        
    user_tower.eval()
    item_tower.eval()
    
    # Build FAISS HNSW candidate index using dense 1..n_items identifiers
    print("[*] Building FAISS HNSW candidate index...")
    all_item_indices = np.arange(1, n_items + 1)
    with torch.no_grad():
        all_item_embs = item_tower(torch.tensor(all_item_indices, dtype=torch.long)).numpy()
    
    hnsw = HNSWIndexer(dim=embed_dim, M=32, ef_construction=64, ef_search=32)
    hnsw.build(all_item_embs, all_item_indices)
    
    # Export User Tower to ONNX
    models_dir = root_dir / "benchmarks" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = models_dir / "user_tower.onnx"
    export_user_tower_onnx(user_tower, onnx_path)
    print(f"[✓] User tower exported to ONNX: {onnx_path.name}")
    
    # ---------------------------------------------------------
    # 3. Stage 2: Feature Engineering & Baseline LightGBM LambdaMART [A]
    # ---------------------------------------------------------
    print("[*] Preparing ranking features and LightGBM LambdaMART baseline...")
    item_pop_map = df_train["video_id"].value_counts().to_dict()
    df_train["item_pop"] = df_train["video_id"].map(item_pop_map).fillna(1).astype(float)
    df_val["item_pop"] = df_val["video_id"].map(item_pop_map).fillna(1).astype(float)
    df_rand["item_pop"] = df_rand["video_id"].map(item_pop_map).fillna(1).astype(float)
    
    df_train["log_pop"] = (np.log1p(df_train["item_pop"]) / 10.0).astype(float)
    df_val["log_pop"] = (np.log1p(df_val["item_pop"]) / 10.0).astype(float)
    df_rand["log_pop"] = (np.log1p(df_rand["item_pop"]) / 10.0).astype(float)
    
    with torch.no_grad():
        def get_tower_scores(df_subset):
            u_t = torch.tensor(df_subset["u_idx"].to_numpy(), dtype=torch.long)
            i_t = torch.tensor(df_subset["i_idx"].to_numpy(), dtype=torch.long)
            u_vecs = user_tower(u_t)
            i_vecs = item_tower(i_t)
            return (u_vecs * i_vecs).sum(dim=-1).numpy()
            
        df_train["retrieval_score"] = get_tower_scores(df_train)
        df_val["retrieval_score"] = get_tower_scores(df_val)
        df_rand["retrieval_score"] = get_tower_scores(df_rand)
        
    feat_cols = ["log_pop", "retrieval_score"]
    
    lgbm_baseline = LambdaMARTBaseline(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=seed)
    lgbm_baseline.fit(df_train, feature_cols=feat_cols, user_col="user_id", label_col="is_click")
    
    # Evaluate LightGBM on held-out unbiased random traffic
    df_rand["lgbm_score"] = lgbm_baseline.predict(df_rand)
    metric_a_ndcg = compute_group_ndcg(df_rand, user_col="user_id", rel_col="is_click", score_col="lgbm_score", k=10)
    metric_a_formatted = float(round(metric_a_ndcg, 3))
    print(f"[A] LightGBM LambdaMART NDCG@10 (Unbiased Random Traffic): {metric_a_formatted:.3f}")
    
    # ---------------------------------------------------------
    # 4. Stage 2: Multi-Task DCN-v2 Ranker [B]
    # ---------------------------------------------------------
    print("[*] Training Multi-Task DCN-v2 Ranker (Click + Long-View)...")
    dcn_model = DCNv2MultiTask(
        num_users=n_users,
        num_items=n_items,
        embed_dim=32,
        num_dense=2,
        cross_layers=3,
        deep_hidden_dims=(128, 64),
    )
    dcn_optimizer = torch.optim.AdamW(dcn_model.parameters(), lr=0.003, weight_decay=1e-4)
    
    u_dcn = torch.tensor(df_train["u_idx"].to_numpy(), dtype=torch.long)
    i_dcn = torch.tensor(df_train["i_idx"].to_numpy(), dtype=torch.long)
    dense_dcn = torch.tensor(df_train[feat_cols].to_numpy(), dtype=torch.float32)
    y_click_dcn = torch.tensor(df_train["is_click"].to_numpy(), dtype=torch.float32)
    y_long_dcn = torch.tensor(df_train["long_view"].to_numpy(), dtype=torch.float32)
    
    dcn_ds = TensorDataset(u_dcn, i_dcn, dense_dcn, y_click_dcn, y_long_dcn)
    dcn_loader = DataLoader(dcn_ds, batch_size=256, shuffle=True)
    
    dcn_model.train()
    bce = nn.BCELoss()
    for ep in range(3):
        ep_loss = 0.0
        for bu, bi, bd, by_c, by_l in dcn_loader:
            dcn_optimizer.zero_grad()
            out = dcn_model(bu, bi, bd)
            loss_c = bce(out["p_click"], by_c)
            loss_l = bce(out["p_long_view"], by_l)
            total_dcn_loss = 0.7 * loss_c + 0.3 * loss_l
            total_dcn_loss.backward()
            dcn_optimizer.step()
            ep_loss += total_dcn_loss.item()
        print(f"  DCN-v2 Epoch {ep + 1}/3 - Loss: {ep_loss / len(dcn_loader):.4f}")
        
    dcn_model.eval()
    
    with torch.no_grad():
        u_rand = torch.tensor(df_rand["u_idx"].to_numpy(), dtype=torch.long)
        i_rand = torch.tensor(df_rand["i_idx"].to_numpy(), dtype=torch.long)
        dense_rand = torch.tensor(df_rand[feat_cols].to_numpy(), dtype=torch.float32)
        dcn_out_rand = dcn_model(u_rand, i_rand, dense_rand)
        df_rand["dcn_score"] = dcn_out_rand["score"].numpy() + 0.5 * df_rand["retrieval_score"].to_numpy()
        
    metric_b_ndcg = compute_group_ndcg(df_rand, user_col="user_id", rel_col="is_click", score_col="dcn_score", k=10)
    metric_b_formatted = float(round(metric_b_ndcg, 3))
    
    user_b_ndcgs = []
    for uid, grp in df_rand.groupby("user_id"):
        if len(grp) >= 5 and grp["is_click"].sum() > 0:
            user_b_ndcgs.append(compute_ndcg_at_k(grp["is_click"].to_numpy(dtype=float), grp["dcn_score"].to_numpy(dtype=float), k=10))
    ci_b = bootstrap_ci(np.array(user_b_ndcgs), n_bootstrap=1000, seed=seed)
    
    print(f"[B] Two-Tower + DCN-v2 NDCG@10 (Unbiased Random Traffic): {metric_b_formatted:.3f} (95% CI: [{ci_b['ci_lower']:.3f}, {ci_b['ci_upper']:.3f}])")
    
    # ---------------------------------------------------------
    # 5. Isotonic Probability Calibration & ECE [C]
    # ---------------------------------------------------------
    print("[*] Calibrating Stage 2 probabilities with Isotonic Regression...")
    propensity_model = ExposurePropensityModel(clip_min=0.001, clip_max=0.5)
    propensity_model.fit_from_logs(df_train, item_col="video_id")
    
    with torch.no_grad():
        u_val = torch.tensor(df_val["u_idx"].to_numpy(), dtype=torch.long)
        i_val = torch.tensor(df_val["i_idx"].to_numpy(), dtype=torch.long)
        dense_val = torch.tensor(df_val[feat_cols].to_numpy(), dtype=torch.float32)
        val_dcn_out = dcn_model(u_val, i_val, dense_val)
        val_scores = val_dcn_out["p_click"].numpy()
        
    val_props = propensity_model.predict_propensity(df_val["video_id"].to_numpy())
    val_weights = 1.0 / np.clip(val_props, 1e-4, 0.5)
    
    calibrator = RankerCalibrator()
    calibrator.fit(val_scores, df_val["is_click"].to_numpy(dtype=float), sample_weight=val_weights)
    
    rand_click_probs = dcn_out_rand["p_click"].numpy()
    uncalib_ece, calib_ece = calibrator.evaluate(rand_click_probs, df_rand["is_click"].to_numpy(dtype=float), n_bins=10)
    metric_c_ece = float(round(calib_ece, 4))
    print(f"[C] Calibration ECE: uncalibrated={uncalib_ece:.4f} -> calibrated={metric_c_ece:.4f}")
    
    # ---------------------------------------------------------
    # 6. Counterfactual Off-Policy Evaluation & Relative Error [E]%
    # ---------------------------------------------------------
    print("[*] Running Counterfactual Policy Evaluation (SNIPS vs Random Traffic Ground Truth)...")
    ope_results = evaluate_counterfactual_policy(
        df_logged=df_val,
        df_random_ground_truth=df_rand,
        propensity_model=propensity_model,
        target_scores_logged=val_scores,
        target_scores_random=rand_click_probs,
        reward_col="is_click",
        item_col="video_id",
        seed=seed,
    )
    metric_e_pct = ope_results["relative_error_pct"]
    print(f"[E]% Counterfactual SNIPS Relative Error: {metric_e_pct:.2f}% (Ground Truth Value: {ope_results['v_ground_truth']:.4f}, SNIPS: {ope_results['v_snips']:.4f})")
    
    # ---------------------------------------------------------
    # 7. FastAPI Serving Benchmark (1,000 requests) [D] ms & [Q] RPS
    # ---------------------------------------------------------
    print("[*] Benchmarking FastAPI Serving Pipeline (1,000 requests)...")
    configure_app(
        session=ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]),
        indexer=hnsw,
        model=dcn_model,
        calib=calibrator,
    )
    
    client = TestClient(app)
    
    # Warmup
    for _ in range(50):
        client.post("/recommend", json={"user_id": 1, "top_k": 10})
        
    num_requests = 1000
    latencies_ms = []
    
    t_start = time.perf_counter()
    sample_users = list(range(1, min(100, n_users + 1)))
    for req_idx in range(num_requests):
        uid = sample_users[req_idx % len(sample_users)]
        t0 = time.perf_counter()
        resp = client.post("/recommend", json={"user_id": uid, "top_k": 10})
        t1 = time.perf_counter()
        assert resp.status_code == 200
        latencies_ms.append((t1 - t0) * 1000.0)
    total_time = time.perf_counter() - t_start
    
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    rps = float(round(num_requests / total_time, 1))
    metric_d_p99 = float(round(p99, 2))
    metric_q_rps = int(round(rps))
    hardware_info = get_hardware_info()
    
    print(f"[D] Serving Latency: p50={p50:.2f} ms, p95={p95:.2f} ms, p99={metric_d_p99:.2f} ms")
    print(f"[Q] Serving Throughput: {metric_q_rps:,} RPS on {hardware_info}")
    
    # ---------------------------------------------------------
    # 8. Export metrics to benchmarks/results/rankwise_metrics.json
    # ---------------------------------------------------------
    final_metrics = {
        "benchmark_metadata": {
            "dataset": "KuaiRand-Pure",
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hardware": hardware_info,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
        "ranking_quality": {
            "baseline_lightgbm_ndcg_at_10": metric_a_formatted,
            "two_tower_dcn_v2_ndcg_at_10": metric_b_formatted,
            "two_tower_dcn_v2_ndcg_at_10_ci95": [float(round(ci_b["ci_lower"], 3)), float(round(ci_b["ci_upper"], 3))],
            "relative_ndcg_gain_pct": float(round((metric_b_formatted - metric_a_formatted) / metric_a_formatted * 100.0, 2)),
            "evaluation_slice": "held-out unbiased random-exposure traffic (log_random_4_22_to_5_08_pure.csv)"
        },
        "counterfactual_evaluation": {
            "snips_policy_value": float(round(ope_results["v_snips"], 4)),
            "unbiased_ground_truth_policy_value": float(round(ope_results["v_ground_truth"], 4)),
            "relative_estimation_error_pct": metric_e_pct,
            "snips_ci95": [float(round(ope_results["snips_ci_lower"], 4)), float(round(ope_results["snips_ci_upper"], 4))],
            "ips_policy_value": float(round(ope_results["v_ips"], 4))
        },
        "calibration": {
            "uncalibrated_ece": float(round(uncalib_ece, 4)),
            "calibrated_ece": metric_c_ece,
            "ece_reduction_pct": float(round((uncalib_ece - metric_c_ece) / uncalib_ece * 100.0, 2)),
            "calibration_method": "Isotonic Regression (10 bins)"
        },
        "serving_performance": {
            "requests_benchmarked": num_requests,
            "p50_latency_ms": float(round(p50, 2)),
            "p95_latency_ms": float(round(p95, 2)),
            "p99_latency_ms": metric_d_p99,
            "throughput_rps": metric_q_rps,
            "hardware_spec": hardware_info
        },
        "summary_metrics": {
            "[A]": f"{metric_a_formatted:.3f}",
            "[B]": f"{metric_b_formatted:.3f}",
            "[E]%": f"{metric_e_pct:.1f}%",
            "[C]": f"{metric_c_ece:.3f}",
            "[D] ms": f"{metric_d_p99:.1f} ms",
            "[Q] RPS": f"{metric_q_rps} RPS",
            "[HARDWARE]": hardware_info
        }
    }
    
    out_file = results_dir / "rankwise_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
        
    print(f"[✓] Benchmark metrics successfully saved to {out_file.relative_to(root_dir)}")
    return final_metrics


if __name__ == "__main__":
    run_benchmarks(seed=42)
