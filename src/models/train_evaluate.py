import os
import sys
import time
import numpy as np
import pandas as pd

# Fix working directory
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join("src", "models"))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from decision_tree       import DecisionTree
from random_forest       import RandomForest
from xgboost_scratch     import XGBoostScratch
from svm_scratch         import SVMScratch
from dnn_scratch         import DNNScratch
from transformer_scratch import TransformerScratch

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH    = os.path.join("data", "processed", "final_dataset.csv")
RESULTS_PATH = os.path.join("results", "model_comparison.csv")
SAMPLE_SIZE  = 50000
RANDOM_SEED  = 42
TEST_SIZE    = 0.3
N_RUNS       = 3

# ── Load Data ─────────────────────────────────────────────────────────────────
def load_data(sample_size=None):
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Loaded {len(df)} pairs")

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=RANDOM_SEED)
        print(f"   Sampled {sample_size} pairs")

    feature_cols = [c for c in df.columns
                    if c not in ["asn1", "asn2", "label"]]
    X = df[feature_cols].values
    y = df["label"].values

    print(f"   Features : {len(feature_cols)}")
    print(f"   Peering  : {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
    print(f"   Non-peer : {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    return X, y


# ── Evaluate One Model ────────────────────────────────────────────────────────
def evaluate_model(name, model, X, y, scale=False):
    print(f"\n{'='*55}")
    print(f"  Training: {name}")
    print(f"{'='*55}")

    overall_accs  = []
    balanced_accs = []
    f1_scores     = []
    train_times   = []
    pred_times    = []

    for run in range(N_RUNS):
        seed = RANDOM_SEED + run
        print(f"\n  Run {run+1}/{N_RUNS} (seed={seed})")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed
        )

        if scale:
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        # Train
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        train_times.append(train_time)

        # Predict
        start  = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start
        pred_times.append(pred_time)

        # Metrics
        oa = accuracy_score(y_test, y_pred) * 100
        ba = balanced_accuracy_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred) * 100

        overall_accs.append(oa)
        balanced_accs.append(ba)
        f1_scores.append(f1)

        print(f"    Overall Acc  : {oa:.2f}%")
        print(f"    Balanced Acc : {ba:.2f}%")
        print(f"    F1 Score     : {f1:.2f}%")
        print(f"    Train Time   : {train_time:.2f}s")

    results = {
        "Model"              : name,
        "Overall Acc (%)"    : round(np.mean(overall_accs),  2),
        "Overall Std"        : round(np.std(overall_accs),   2),
        "Balanced Acc (%)"   : round(np.mean(balanced_accs), 2),
        "Balanced Std"       : round(np.std(balanced_accs),  2),
        "F1 Score (%)"       : round(np.mean(f1_scores),     2),
        "F1 Std"             : round(np.std(f1_scores),      2),
        "Avg Train Time (s)" : round(np.mean(train_times),   2),
        "Avg Pred Time (s)"  : round(np.mean(pred_times),    4),
    }

    print(f"\n  ── Average across {N_RUNS} runs ──")
    print(f"  Overall Acc  : {results['Overall Acc (%)']:.2f}%"
          f" ± {results['Overall Std']:.2f}")
    print(f"  Balanced Acc : {results['Balanced Acc (%)']:.2f}%"
          f" ± {results['Balanced Std']:.2f}")
    print(f"  F1 Score     : {results['F1 Score (%)']:.2f}%"
          f" ± {results['F1 Std']:.2f}")
    print(f"  Train Time   : {results['Avg Train Time (s)']:.2f}s")

    # Save intermediate results after each model
    os.makedirs("results", exist_ok=True)
    pd.DataFrame([results]).to_csv(
        os.path.join("results", f"{name.replace(' ','_')}_result.csv"),
        index=False
    )
    print(f"Intermediate result saved!")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("  PEERING PARTNER PREDICTION")
    print("  Training all 6 models from scratch")
    print("="*55)

    X, y = load_data(sample_size=SAMPLE_SIZE)

    # ── Define all 6 models ───────────────────────────────────────────────────
    models = [
        (
            "Decision Tree",
            DecisionTree(
                max_depth         = 10,
                min_samples_split = 5,
                min_samples_leaf  = 1
            ),
            False
        ),
        (
            "Random Forest",
            RandomForest(
                n_trees           = 50,
                max_depth         = 10,
                min_samples_split = 5,
                min_samples_leaf  = 1,
                max_features      = "sqrt"
            ),
            False
        ),
        (
            "XGBoost",
            XGBoostScratch(
                n_trees           = 50,
                max_depth         = 6,
                learning_rate     = 0.1,
                min_samples_split = 5,
                subsample         = 0.8
            ),
            False
        ),
        (
            "SVM",
            SVMScratch(
                learning_rate = 0.001,
                lambda_param  = 0.01,
                n_epochs      = 100,
                batch_size    = 256
            ),
            True    # needs scaling
        ),
        (
            "DNN",
            DNNScratch(
                hidden_layers = [100, 100, 100],
                learning_rate = 0.001,
                n_epochs      = 100,
                batch_size    = 256
            ),
            True    # needs scaling
        ),
        (
            "Transformer",
            TransformerScratch(
                d_model       = 16,
                n_heads       = 2,
                d_ff          = 32,
                learning_rate = 0.001,
                n_epochs      = 50,
                batch_size    = 256
            ),
            True    # needs scaling
        ),
    ]

    # ── Train all models ──────────────────────────────────────────────────────
    all_results = []
    total_start = time.time()

    for name, model, scale in models:
        result = evaluate_model(name, model, X, y, scale=scale)
        all_results.append(result)

    total_time = time.time() - total_start

    # ── Final Comparison Table ────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  FINAL RESULTS COMPARISON")
    print("="*55)

    df_results = pd.DataFrame(all_results)
    print(df_results[[
        "Model",
        "Overall Acc (%)",
        "Balanced Acc (%)",
        "F1 Score (%)",
        "Avg Train Time (s)"
    ]].to_string(index=False))

    print(f"\n  Total experiment time: {total_time/60:.1f} minutes")

    # ── Paper Comparison ──────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  COMPARISON WITH PAPER (Table 1)")
    print("="*55)

    paper = {
        "Decision Tree" : (None,   None),
        "Random Forest" : (97.375, 98.430),
        "XGBoost"       : (97.593, 98.558),
        "SVM"           : (95.560, 97.348),
        "DNN"           : (96.181, 97.695),
        "Transformer"   : (96.222, 96.211),
    }

    print(f"{'Model':<16} {'Paper Acc':>10} {'Our Acc':>10}"
          f" {'Paper F1':>10} {'Our F1':>10}")
    print("-"*60)

    for r in all_results:
        name          = r["Model"]
        our_acc       = r["Overall Acc (%)"]
        our_f1        = r["F1 Score (%)"]
        p_acc, p_f1   = paper.get(name, (None, None))
        p_acc_str     = f"{p_acc:.2f}%" if p_acc else "N/A"
        p_f1_str      = f"{p_f1:.2f}%" if p_f1 else "N/A"
        print(f"{name:<16} {p_acc_str:>10} {our_acc:>9.2f}%"
              f" {p_f1_str:>10} {our_f1:>9.2f}%")

    # ── Save final results ────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    df_results.to_csv(RESULTS_PATH, index=False)
    print(f"\n  Final results saved to: {RESULTS_PATH}")
    print("\n All 6 models trained and evaluated!")
    print("  Script finished!")