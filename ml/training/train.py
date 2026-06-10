import os
import argparse
import json
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Local dataset loaders
from ml.data.oxford_loader import OxfordDatasetLoader
from ml.data.real_audio_loader import RealAudioDatasetLoader

def train_vita_voice(dataset_type="oxford", root_dir="datasets", reduction_method="pca", n_components=16, checkpoints_dir="ml/checkpoints"):
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # 1. Instantiate the selected Dataset Loader
    if dataset_type == "oxford":
        data_path = os.path.join(root_dir, "parkinsons.data")
        wav_dir = os.path.join(root_dir, "synthesized_wavs")
        cache_path = os.path.join(root_dir, "oxford_cache.joblib")
        # Oxford loader has cache, so we can delete the cache to force reload if features shape changed
        # Since we added new clinical features, we should force delete the cache
        if os.path.exists(cache_path):
            os.remove(cache_path)
        print(f"Loading Oxford demo dataset from {data_path}...")
        loader = OxfordDatasetLoader(data_path=data_path, wav_dir=wav_dir, cache_path=cache_path)
    elif dataset_type == "real":
        real_dir = os.path.join(root_dir, "real")
        cache_path = os.path.join(root_dir, "real_cache.joblib")
        if os.path.exists(cache_path):
            os.remove(cache_path)
        print(f"Loading real clinical voice recordings from {real_dir}...")
        loader = RealAudioDatasetLoader(root_dir=real_dir, cache_path=cache_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
        
    X_dict, y_arr, metadata = loader.load_samples()
    
    if len(y_arr) == 0:
        print("[ERROR] Cannot train model. The selected dataset is empty.")
        return False
        
    X_cli = X_dict['X_cli']
    X_w2v = X_dict['X_w2v']
    feature_names = X_dict['feature_names']
    
    # Write feature names index for reference during inference
    joblib.dump(feature_names, os.path.join(checkpoints_dir, "feature_names.joblib"))
    
    # 2. PCA for 2D visualization (always PCA, 2 components for clusters plot)
    print("Fitting PCA for 2D visualization clusters...")
    pca_vis = PCA(n_components=2)
    X_w2v_2d = pca_vis.fit_transform(X_w2v)
    joblib.dump(pca_vis, os.path.join(checkpoints_dir, "pca_model.joblib"))
    
    # Save coordinate references
    import pandas as pd
    vis_df = pd.DataFrame({
        'pca_x': X_w2v_2d[:, 0],
        'pca_y': X_w2v_2d[:, 1],
        'status': y_arr
    })
    vis_df.to_csv("datasets/pca_visualization_clusters.csv", index=False)
    
    # 3. Dimensionality Reduction Layer
    if reduction_method == "pca":
        print(f"Fitting PCA reduction for neural embeddings ({n_components} components)...")
        reducer = PCA(n_components=n_components, random_state=42)
        X_w2v_reduced = reducer.fit_transform(X_w2v)
    elif reduction_method == "umap":
        print(f"Fitting UMAP reduction for neural embeddings ({n_components} components)...")
        import umap
        reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=15, min_dist=0.1)
        X_w2v_reduced = reducer.fit_transform(X_w2v)
    else:
        print("Bypassing dimensionality reduction for neural embeddings...")
        reducer = None
        X_w2v_reduced = X_w2v
        
    # Save reducer
    joblib.dump(reducer, os.path.join(checkpoints_dir, "reducer.joblib"))
    joblib.dump(reduction_method, os.path.join(checkpoints_dir, "reduction_method.joblib"))
    
    # 4. Concatenate Acoustic + Reduced Embeddings
    X_fused = np.hstack((X_cli, X_w2v_reduced))
    print(f"Fused feature matrix shape: {X_fused.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fused)
    joblib.dump(scaler, os.path.join(checkpoints_dir, "scaler.joblib"))
    
    # Save a copy of training scaled dataset to use as background for SHAP Explainer
    joblib.dump(X_scaled, os.path.join(checkpoints_dir, "background_data.joblib"))
    
    # 5. Define Benchmark Classifiers
    models = {
        'svm': SVC(kernel='rbf', C=2.0, probability=True, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'xgboost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, eval_metric='logloss', random_state=42),
        'lightgbm': LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42, verbose=-1),
        'logistic_regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    }
    
    # 6. Benchmarking using 5-Fold Stratified Cross-Validation
    print("\nStarting model benchmarking (5-Fold Stratified Cross-Validation)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    benchmarks = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        accs, precs, recs, f1s, aucs = [], [], [], [], []
        
        for train_idx, val_idx in skf.split(X_scaled, y_arr):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_arr[train_idx], y_arr[val_idx]
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]
            
            # Compute metrics
            accs.append(accuracy_score(y_val, y_pred))
            precs.append(precision_score(y_val, y_pred, zero_division=0))
            recs.append(recall_score(y_val, y_pred, zero_division=0))
            f1s.append(f1_score(y_val, y_pred, zero_division=0))
            aucs.append(roc_auc_score(y_val, y_prob))
            
        benchmarks[model_name] = {
            'accuracy': float(np.mean(accs)),
            'precision': float(np.mean(precs)),
            'recall': float(np.mean(recs)),
            'f1': float(np.mean(f1s)),
            'roc_auc': float(np.mean(aucs))
        }
        
    # 7. Print Benchmark Table in Markdown
    print("\n### Model Performance Comparison")
    print("| Model Class | Accuracy | Precision | Recall | F1-Score | ROC-AUC |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for m_name, metrics in benchmarks.items():
        print(f"| {m_name.upper()} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['roc_auc']:.4f} |")
        
    # Save benchmarks to JSON
    with open(os.path.join(checkpoints_dir, "model_benchmarks.json"), "w") as f:
        json.dump(benchmarks, f, indent=2)
        
    # 8. Auto-Select Best Model (Based on highest F1-Score)
    best_model_name = max(benchmarks, key=lambda k: benchmarks[k]['f1'])
    print(f"\nAuto-Selected Winner: **{best_model_name.upper()}** (F1-Score: {benchmarks[best_model_name]['f1']:.4f})")
    
    # Train winning model on all data
    best_model = models[best_model_name]
    best_model.fit(X_scaled, y_arr)
    
    # Save winning checkpoint
    joblib.dump(best_model, os.path.join(checkpoints_dir, "classifier_model.joblib"))
    joblib.dump(best_model_name, os.path.join(checkpoints_dir, "model_type.joblib"))
    print(f"Saved best model checkpoint and type '{best_model_name}' to checkpoints.")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VitaVoice Upgraded Model Training & Benchmarking Pipeline")
    parser.add_argument("--dataset", type=str, default="oxford", choices=["oxford", "real"],
                        help="The dataset type to train on (default: 'oxford')")
    parser.add_argument("--reduction", type=str, default="pca", choices=["pca", "umap", "none"],
                        help="Dimensionality reduction method for neural embeddings (default: 'pca')")
    parser.add_argument("--components", type=int, default=16,
                        help="Number of components for reduction (default: 16)")
    parser.add_argument("--data-dir", type=str, default="datasets",
                        help="Root folder of the datasets (default: 'datasets')")
    args = parser.parse_args()
    
    train_vita_voice(
        dataset_type=args.dataset, 
        root_dir=args.data_dir, 
        reduction_method=args.reduction, 
        n_components=args.components
    )
