import os
import re
import argparse
import json
import warnings
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.base import clone

# Classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

# Local dataset loaders
from ml.data.oxford_loader import OxfordDatasetLoader
from ml.data.real_audio_loader import RealAudioDatasetLoader


def _get_subject_ids(metadata):
    """
    Extracts subject IDs from metadata names.
    Oxford dataset uses names like 'phon_R01_S01_1' — subject = 'phon_R01_S01'.
    This prevents patient-level data leakage across CV folds.
    """
    ids = []
    for m in metadata:
        name = m.get('name', '')
        # Strip trailing recording number (e.g. phon_R01_S01_3 -> phon_R01_S01)
        match = re.match(r'^(.+?)_\d+$', name)
        ids.append(match.group(1) if match else name)
    return np.array(ids)


def _try_import_advanced():
    """
    Attempt to import XGBoost, LightGBM, and imbalanced-learn.
    Returns dict of available libraries, gracefully degrades if missing.
    """
    libs = {}
    try:
        from xgboost import XGBClassifier
        libs['xgboost'] = XGBClassifier
        print("  [OK] XGBoost available")
    except ImportError:
        print("  [!] XGBoost not installed - skipping (pip install xgboost)")

    try:
        from lightgbm import LGBMClassifier
        libs['lightgbm'] = LGBMClassifier
        print("  [OK] LightGBM available")
    except ImportError:
        print("  [!] LightGBM not installed - skipping (pip install lightgbm)")

    try:
        from imblearn.over_sampling import SMOTE
        libs['smote'] = SMOTE
        print("  [OK] imbalanced-learn / SMOTE available")
    except ImportError:
        print("  [!] imbalanced-learn not installed - skipping SMOTE (pip install imbalanced-learn)")

    try:
        from imblearn.ensemble import BalancedRandomForestClassifier
        libs['balanced_rf'] = BalancedRandomForestClassifier
        print("  [OK] BalancedRandomForestClassifier available")
    except ImportError:
        pass

    return libs


def train_vita_voice(
    dataset_type="oxford",
    root_dir="datasets",
    reduction_method="pca",
    n_components=16,
    checkpoints_dir="ml/checkpoints"
):
    os.makedirs(checkpoints_dir, exist_ok=True)

    print("\n" + "="*70)
    print("  VitaVoice ML Training Pipeline - Research-Upgraded Edition")
    print("="*70)
    print("\n[1/9] Checking advanced library availability...")
    libs = _try_import_advanced()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Load dataset
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/9] Loading dataset...")

    if dataset_type == "oxford":
        data_path = os.path.join(root_dir, "parkinsons.data")
        wav_dir = os.path.join(root_dir, "synthesized_wavs")
        cache_path = os.path.join(root_dir, "oxford_cache.joblib")
        # if os.path.exists(cache_path):
        #     os.remove(cache_path)
        print(f"Loading Oxford dataset (direct tabular mode) from {data_path}...")
        # direct_mode=True reads all 22 features including RPDE, DFA, PPE, etc.
        loader = OxfordDatasetLoader(
            data_path=data_path,
            wav_dir=wav_dir,
            cache_path=cache_path,
            direct_mode=True
        )
    elif dataset_type == "real":
        real_dir = os.path.join(root_dir, "real")
        cache_path = os.path.join(root_dir, "real_cache.joblib")
        # if os.path.exists(cache_path):
        #     os.remove(cache_path)
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

    print(f"\n  Dataset shape    : {X_cli.shape[0]} samples x {X_cli.shape[1]} features")
    print(f"  Class distribution: {int(np.sum(y_arr==1))} PD / {int(np.sum(y_arr==0))} Healthy")
    print(f"  Imbalance ratio  : {np.sum(y_arr==1)/np.sum(y_arr==0):.1f}:1")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. PCA Visualization (always 2D, for cluster map only)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/9] Fitting PCA for 2D embedding visualization...")
    pca_vis = PCA(n_components=2)
    X_w2v_2d = pca_vis.fit_transform(X_w2v)
    joblib.dump(pca_vis, os.path.join(checkpoints_dir, "pca_model.joblib"))

    import pandas as pd
    vis_df = pd.DataFrame({
        'pca_x': X_w2v_2d[:, 0],
        'pca_y': X_w2v_2d[:, 1],
        'status': y_arr
    })
    vis_df.to_csv("datasets/pca_visualization_clusters.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. WavLM Dimensionality Reduction for OOD detection
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[4/9] Fitting {reduction_method.upper()} reducer for WavLM embeddings ({n_components} components)...")

    if reduction_method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
        X_w2v_reduced = reducer.fit_transform(X_w2v)
    elif reduction_method == "umap":
        import umap
        reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=15, min_dist=0.1)
        X_w2v_reduced = reducer.fit_transform(X_w2v)
    else:
        reducer = None
        X_w2v_reduced = X_w2v

    joblib.dump(reducer, os.path.join(checkpoints_dir, "reducer.joblib"))
    joblib.dump(reduction_method, os.path.join(checkpoints_dir, "reduction_method.joblib"))

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Feature Engineering: Log Transforms + Scaling
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[5/9] Feature engineering + scaling...")
    X_fused = X_cli
    print(f"  Feature matrix for classification: {X_fused.shape}")

    # Apply log1p to right-skewed perturbation features (jitter, shimmer, NHR)
    LOG_TRANSFORM_COLS = [
        'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'Jitter:DDP', 'NHR'
    ]
    log_transform_indices = []
    feature_names_list = list(feature_names)
    for col in LOG_TRANSFORM_COLS:
        if col in feature_names_list:
            log_transform_indices.append(feature_names_list.index(col))

    def apply_log_transforms(X, indices):
        X_out = X.copy().astype(np.float64)
        for idx in indices:
            X_out[:, idx] = np.log1p(np.maximum(X_out[:, idx], 0))
        return X_out

    X_fused_log = apply_log_transforms(X_fused, log_transform_indices)
    print(f"  Log1p transforms applied to {len(log_transform_indices)} skewed features")

    # Fit initial scaler on all features to run feature selection
    y_arr_np = np.array(y_arr)
    initial_scaler = StandardScaler()
    X_scaled_initial = initial_scaler.fit_transform(X_fused_log)

    # Perform L1 Logistic Regression feature selection to select top features
    print("\n[5.5/9] Performing L1-regularized feature selection (collinearity reduction)...")
    selector = SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', C=0.8, class_weight='balanced', random_state=42),
        max_features=12
    )
    selector.fit(X_scaled_initial, y_arr_np)
    selected_indices = list(selector.get_support(indices=True))
    selected_feature_names = [feature_names_list[i] for i in selected_indices]
    print(f"  Selected {len(selected_feature_names)} features: {selected_feature_names}")
    
    # Save SELECTED feature names for inference
    joblib.dump(selected_feature_names, os.path.join(checkpoints_dir, "feature_names.joblib"))

    # Update transform indices matching the selected features
    selected_log_transform_indices = []
    for col in LOG_TRANSFORM_COLS:
        if col in selected_feature_names:
            selected_log_transform_indices.append(selected_feature_names.index(col))
            
    # Save transform config for inference (consistent with selected features)
    transform_config = {
        'log_transform_indices': selected_log_transform_indices,
        'log_transform_cols': LOG_TRANSFORM_COLS
    }
    joblib.dump(transform_config, os.path.join(checkpoints_dir, "transform_config.joblib"))

    # Filter features matrix to selected features
    X_fused_selected = X_fused[:, selected_indices]
    X_fused_log_selected = apply_log_transforms(X_fused_selected, selected_log_transform_indices)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fused_log_selected)
    joblib.dump(scaler, os.path.join(checkpoints_dir, "scaler.joblib"))
    joblib.dump(X_scaled, os.path.join(checkpoints_dir, "background_data.joblib"))

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Build Model Grid (with advanced models if available)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[6/9] Building model benchmarking grid...")
    from sklearn.model_selection import RandomizedSearchCV

    # Compute positive class weight for imbalanced datasets
    n_neg = np.sum(y_arr_np == 0)
    n_pos = np.sum(y_arr_np == 1)
    scale_pos_weight = float(n_neg / n_pos)
    print(f"  scale_pos_weight (neg/pos ratio): {scale_pos_weight:.2f}")

    grids = {
        'svm': {
            'estimator': CalibratedClassifierCV(
                SVC(class_weight='balanced', random_state=42),
                method='sigmoid',
                cv=3
            ),
            'params': {
                'estimator__C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
                'estimator__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'estimator__kernel': ['rbf', 'linear']
            },
            'n_iter': 30
        },
        'random_forest': {
            'estimator': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
            'params': {
                'n_estimators': [200, 300, 500, 700],
                'max_depth': [4, 6, 8, 10, None],
                'min_samples_split': [2, 3, 5],
                'min_samples_leaf': [1, 2, 3],
                'max_features': ['sqrt', 'log2', 0.3, 0.5],
                'criterion': ['gini', 'entropy'],
            },
            'n_iter': 50
        },
        'gradient_boosting': {
            'estimator': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [2, 3, 4, 5],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10],
                'max_features': ['sqrt', 'log2'],
            },
            'n_iter': 50
        },
    }

    # Add XGBoost if available
    if 'xgboost' in libs:
        grids['xgboost'] = {
            'estimator': libs['xgboost'](
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                use_label_encoder=False if hasattr(libs['xgboost'](), 'use_label_encoder') else None,
                n_jobs=-1,
                verbosity=0
            ),
            'params': {
                'n_estimators': [100, 200, 300, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 4, 5, 6, 7],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0.5, 1.0, 2.0, 5.0],
                'min_child_weight': [1, 3, 5],
            },
            'n_iter': 80
        }

    # Add LightGBM if available
    if 'lightgbm' in libs:
        grids['lightgbm'] = {
            'estimator': libs['lightgbm'](
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
                verbosity=-1,
                force_col_wise=True
            ),
            'params': {
                'n_estimators': [100, 200, 300, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 4, 5, 6, 8, -1],
                'num_leaves': [15, 31, 63, 127],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.5, 0.7, 0.8, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0.1, 0.5, 1.0, 2.0],
                'min_child_samples': [5, 10, 20],
            },
            'n_iter': 80
        }

    # Add BalancedRandomForest if available
    if 'balanced_rf' in libs:
        grids['balanced_rf'] = {
            'estimator': libs['balanced_rf'](random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [200, 300, 500],
                'max_depth': [6, 8, 10, None],
                'min_samples_split': [2, 3, 5],
                'max_features': ['sqrt', 'log2'],
            },
            'n_iter': 30
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Cross-Validation Strategy
    # ─────────────────────────────────────────────────────────────────────────
    subject_ids = _get_subject_ids(metadata)
    n_unique_subjects = len(np.unique(subject_ids))
    print(f"\n  Unique subjects for GroupKFold: {n_unique_subjects}")

    if n_unique_subjects >= 10:
        print("  Using StratifiedGroupKFold (prevents patient-level data leakage)")
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        cv_split_args = {'groups': subject_ids}
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        search_fit_args = {'groups': subject_ids}
    else:
        print("  Using StratifiedKFold (not enough unique subjects for GroupKFold)")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_split_args = {}
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        search_fit_args = {}

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Hyperparameter Optimization
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[7/9] Hyperparameter optimization across base models...")
    best_tuned_models = {}

    # SMOTE setup
    smote = libs.get('smote', None)
    use_smote = smote is not None

    for model_name, config in grids.items():
        print(f"\n  Optimizing: {model_name.upper()}")
        estimator = config['estimator']
        params = {k: v for k, v in config['params'].items() if k is not None}

        try:
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=params,
                n_iter=config['n_iter'],
                cv=inner_cv,
                scoring='roc_auc',
                random_state=42,
                n_jobs=1,
                error_score=0.0
            )
            search.fit(X_scaled, y_arr_np, **search_fit_args)
            best_model = search.best_estimator_
            best_tuned_models[model_name] = best_model
            print(f"  Best params: {search.best_params_}")
            print(f"  Best inner CV ROC-AUC: {search.best_score_:.4f}")
        except Exception as e:
            print(f"  [ERROR] Skipping {model_name}: {e}")
            continue

    # ─────────────────────────────────────────────────────────────────────────
    # 7.5. Build Stacking and Soft Voting Ensembles
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[7.5/9] Building Stacking and Soft Voting Ensembles...")
    base_ensemble_estimators = []
    # Include all tuned classifiers that support predict_proba
    for base_name in ['svm', 'random_forest', 'xgboost', 'lightgbm', 'balanced_rf']:
        if base_name in best_tuned_models:
            base_ensemble_estimators.append((base_name, best_tuned_models[base_name]))
            
    if len(base_ensemble_estimators) >= 2:
        voting_clf = VotingClassifier(
            estimators=base_ensemble_estimators,
            voting='soft'
        )
        stacking_clf = StackingClassifier(
            estimators=base_ensemble_estimators,
            final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
            cv=3,
            n_jobs=-1
        )
        best_tuned_models['voting'] = voting_clf
        best_tuned_models['stacking'] = stacking_clf
        print("  Voting and Stacking ensembles successfully added to evaluation list.")

    # ─────────────────────────────────────────────────────────────────────────
    # 7.6. Outer CV Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    print("\nEvaluating all models using outer 5-fold CV...")
    benchmarks = {}

    for model_name, best_model in best_tuned_models.items():
        print(f"  Evaluating: {model_name.upper()}...")
        accs, precs, recs, f1s, aucs, pr_aucs = [], [], [], [], [], []

        split_fn = cv.split(X_scaled, y_arr_np, **cv_split_args)

        for fold_idx, (train_idx, val_idx) in enumerate(split_fn):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_arr_np[train_idx], y_arr_np[val_idx]

            # Apply SMOTE on training fold only
            if use_smote:
                try:
                    sm = smote(random_state=42 + fold_idx, k_neighbors=min(5, np.sum(y_train==0) - 1))
                    X_train, y_train = sm.fit_resample(X_train, y_train)
                except Exception:
                    pass  # Fall back to unbalanced if SMOTE fails

            fold_model = clone(best_model)
            try:
                fold_model.fit(X_train, y_train)
            except Exception:
                # Fallback for models that might fail clone/fit in edge cases
                fold_model = best_model
                fold_model.fit(X_train, y_train)

            y_pred = fold_model.predict(X_val)
            y_prob = fold_model.predict_proba(X_val)[:, 1]

            accs.append(accuracy_score(y_val, y_pred))
            precs.append(precision_score(y_val, y_pred, zero_division=0))
            recs.append(recall_score(y_val, y_pred, zero_division=0))
            f1s.append(f1_score(y_val, y_pred, zero_division=0))
            aucs.append(roc_auc_score(y_val, y_prob))
            pr_aucs.append(average_precision_score(y_val, y_prob))

        benchmarks[model_name] = {
            'accuracy': float(np.mean(accs)),
            'precision': float(np.mean(precs)),
            'recall': float(np.mean(recs)),
            'f1': float(np.mean(f1s)),
            'roc_auc': float(np.mean(aucs)),
            'pr_auc': float(np.mean(pr_aucs))
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Print Benchmark Table
    # ─────────────────────────────────────────────────────────────────────────
    print("\n\n" + "="*75)
    print("  Model Performance Comparison (5-Fold GroupKFold CV + SMOTE)")
    print("="*75)
    print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9} {'PR-AUC':>8}")
    print("-"*75)
    for m_name, metrics in benchmarks.items():
        pr = metrics.get('pr_auc', 0.0)
        print(f"  {m_name.upper():<20} {metrics['accuracy']:.4f}    {metrics['precision']:.4f}  "
              f"{metrics['recall']:.4f}  {metrics['f1']:.4f}  {metrics['roc_auc']:.4f}  {pr:.4f}")

    # Save benchmarks
    with open(os.path.join(checkpoints_dir, "model_benchmarks.json"), "w") as f:
        json.dump(benchmarks, f, indent=2)

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Select Best Model by ROC-AUC + Calibrate Probabilities
    # ─────────────────────────────────────────────────────────────────────────
    if not benchmarks:
        print("[ERROR] No models benchmarked successfully.")
        return False

    best_model_name = max(benchmarks, key=lambda k: benchmarks[k]['roc_auc'])
    best_metrics = benchmarks[best_model_name]
    print(f"\n  * Selected Best Model: {best_model_name.upper()}")
    print(f"    F1-Score: {best_metrics['f1']:.4f} | ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"    Recall:   {best_metrics['recall']:.4f} | Precision: {best_metrics['precision']:.4f}")

    # Apply SMOTE on full training data before final fit
    X_final, y_final = X_scaled, y_arr_np
    if use_smote:
        try:
            sm_full = smote(random_state=42, k_neighbors=min(5, n_neg - 1))
            X_final, y_final = sm_full.fit_resample(X_final, y_final)
            print(f"\n  SMOTE applied: {len(y_final)} samples after oversampling "
                  f"({int(np.sum(y_final==1))} PD / {int(np.sum(y_final==0))} Healthy)")
        except Exception as e:
            print(f"  [WARN] SMOTE on full data failed: {e}. Using original data.")

    best_model_raw = best_tuned_models[best_model_name]
    best_model_raw.fit(X_final, y_final)

    # Calibrate probability outputs using Platt scaling (sigmoid)
    print("\n  Calibrating probability outputs (Platt scaling)...")
    inner_cv_cal = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    best_model_calibrated = CalibratedClassifierCV(
        estimator=clone(best_model_raw),
        method='sigmoid',
        cv=inner_cv_cal
    )
    best_model_calibrated.fit(X_scaled, y_arr_np)  # Calibrate on original (uncapitalized) data

    # Save calibrated model
    joblib.dump(best_model_calibrated, os.path.join(checkpoints_dir, "classifier_model.joblib"))
    # Save uncalibrated for TreeSHAP
    joblib.dump(best_model_raw, os.path.join(checkpoints_dir, "classifier_model_raw.joblib"))
    joblib.dump(best_model_name, os.path.join(checkpoints_dir, "model_type.joblib"))

    print(f"  Saved calibrated model checkpoint (type: {best_model_name})")

    # ─────────────────────────────────────────────────────────────────────────
    # 10. Build WavLM Reference Assets
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[8/9] Generating WavLM reference assets...")
    try:
        from sklearn.svm import OneClassSVM
        from sklearn.covariance import LedoitWolf

        y_np = y_arr_np
        healthy_mask = (y_np == 0)
        parkinsons_mask = (y_np == 1)

        healthy_centroid = np.mean(X_w2v[healthy_mask], axis=0)
        parkinsons_centroid = np.mean(X_w2v[parkinsons_mask], axis=0)

        ood_detector = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
        ood_detector.fit(X_w2v)
        train_scores = ood_detector.score_samples(X_w2v)
        ood_threshold = float(np.percentile(train_scores, 5))

        if reducer is not None:
            X_w2v_red = reducer.transform(X_w2v)
        else:
            X_w2v_red = X_w2v

        mean_healthy_red = np.mean(X_w2v_red[healthy_mask], axis=0)
        mean_pd_red = np.mean(X_w2v_red[parkinsons_mask], axis=0)

        cov_h = LedoitWolf().fit(X_w2v_red[healthy_mask]).covariance_
        cov_p = LedoitWolf().fit(X_w2v_red[parkinsons_mask]).covariance_
        reg = 1e-4 * np.eye(X_w2v_red.shape[1])
        cov_h_inv = np.linalg.inv(cov_h + reg)
        cov_p_inv = np.linalg.inv(cov_p + reg)

        references = {
            'healthy_centroid': healthy_centroid,
            'parkinsons_centroid': parkinsons_centroid,
            'train_embeddings': X_w2v,
            'train_labels': y_np,
            'ood_detector': ood_detector,
            'ood_threshold': ood_threshold,
            'mean_healthy_reduced': mean_healthy_red,
            'mean_parkinsons_reduced': mean_pd_red,
            'cov_healthy_inv': cov_h_inv,
            'cov_parkinsons_inv': cov_p_inv
        }
        joblib.dump(references, os.path.join(checkpoints_dir, "wavlm_references.joblib"))
        print("  WavLM references saved.")
    except Exception as e:
        print(f"  [WARN] Error generating WavLM references: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # 11. Final Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[9/9] Training complete!")
    print("\n" + "="*70)
    print("  FINAL MODEL PERFORMANCE SUMMARY")
    print("="*70)
    print(f"  Best Model   : {best_model_name.upper()}")
    print(f"  Accuracy     : {best_metrics['accuracy']:.4f}")
    print(f"  F1-Score     : {best_metrics['f1']:.4f}")
    print(f"  ROC-AUC      : {best_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC       : {best_metrics.get('pr_auc', 0.0):.4f}")
    print(f"  Recall (Sens): {best_metrics['recall']:.4f}")
    print(f"  Precision    : {best_metrics['precision']:.4f}")
    print("="*70)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VitaVoice Upgraded ML Training & Benchmarking Pipeline")
    parser.add_argument("--dataset", type=str, default="oxford", choices=["oxford", "real"],
                        help="Dataset type (default: 'oxford')")
    parser.add_argument("--reduction", type=str, default="pca", choices=["pca", "umap", "none"],
                        help="Dimensionality reduction for WavLM embeddings (default: 'pca')")
    parser.add_argument("--components", type=int, default=16,
                        help="Number of PCA/UMAP components (default: 16)")
    parser.add_argument("--data-dir", type=str, default="datasets",
                        help="Root folder of datasets (default: 'datasets')")
    args = parser.parse_args()

    train_vita_voice(
        dataset_type=args.dataset,
        root_dir=args.data_dir,
        reduction_method=args.reduction,
        n_components=args.components
    )
