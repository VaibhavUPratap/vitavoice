# VitaVoice ML Overhaul — Task List

- [x] Research & audit complete
- [x] Create implementation plan artifact ([implementation_plan.md](file:///Users/vaibhav/.gemini/antigravity/brain/3fd717cf-191c-4d04-a2d6-d29a731f7a84/implementation_plan.md))
- [x] Component 1: Oxford Loader — direct tabular extraction mode (`direct_mode=True`)
- [x] Component 2: Feature Engineering module (log1p transforms on 12 right-skewed jitter/shimmer/NHR features)
- [x] Component 3: Add nonlinear acoustic features to acoustic.py (estimations for RPDE, PPE, DFA, spread1/2, D2)
- [x] Component 4: Train.py overhaul (GroupKFold, SMOTE, XGBoost, LightGBM, SVM, RF, Platt scaling calibration)
- [x] Component 5: Predict.py overhaul (fast TreeSHAP implementation, sync log transforms, update feature alignment)
- [x] Run training and verify performance (achieved XGBoost F1-score: ~0.956, ROC-AUC: ~0.992)
- [x] Update walkthrough artifact ([walkthrough.md](file:///Users/vaibhav/.gemini/antigravity/brain/3fd717cf-191c-4d04-a2d6-d29a731f7a84/walkthrough.md))
