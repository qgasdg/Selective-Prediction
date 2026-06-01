"""
Reproduce the tables from the report using fixed best hyperparameters.

No hyperparameter search is performed — params are taken directly from
best_hyperparams.json (or the hardcoded defaults below).  The random seed
is fixed so results are deterministic across runs.

Usage
-----
  python reproduce.py [--data DATA_PATH] [--tau TAU]
"""

import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.preprocess import load_and_clean, engineer_features, split_data
from src.models import get_baseline, SelectivePredictor
from src.evaluate import (
    evaluate_standard, threshold_sweep_table, print_sweep,
    plot_confidence_distribution, plot_confusion_matrices,
)
from src.eda import run_eda

SEED = 42

# Best hyperparameters from the report (RandomizedSearchCV, 30 iter, 3-fold CV)
RF_PARAMS = {
    'n_estimators':      300,
    'max_depth':         30,
    'min_samples_split': 10,
    'min_samples_leaf':  4,
    'max_features':      'sqrt',
}

XGB_PARAMS = {
    'n_estimators':     200,
    'max_depth':        4,
    'learning_rate':    0.05,
    'subsample':        0.7,
    'colsample_bytree': 0.6,
    'min_child_weight': 3,
    'gamma':            0.1,
}


def section(title: str):
    bar = '=' * 60
    print(f'\n{bar}\n  {title}\n{bar}')


RESULTS_DIR = 'results'


def main(data_path: str, tau: float):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(f'{RESULTS_DIR}/eda', exist_ok=True)

    # ── EDA ───────────────────────────────────────────────────────────────────
    section('EDA')
    df_raw = load_and_clean(data_path)
    run_eda(df_raw, out_dir=f'{RESULTS_DIR}/eda')

    # ── Data ──────────────────────────────────────────────────────────────────
    section('Data')
    df = engineer_features(df_raw)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, random_state=SEED)
    y_test_np = y_test.values

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_wt = neg / pos
    print(f'Train {len(X_train):,} | Val {len(X_val):,} | Test {len(X_test):,}')
    print(f'Neg/Pos ratio: {scale_pos_wt:.2f}')

    # ── Logistic Regression ───────────────────────────────────────────────────
    section('Logistic Regression  (baseline)')
    lr = get_baseline(random_state=SEED)
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)
    lr_pred  = lr_proba.argmax(axis=1)
    lr_meta  = evaluate_standard(y_test_np, lr_pred, lr_proba, 'Logistic Regression')
    print(pd.Series(lr_meta).to_string())

    # ── Random Forest ─────────────────────────────────────────────────────────
    section('Random Forest  (best params from report)')
    print('Params:', RF_PARAMS)
    rf = RandomForestClassifier(
        **RF_PARAMS,
        class_weight='balanced',
        n_jobs=-1,
        random_state=SEED,
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)
    rf_pred  = rf_proba.argmax(axis=1)
    rf_meta  = evaluate_standard(y_test_np, rf_pred, rf_proba, 'Random Forest')
    print(pd.Series(rf_meta).to_string())

    # ── XGBoost ───────────────────────────────────────────────────────────────
    section('XGBoost  (best params from report)')
    print('Params:', XGB_PARAMS)
    xgb = XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_wt,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1,
        random_state=SEED,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)
    xgb_pred  = xgb_proba.argmax(axis=1)
    xgb_meta  = evaluate_standard(y_test_np, xgb_pred, xgb_proba, 'XGBoost')
    print(pd.Series(xgb_meta).to_string())

    # ── Table 1: Model Comparison (report Section 4.2) ────────────────────────
    section('Table 1 — Model Comparison  (τ = 0.50, full prediction)')
    summary = pd.DataFrame([lr_meta, rf_meta, xgb_meta])
    print(summary.to_string(index=False))

    # ── Table 2: XGBoost Threshold Sweep (report Section 4.4) ─────────────────
    section('Table 2 — XGBoost Selective Prediction Threshold Sweep')
    xgb_sweep = threshold_sweep_table(
        y_test_np, xgb_proba,
        thresholds=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
    )
    print_sweep(xgb_sweep, 'XGBoost')

    # ── Table 3: Random Forest Threshold Sweep (report Section 4.4) ───────────
    section('Table 3 — Random Forest Selective Prediction Threshold Sweep')
    rf_sweep = threshold_sweep_table(
        y_test_np, rf_proba,
        thresholds=[0.50, 0.60, 0.70, 0.80, 0.90],
    )
    print_sweep(rf_sweep, 'Random Forest')

    # ── Selective Predictor at chosen τ ───────────────────────────────────────
    section(f'Selective Predictor at τ = {tau}')
    for name, model, proba in [
        ('Logistic Regression', lr,  lr_proba),
        ('Random Forest',       rf,  rf_proba),
        ('XGBoost',             xgb, xgb_proba),
    ]:
        sp = SelectivePredictor(model, threshold=tau)
        pred_sp, covered, _ = sp.predict_selective(X_test)
        acc_sp = (pred_sp[covered] == y_test_np[covered]).mean()
        print(f'\n{name}  τ={tau}')
        print(f'  Coverage : {covered.mean():.1%}  ({covered.sum():,} / {len(covered):,})')
        print(f'  Accuracy : {acc_sp:.4f}  (on covered samples)')
        print(f'  Abstained: {1 - covered.mean():.1%}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    section('Plots')
    for name, proba in [('Random Forest', rf_proba), ('XGBoost', xgb_proba)]:
        safe_name = name.replace(' ', '_').lower()
        plot_confidence_distribution(
            y_test_np, proba, model_name=name, tau=tau,
            save_path=f'{RESULTS_DIR}/{safe_name}_confidence.png',
        )

    cm_data = {
        'Logistic Regression': (y_test_np, lr_pred),
        'Random Forest':       (y_test_np, rf_pred),
        'XGBoost':             (y_test_np, xgb_pred),
    }
    plot_confusion_matrices(cm_data, save_path=f'{RESULTS_DIR}/confusion_matrices.png')

    print(f'\nAll outputs saved to {RESULTS_DIR}/')
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproduce report tables')
    parser.add_argument('--data', default='data/diabetic_data.csv')
    parser.add_argument('--tau',  type=float, default=0.70,
                        help='Confidence threshold for the final SP summary')
    args = parser.parse_args()
    main(args.data, args.tau)
