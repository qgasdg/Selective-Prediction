"""
Main training pipeline.

Usage
-----
  python main.py [--data DATA_PATH] [--n_iter N] [--tau TAU]

  --data    path to diabetic_data.csv  (default: data/diabetic_data.csv)
  --n_iter  RandomizedSearchCV iterations per model (default: 30)
  --tau     confidence threshold for the Selective Predictor (default: 0.70)
"""

import os
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from src.preprocess import load_and_clean, engineer_features, split_data
from src.models import (
    SelectivePredictor,
    get_baseline,
    tune_random_forest,
    tune_xgboost,
)
from src.evaluate import (
    evaluate_standard,
    coverage_accuracy_curve,
    threshold_sweep_table,
    print_sweep,
    plot_coverage_accuracy,
    plot_confidence_distribution,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_feature_importance,
)
from src.eda import run_eda

RESULTS_DIR = 'results'


# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str):
    bar = '=' * 60
    print(f'\n{bar}\n  {title}\n{bar}')


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_path: str, n_iter: int, tau: float):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 0. EDA ───────────────────────────────────────────────────────────────
    section('0. EDA')
    df_raw = load_and_clean(data_path)
    run_eda(df_raw, out_dir=f'{RESULTS_DIR}/eda')

    # ── 1. Feature engineering & split ───────────────────────────────────────
    section('1. Feature Engineering & Split')
    df = engineer_features(df_raw)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Class imbalance ratio for XGBoost scale_pos_weight
    neg, pos      = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_wt  = neg / pos
    print(f'\nNeg/Pos ratio: {scale_pos_wt:.2f}  →  used as XGB scale_pos_weight')

    # ── 2. Baseline: Logistic Regression ─────────────────────────────────────
    section('2. Baseline — Logistic Regression')
    lr = get_baseline()
    lr.fit(X_train, y_train)

    lr_proba = lr.predict_proba(X_test)
    lr_pred  = lr_proba.argmax(axis=1)
    lr_meta  = evaluate_standard(y_test.values, lr_pred, lr_proba, 'Logistic Regression')
    print(pd.Series(lr_meta))

    lr_sweep = threshold_sweep_table(y_test.values, lr_proba)
    print_sweep(lr_sweep, 'Logistic Regression')

    # ── 3. Random Forest (tuned) ──────────────────────────────────────────────
    section('3. Random Forest — Hyperparameter Tuning')
    rf_best, rf_params, rf_cv = tune_random_forest(
        X_train, y_train, n_iter=n_iter, cv=3
    )
    print(f'\nBest params: {rf_params}')

    rf_proba = rf_best.predict_proba(X_test)
    rf_pred  = rf_proba.argmax(axis=1)
    rf_meta  = evaluate_standard(y_test.values, rf_pred, rf_proba, 'Random Forest')
    print(pd.Series(rf_meta))

    rf_sweep = threshold_sweep_table(y_test.values, rf_proba)
    print_sweep(rf_sweep, 'Random Forest')

    # Feature importance
    rf_imp = pd.Series(rf_best.feature_importances_, index=X_train.columns)
    plot_feature_importance(
        rf_imp, 'Random Forest', top_n=20,
        save_path=f'{RESULTS_DIR}/rf_feature_importance.png'
    )
    rf_imp.nlargest(30).to_csv(f'{RESULTS_DIR}/rf_feature_importance.csv',
                                header=['importance'])

    # ── 4. XGBoost (tuned) ────────────────────────────────────────────────────
    section('4. XGBoost — Hyperparameter Tuning')
    xgb_best, xgb_params, xgb_cv = tune_xgboost(
        X_train, y_train,
        scale_pos_weight=scale_pos_wt,
        n_iter=n_iter, cv=3
    )
    print(f'\nBest params: {xgb_params}')

    xgb_proba = xgb_best.predict_proba(X_test)
    xgb_pred  = xgb_proba.argmax(axis=1)
    xgb_meta  = evaluate_standard(y_test.values, xgb_pred, xgb_proba, 'XGBoost')
    print(pd.Series(xgb_meta))

    xgb_sweep = threshold_sweep_table(y_test.values, xgb_proba)
    print_sweep(xgb_sweep, 'XGBoost')

    xgb_imp = pd.Series(xgb_best.feature_importances_, index=X_train.columns)
    plot_feature_importance(
        xgb_imp, 'XGBoost', top_n=20,
        save_path=f'{RESULTS_DIR}/xgb_feature_importance.png'
    )

    # ── 5. Selective Prediction Analysis ─────────────────────────────────────
    section('5. Selective Prediction Analysis')

    # Coverage-Accuracy curves
    curves = {}
    for name, proba in [('Logistic Regression', lr_proba),
                         ('Random Forest',       rf_proba),
                         ('XGBoost',             xgb_proba)]:
        taus, covs, accs = coverage_accuracy_curve(y_test.values, proba)
        curves[name] = (taus, covs, accs)

    plot_coverage_accuracy(curves, save_path=f'{RESULTS_DIR}/coverage_accuracy.png')

    # Confidence distributions
    for name, proba in [('Random Forest', rf_proba), ('XGBoost', xgb_proba)]:
        safe_name = name.replace(' ', '_').lower()
        plot_confidence_distribution(
            y_test.values, proba, model_name=name, tau=tau,
            save_path=f'{RESULTS_DIR}/{safe_name}_confidence.png'
        )

    # Selective predictor at chosen τ
    section(f'5b. Selective Predictor at τ = {tau}')
    for name, model, proba in [
        ('Logistic Regression', lr,      lr_proba),
        ('Random Forest',       rf_best, rf_proba),
        ('XGBoost',             xgb_best, xgb_proba),
    ]:
        sp = SelectivePredictor(model, threshold=tau)
        pred_sp, covered, conf = sp.predict_selective(X_test)
        acc_sp  = (pred_sp[covered] == y_test.values[covered]).mean()
        print(f'\n{name}  τ={tau}')
        print(f'  Coverage : {covered.mean():.1%}  ({covered.sum():,} / {len(covered):,})')
        print(f'  Accuracy : {acc_sp:.4f}  (on covered samples)')
        print(f'  Abstained: {1 - covered.mean():.1%}')

    # ── 6. Summary tables ─────────────────────────────────────────────────────
    section('6. Summary')
    all_meta = [lr_meta, rf_meta, xgb_meta]
    summary_df = pd.DataFrame(all_meta)
    print('\n' + summary_df.to_string(index=False))
    summary_df.to_csv(f'{RESULTS_DIR}/model_comparison.csv', index=False)

    # ROC curves
    roc_data = {
        'Logistic Regression': (y_test.values, lr_proba[:, 1]),
        'Random Forest':       (y_test.values, rf_proba[:, 1]),
        'XGBoost':             (y_test.values, xgb_proba[:, 1]),
    }
    plot_roc_curves(roc_data, save_path=f'{RESULTS_DIR}/roc_curves.png')

    # Confusion matrices
    cm_data = {
        'Logistic Regression': (y_test.values, lr_pred),
        'Random Forest':       (y_test.values, rf_pred),
        'XGBoost':             (y_test.values, xgb_pred),
    }
    plot_confusion_matrices(cm_data, save_path=f'{RESULTS_DIR}/confusion_matrices.png')

    # Threshold sweep tables
    for name, sweep in [('logistic_regression', lr_sweep),
                         ('random_forest',       rf_sweep),
                         ('xgboost',             xgb_sweep)]:
        sweep.to_csv(f'{RESULTS_DIR}/{name}_threshold_sweep.csv', index=False)

    # Best hyperparameters
    save_json({'random_forest': rf_params, 'xgboost': xgb_params},
              f'{RESULTS_DIR}/best_hyperparams.json')

    print(f'\nAll outputs saved to  {RESULTS_DIR}/')
    print('Done.')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Term Project — Selective Prediction')
    parser.add_argument('--data',   default='data/diabetic_data.csv',
                        help='Path to diabetic_data.csv')
    parser.add_argument('--n_iter', type=int, default=30,
                        help='RandomizedSearchCV iterations per model')
    parser.add_argument('--tau',    type=float, default=0.70,
                        help='Confidence threshold for Selective Predictor')
    args = parser.parse_args()
    main(args.data, args.n_iter, args.tau)
