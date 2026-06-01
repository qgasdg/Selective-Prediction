"""
Evaluation metrics and visualisations for selective prediction.

Standard metrics : Accuracy, Precision, Recall, F1 (binary & macro), ROC-AUC
SP-specific      : Coverage-Accuracy curve, threshold sweep table, abstention analysis
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix, RocCurveDisplay,
)

# ── Standard classification metrics ─────────────────────────────────────────

def evaluate_standard(y_true, y_pred, y_proba=None, name: str = 'Model') -> dict:
    """Return a dict of standard binary classification metrics."""
    m = {
        'Model':       name,
        'Accuracy':    accuracy_score(y_true, y_pred),
        'Precision':   precision_score(y_true, y_pred, zero_division=0),
        'Recall':      recall_score(y_true, y_pred, zero_division=0),
        'F1 (binary)': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'F1 (macro)':  f1_score(y_true, y_pred, average='macro',  zero_division=0),
    }
    if y_proba is not None:
        m['ROC-AUC'] = roc_auc_score(y_true, y_proba[:, 1])
    return m


# ── Selective Prediction metrics ─────────────────────────────────────────────

def coverage_accuracy_curve(y_true: np.ndarray,
                             proba:  np.ndarray,
                             n_points: int = 60):
    """
    Sweep confidence threshold from 0.50 to 0.99 and record
    (threshold, coverage, accuracy_on_covered) at each step.

    Returns
    -------
    thresholds : (n,) array
    coverages  : (n,) array   – fraction of samples predicted
    accuracies : (n,) array   – accuracy among predicted samples
    """
    thresholds  = np.linspace(0.50, 0.99, n_points)
    confidence  = proba.max(axis=1)
    pred        = proba.argmax(axis=1)

    coverages, accuracies = [], []
    valid_tau = []
    for tau in thresholds:
        mask = confidence >= tau
        if mask.sum() == 0:
            break
        valid_tau.append(tau)
        coverages.append(mask.mean())
        accuracies.append(accuracy_score(y_true[mask], pred[mask]))

    return np.array(valid_tau), np.array(coverages), np.array(accuracies)


def threshold_sweep_table(y_true: np.ndarray,
                           proba:  np.ndarray,
                           thresholds=None) -> pd.DataFrame:
    """
    Return a DataFrame summarising selective prediction performance
    at each τ value.

    Columns: τ, Coverage, Accuracy, F1 (binary), Recall, Abstained %
    """
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    confidence = proba.max(axis=1)
    pred       = proba.argmax(axis=1)
    rows = []
    for tau in thresholds:
        mask = confidence >= tau
        n    = mask.sum()
        if n == 0:
            continue
        rows.append({
            'τ':           tau,
            'Coverage':    mask.mean(),
            'Accuracy':    accuracy_score(y_true[mask], pred[mask]),
            'F1 (binary)': f1_score(y_true[mask], pred[mask],
                                    average='binary', zero_division=0),
            'Recall':      recall_score(y_true[mask], pred[mask], zero_division=0),
            'Abstained':   1 - mask.mean(),
            'n_predicted': int(n),
        })
    return pd.DataFrame(rows)


def print_sweep(df: pd.DataFrame, model_name: str = ''):
    fmt = {
        'τ': '{:.2f}', 'Coverage': '{:.1%}', 'Accuracy': '{:.4f}',
        'F1 (binary)': '{:.4f}', 'Recall': '{:.4f}', 'Abstained': '{:.1%}',
    }
    header = f"  Selective Prediction Sweep — {model_name}  "
    print(f"\n{'='*len(header)}\n{header}\n{'='*len(header)}")
    print(df.to_string(index=False, float_format='{:.4f}'.format))


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_coverage_accuracy(curves: dict, save_path: str):
    """
    Coverage-Accuracy Trade-off curve for multiple models.

    Parameters
    ----------
    curves : {model_name: (thresholds, coverages, accuracies)}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors

    for i, (name, (taus, covs, accs)) in enumerate(curves.items()):
        c = colors[i % len(colors)]
        axes[0].plot(covs, accs, 'o-', ms=3, color=c, label=name)
        axes[1].plot(taus, accs, 'o-', ms=3, color=c, label=name)

    axes[0].set_xlabel('Coverage  (예측한 비율)')
    axes[0].set_ylabel('Accuracy  (예측된 샘플 기준)')
    axes[0].set_title('Coverage – Accuracy Trade-off')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Confidence Threshold  τ')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Threshold vs Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")


def plot_confidence_distribution(y_true: np.ndarray,
                                  proba:  np.ndarray,
                                  model_name: str,
                                  tau: float = 0.70,
                                  save_path: str | None = None):
    """Histogram of confidence scores split by true class label."""
    confidence = proba.max(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))

    for label, lname, color in [(0, '재입원 없음 (Y=0)', '#4c9be8'),
                                  (1, '30일 내 재입원 (Y=1)', '#e87c4c')]:
        ax.hist(confidence[y_true == label], bins=50, alpha=0.6,
                density=True, label=lname, color=color)

    ax.axvline(tau, color='red', linestyle='--', linewidth=1.5,
               label=f'τ = {tau}')
    ax.set_xlabel('Confidence  max P(Y|x)')
    ax.set_ylabel('Density')
    ax.set_title(f'{model_name}: Confidence Distribution by Class')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved → {save_path}")
    else:
        plt.show()


def plot_roc_curves(roc_data: dict, save_path: str):
    """
    Parameters
    ----------
    roc_data : {model_name: (y_true, y_score)}
    """
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10.colors

    for i, (name, (y_true, y_score)) in enumerate(roc_data.items()):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                label=f'{name}  (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")


def plot_confusion_matrices(cm_data: dict, save_path: str):
    """
    Parameters
    ----------
    cm_data : {model_name: (y_true, y_pred)}
    """
    n = len(cm_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, (y_true, y_pred)) in zip(axes, cm_data.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['재입원 없음', '30일 내 재입원'],
                    yticklabels=['재입원 없음', '30일 내 재입원'])
        ax.set_title(name)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")


def plot_feature_importance(importances: pd.Series,
                             model_name: str,
                             top_n: int = 20,
                             save_path: str | None = None):
    top = importances.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1))
    top.plot(kind='barh', ax=ax, color='#4c9be8')
    ax.set_title(f'{model_name}: Top {top_n} Feature Importances')
    ax.set_xlabel('Importance')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved → {save_path}")
    else:
        plt.show()
