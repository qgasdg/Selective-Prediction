"""
Model definitions and hyperparameter tuning.

Models
------
- Logistic Regression (baseline)
- Random Forest        (tuned via RandomizedSearchCV)
- XGBoost              (tuned via RandomizedSearchCV)

SelectivePredictor
------------------
A thin wrapper that adds selective-prediction (abstention) behaviour to any
sklearn-compatible classifier.  The model predicts only when
  max(P(Y=0|x), P(Y=1|x)) >= threshold τ,
and returns a "refused" flag for the rest.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


# ── Selective Prediction wrapper ─────────────────────────────────────────────

class SelectivePredictor:
    """
    Wraps any sklearn-style binary classifier with selective prediction.

    Parameters
    ----------
    model     : fitted sklearn estimator with predict_proba
    threshold : confidence threshold τ ∈ [0.5, 1.0]
                samples with max(p) < τ are abstained
    """

    def __init__(self, model, threshold: float = 0.7):
        self.model     = model
        self.threshold = threshold

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict_selective(self, X):
        """
        Returns
        -------
        pred      : np.ndarray[int]   – predicted class (all rows)
        covered   : np.ndarray[bool]  – True when confidence >= threshold
        confidence: np.ndarray[float] – max(P(Y=0|x), P(Y=1|x))
        """
        proba      = self.predict_proba(X)
        confidence = proba.max(axis=1)
        pred       = proba.argmax(axis=1)
        covered    = confidence >= self.threshold
        return pred, covered, confidence

    def set_threshold(self, tau: float):
        self.threshold = tau
        return self


# ── Model factories ──────────────────────────────────────────────────────────

def get_baseline(random_state: int = 42) -> LogisticRegression:
    """Logistic Regression with balanced class weights (baseline)."""
    return LogisticRegression(
        class_weight='balanced',
        max_iter=2000,
        solver='lbfgs',
        random_state=random_state,
    )


def get_random_forest(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        class_weight='balanced',
        n_jobs=-1,
        random_state=random_state,
    )


def get_xgboost(scale_pos_weight: float = 1.0,
                random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1,
        random_state=random_state,
        verbosity=0,
    )


# ── Hyperparameter search ────────────────────────────────────────────────────

_RF_PARAM_DIST = {
    'n_estimators':    [100, 200, 300, 500],
    'max_depth':       [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':    ['sqrt', 'log2'],
}

_XGB_PARAM_DIST = {
    'n_estimators':     [100, 200, 300, 500],
    'max_depth':        [3, 4, 5, 6, 7],
    'learning_rate':    [0.01, 0.05, 0.1, 0.2],
    'subsample':        [0.6, 0.7, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma':            [0, 0.1, 0.2],
}


def tune_random_forest(X_train, y_train,
                       n_iter: int = 30,
                       cv: int = 3,
                       random_state: int = 42):
    """
    RandomizedSearchCV over Random Forest.
    Optimises ROC-AUC (appropriate for imbalanced classes).

    Returns
    -------
    best_estimator : fitted RandomForestClassifier
    best_params    : dict
    cv_results     : pd.DataFrame
    """
    rf     = get_random_forest(random_state)
    search = RandomizedSearchCV(
        rf, _RF_PARAM_DIST,
        n_iter=n_iter, cv=cv,
        scoring='roc_auc',
        refit=True,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )
    search.fit(X_train, y_train)

    import pandas as pd
    cv_df = pd.DataFrame(search.cv_results_).sort_values(
        'mean_test_score', ascending=False
    )
    return search.best_estimator_, search.best_params_, cv_df


def tune_xgboost(X_train, y_train,
                 scale_pos_weight: float = 1.0,
                 n_iter: int = 30,
                 cv: int = 3,
                 random_state: int = 42):
    """
    RandomizedSearchCV over XGBoost.

    Returns
    -------
    best_estimator : fitted XGBClassifier
    best_params    : dict
    cv_results     : pd.DataFrame
    """
    xgb    = get_xgboost(scale_pos_weight, random_state)
    search = RandomizedSearchCV(
        xgb, _XGB_PARAM_DIST,
        n_iter=n_iter, cv=cv,
        scoring='roc_auc',
        refit=True,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )
    search.fit(X_train, y_train)

    import pandas as pd
    cv_df = pd.DataFrame(search.cv_results_).sort_values(
        'mean_test_score', ascending=False
    )
    return search.best_estimator_, search.best_params_, cv_df
