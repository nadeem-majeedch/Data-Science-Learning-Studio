"""
Evaluation utilities for Data Science Lab.

Provides reusable functions for computing metrics, cross-validation
scores, and generating Plotly figures for model evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import cross_validate


# ── Data classes ────────────────────────────────────────────────────


@dataclass
class ClassificationMetrics:
    """Structured classification evaluation results."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    report_text: str
    confusion: np.ndarray
    classes: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray | None  # probability estimates for ROC


@dataclass
class RegressionMetrics:
    """Structured regression evaluation results."""

    r2: float
    mae: float
    mse: float
    rmse: float
    residuals: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray


@dataclass
class CrossValResult:
    """Cross-validation summary."""

    scores: dict[str, np.ndarray]
    means: dict[str, float]
    stds: dict[str, float]


# ═════════════════════════════════════════════════════════════════════
#  CLASSIFICATION METRICS
# ═════════════════════════════════════════════════════════════════════


def compute_classification_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Compute all classification metrics."""
    n_classes = len(np.unique(y_test))
    avg = "binary" if n_classes == 2 else "weighted"

    acc = round(accuracy_score(y_test, y_pred), 4)
    prec = round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4)
    rec = round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4)
    f1 = round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    classes = np.array(sorted(np.unique(y_test)))

    # ROC AUC
    roc_auc_val = None
    if y_prob is not None:
        try:
            if n_classes == 2 and y_prob.ndim == 1:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc_val = round(auc(fpr, tpr), 4)
            elif n_classes > 2 and y_prob.ndim == 2:
                from sklearn.metrics import roc_auc_score
                roc_auc_val = round(roc_auc_score(y_test, y_prob, multi_class="ovr"), 4)
        except Exception:
            pass

    return ClassificationMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc_auc_val,
        report_text=report,
        confusion=cm,
        classes=classes,
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
    )


# ═════════════════════════════════════════════════════════════════════
#  REGRESSION METRICS
# ═════════════════════════════════════════════════════════════════════


def compute_regression_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> RegressionMetrics:
    """Compute all regression metrics."""
    return RegressionMetrics(
        r2=round(r2_score(y_test, y_pred), 4),
        mae=round(mean_absolute_error(y_test, y_pred), 4),
        mse=round(mean_squared_error(y_test, y_pred), 4),
        rmse=round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        residuals=y_test - y_pred,
        y_test=y_test,
        y_pred=y_pred,
    )


# ═════════════════════════════════════════════════════════════════════
#  CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════


def run_cross_validation(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task: str = "classification",
    cv: int = 5,
    scoring: str | list[str] | None = None,
    random_state: int = 42,
) -> CrossValResult:
    """Run k-fold cross-validation on a Pipeline.

    Parameters
    ----------
    task : ``"classification"`` or ``"regression"``
    cv : number of folds
    scoring : sklearn scoring string(s); uses sensible defaults per task
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    if scoring is None:
        if task == "classification":
            scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
        else:
            scoring = ["r2", "neg_mean_absolute_error", "neg_mean_squared_error"]

    results = cross_validate(pipeline, X, y, cv=kf, scoring=scoring, error_score="raise")

    scores = {}
    means = {}
    stds = {}
    for key in scoring:
        test_key = f"test_{key}"
        if test_key in results:
            vals = results[test_key]
            scores[key] = vals
            means[key] = round(float(vals.mean()), 4)
            stds[key] = round(float(vals.std()), 4)

    return CrossValResult(scores=scores, means=means, stds=stds)


# ═════════════════════════════════════════════════════════════════════
#  PLOTLY FIGURES
# ═════════════════════════════════════════════════════════════════════


def fig_confusion_matrix(cm: np.ndarray, classes: np.ndarray) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[str(c) for c in classes],
            y=[str(c) for c in classes],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 14},
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_white",
        width=500,
        height=450,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def fig_roc_curve(
    y_test: np.ndarray,
    y_prob: np.ndarray,
) -> go.Figure:
    """ROC curve for binary classification."""
    n_classes = len(np.unique(y_test))
    fig = go.Figure()

    if n_classes == 2 and y_prob.ndim == 1:
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"ROC (AUC = {roc_auc_val:.3f})",
            line=dict(width=2),
        ))
    elif n_classes > 2 and y_prob.ndim == 2:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_test, classes=np.unique(y_test))
        for i in range(y_bin.shape[1]):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc_val = auc(fpr, tpr)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"Class {classes[i]} (AUC = {roc_auc_val:.3f})",
            ))
    else:
        fig.update_layout(title="ROC not available (need probability predictions)")

    # Diagonal baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name="Random baseline",
        line=dict(dash="dash", color="gray"),
    ))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        width=600,
        height=500,
    )
    return fig


def fig_residual_plot(y_test: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Residual scatter plot with zero line."""
    residuals = y_test - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode="markers",
        marker=dict(opacity=0.6),
        name="Residuals",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title="Residual Plot",
        xaxis_title="Predicted",
        yaxis_title="Residual (Actual − Predicted)",
        template="plotly_white",
    )
    return fig


def fig_cross_val_bars(means: dict[str, float], stds: dict[str, float]) -> go.Figure:
    """Bar chart of cross-validation scores with error bars."""
    labels = list(means.keys())
    vals = [means[k] for k in labels]
    errs = [stds[k] for k in labels]

    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        error_y=dict(type="data", array=errs, visible=True),
        marker_color="#636EFA",
    ))
    fig.update_layout(
        title="Cross-Validation Scores (mean ± std)",
        yaxis_title="Score",
        template="plotly_white",
    )
    return fig
