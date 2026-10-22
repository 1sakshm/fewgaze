"""Downstream classifier. Every learned transform is inside one sklearn Pipeline that is fit on
the training selection only, so scaling statistics never see held-out subjects."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def build_classifier(clf_cfg: dict[str, Any], polynomial_degree: int = 1, seed: int = 0) -> Pipeline:
    steps: list[Any] = [StandardScaler()]
    if polynomial_degree > 1:
        steps += [PolynomialFeatures(degree=polynomial_degree, include_bias=False), StandardScaler()]
    steps.append(
        LogisticRegression(
            C=float(clf_cfg.get("C", 1.0)),
            class_weight=clf_cfg.get("class_weight", "balanced"),
            max_iter=int(clf_cfg.get("max_iter", 5000)),
            random_state=seed,
        )
    )
    return make_pipeline(*steps)


def fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    clf_cfg: dict[str, Any],
    polynomial_degree: int = 1,
    seed: int = 0,
) -> tuple[np.ndarray, Pipeline, float, float]:
    """Return P(camera-directed) for X_test, the fitted pipeline, fit seconds and predict seconds."""
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training labels contain a single class; cannot fit a classifier")
    model = build_classifier(clf_cfg, polynomial_degree, seed)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    t1 = time.perf_counter()
    score = model.predict_proba(X_test)[:, 1]
    return score, model, t1 - t0, time.perf_counter() - t1
