from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURES = [
    "area_m2",
    "district",
    "house_type",
    "legal_status",
    "main_door_direction",
    "balcony_direction",
    "bedrooms",
    "toilets",
    "frontage_m",
    "alley_width_m",
]
TARGET = "price_billion"

NUMERIC_FEATURES = ["area_m2", "bedrooms", "toilets", "frontage_m", "alley_width_m"]
CATEGORICAL_FEATURES = [
    "district",
    "house_type",
    "legal_status",
    "main_door_direction",
    "balcony_direction",
]


@dataclass
class TrainedArtifacts:
    clustering_pipeline: Pipeline
    pricing_pipeline: Pipeline
    feature_names: List[str]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def train_models(df: pd.DataFrame, n_clusters: int = 4) -> TrainedArtifacts:
    x = df[FEATURES].copy()
    y = df[TARGET].copy()

    cluster_pipe = Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            (
                "kmeans",
                KMeans(n_clusters=n_clusters, n_init=20, random_state=42),
            ),
        ]
    )
    cluster_pipe.fit(x)

    reg_pipe = Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )
    reg_pipe.fit(x, y)

    return TrainedArtifacts(
        clustering_pipeline=cluster_pipe,
        pricing_pipeline=reg_pipe,
        feature_names=FEATURES,
    )


def predict_cluster(artifacts: TrainedArtifacts, row: pd.DataFrame) -> int:
    return int(artifacts.clustering_pipeline.predict(row)[0])


def predict_price(artifacts: TrainedArtifacts, row: pd.DataFrame) -> float:
    return float(artifacts.pricing_pipeline.predict(row)[0])
