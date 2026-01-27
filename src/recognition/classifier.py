"""Point Cloud 분류기

Phase 4: 객체 인식 - 분류
- 전통적 분류기 (SVM, Random Forest)
- 간단한 규칙 기반 분류
"""

import numpy as np
import open3d as o3d
from typing import List, Dict, Tuple, Optional
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

from .features import FeatureExtractor


class PointCloudClassifier:
    """Point Cloud 분류기

    특징 추출 + 분류기를 결합한 파이프라인
    """

    def __init__(self, classifier_type: str = "svm"):
        """
        Args:
            classifier_type: 'svm' 또는 'rf' (Random Forest)
        """
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.classifier_type = classifier_type

        if classifier_type == "svm":
            self.classifier = SVC(kernel='rbf', probability=True)
        elif classifier_type == "rf":
            self.classifier = RandomForestClassifier(n_estimators=100)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        self.classes: List[str] = []
        self.is_trained = False

    def extract_features(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """Point Cloud에서 특징 추출"""
        return self.feature_extractor.extract_all_features(pcd)

    def train(self,
              point_clouds: List[o3d.geometry.PointCloud],
              labels: List[str]) -> Dict[str, float]:
        """분류기 학습

        Args:
            point_clouds: 학습 데이터 (Point Cloud 리스트)
            labels: 각 Point Cloud의 클래스 레이블

        Returns:
            학습 결과 메트릭
        """
        print(f"학습 시작: {len(point_clouds)}개 샘플")

        # 특징 추출
        features = []
        for i, pcd in enumerate(point_clouds):
            feat = self.extract_features(pcd)
            features.append(feat)
            if (i + 1) % 10 == 0:
                print(f"  특징 추출: {i + 1}/{len(point_clouds)}")

        X = np.array(features)
        y = np.array(labels)

        # 클래스 목록 저장
        self.classes = list(set(labels))

        # 특징 정규화
        X_scaled = self.scaler.fit_transform(X)

        # 분류기 학습
        self.classifier.fit(X_scaled, y)
        self.is_trained = True

        # 학습 정확도
        train_score = self.classifier.score(X_scaled, y)

        print(f"학습 완료: 정확도 {train_score:.2%}")

        return {
            "train_accuracy": train_score,
            "n_samples": len(point_clouds),
            "n_classes": len(self.classes),
            "classes": self.classes
        }

    def predict(self, pcd: o3d.geometry.PointCloud) -> Tuple[str, Dict[str, float]]:
        """단일 Point Cloud 분류

        Args:
            pcd: 분류할 Point Cloud

        Returns:
            (예측 클래스, 클래스별 확률)
        """
        if not self.is_trained:
            raise RuntimeError("분류기가 학습되지 않았습니다.")

        # 특징 추출
        features = self.extract_features(pcd)
        features_scaled = self.scaler.transform([features])

        # 예측
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]

        prob_dict = {cls: prob for cls, prob in zip(self.classifier.classes_, probabilities)}

        return prediction, prob_dict

    def predict_batch(self,
                      point_clouds: List[o3d.geometry.PointCloud]) -> List[Tuple[str, Dict]]:
        """여러 Point Cloud 배치 분류"""
        results = []
        for pcd in point_clouds:
            pred, probs = self.predict(pcd)
            results.append((pred, probs))
        return results

    def save(self, filepath: str) -> None:
        """모델 저장"""
        if not self.is_trained:
            raise RuntimeError("학습된 모델이 없습니다.")

        model_data = {
            "classifier": self.classifier,
            "scaler": self.scaler,
            "classes": self.classes,
            "classifier_type": self.classifier_type
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"모델 저장: {filepath}")

    def load(self, filepath: str) -> None:
        """모델 로드"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.classifier = model_data["classifier"]
        self.scaler = model_data["scaler"]
        self.classes = model_data["classes"]
        self.classifier_type = model_data["classifier_type"]
        self.is_trained = True

        print(f"모델 로드: {filepath}")
        print(f"  클래스: {self.classes}")


class SimpleRuleClassifier:
    """규칙 기반 간단 분류기

    학습 없이 기하학적 특성으로 분류
    학습용 베이스라인으로 사용
    """

    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    def classify(self, pcd: o3d.geometry.PointCloud) -> Tuple[str, Dict[str, float]]:
        """규칙 기반 분류

        기하학적 특성을 기반으로 간단한 규칙 적용

        Returns:
            (예측 클래스, 신뢰도 점수)
        """
        features = self.feature_extractor.extract_geometric_features(pcd)

        # 규칙 정의
        linearity = features["linearity"]
        planarity = features["planarity"]
        sphericity = features["sphericity"]
        aspect_xy = features["aspect_ratio_xy"]

        scores = {
            "plane": 0.0,
            "sphere": 0.0,
            "box": 0.0,
            "cylinder": 0.0,
            "unknown": 0.1
        }

        # 평면성이 높으면 plane
        if planarity > 0.6:
            scores["plane"] = planarity

        # 구형성이 높으면 sphere
        if sphericity > 0.5:
            scores["sphere"] = sphericity

        # 비율이 비슷하고 평면성/구형성이 낮으면 box
        if 0.7 < aspect_xy < 1.3 and planarity < 0.4 and sphericity < 0.4:
            scores["box"] = 1 - max(planarity, sphericity)

        # 길쭉하면 cylinder
        if linearity > 0.5:
            scores["cylinder"] = linearity

        # 최대 점수 클래스 선택
        predicted = max(scores, key=scores.get)
        confidence = scores[predicted]

        return predicted, scores

    def classify_batch(self,
                       point_clouds: List[o3d.geometry.PointCloud]) -> List[Tuple[str, Dict]]:
        """배치 분류"""
        return [self.classify(pcd) for pcd in point_clouds]
