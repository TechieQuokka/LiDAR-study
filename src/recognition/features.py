"""Point Cloud 특징 추출

Phase 4: 객체 인식 - 특징 추출
- 기하학적 특징 (크기, 비율)
- FPFH (Fast Point Feature Histograms)
- 간단한 통계 특징
"""

import numpy as np
import open3d as o3d
from typing import Dict, Any


class FeatureExtractor:
    """Point Cloud 특징 추출기

    객체 인식을 위한 다양한 특징 추출 방법 제공
    """

    def __init__(self):
        self.fpfh_radius = 0.25  # FPFH 계산 반경
        self.fpfh_max_nn = 100   # FPFH 최대 이웃

    def extract_geometric_features(self, pcd: o3d.geometry.PointCloud) -> Dict[str, Any]:
        """기하학적 특징 추출

        크기, 비율 등 간단한 기하학적 특성

        Args:
            pcd: 입력 Point Cloud

        Returns:
            특징 딕셔너리
        """
        points = np.asarray(pcd.points)

        # Bounding Box
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()  # [width, height, depth]

        # 중심점
        centroid = points.mean(axis=0)

        # 분산
        variance = points.var(axis=0)

        # 주성분 분석 (PCA)
        centered = points - centroid
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # 내림차순

        features = {
            # 크기 관련
            "n_points": len(points),
            "width": extent[0],
            "height": extent[1],
            "depth": extent[2],
            "volume": np.prod(extent),
            "surface_area": 2 * (extent[0]*extent[1] + extent[1]*extent[2] + extent[0]*extent[2]),

            # 비율 관련
            "aspect_ratio_xy": extent[0] / max(extent[1], 1e-6),
            "aspect_ratio_xz": extent[0] / max(extent[2], 1e-6),
            "aspect_ratio_yz": extent[1] / max(extent[2], 1e-6),

            # 분포 관련
            "centroid": centroid.tolist(),
            "variance": variance.tolist(),

            # PCA 특징 (형상 분석)
            "linearity": (eigenvalues[0] - eigenvalues[1]) / (eigenvalues[0] + 1e-6),
            "planarity": (eigenvalues[1] - eigenvalues[2]) / (eigenvalues[0] + 1e-6),
            "sphericity": eigenvalues[2] / (eigenvalues[0] + 1e-6),
        }

        return features

    def extract_fpfh(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """FPFH (Fast Point Feature Histograms) 추출

        각 점의 국소 기하학적 특성을 33차원 히스토그램으로 표현

        Args:
            pcd: 입력 Point Cloud (법선 필요)

        Returns:
            FPFH 특징 (N x 33)
        """
        # 법선이 없으면 추정
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )

        # FPFH 계산
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.fpfh_radius,
                max_nn=self.fpfh_max_nn
            )
        )

        return np.asarray(fpfh.data).T  # (N, 33) 형태로 반환

    def extract_global_fpfh(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """전역 FPFH 특징 (평균)

        객체 전체를 대표하는 33차원 특징 벡터

        Args:
            pcd: 입력 Point Cloud

        Returns:
            33차원 특징 벡터
        """
        fpfh = self.extract_fpfh(pcd)
        return fpfh.mean(axis=0)  # 모든 점의 평균

    def extract_all_features(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """모든 특징을 결합한 특징 벡터

        기하학적 특징 + 전역 FPFH

        Args:
            pcd: 입력 Point Cloud

        Returns:
            결합된 특징 벡터
        """
        # 기하학적 특징
        geo_features = self.extract_geometric_features(pcd)
        geo_vector = np.array([
            geo_features["n_points"] / 1000,  # 정규화
            geo_features["width"],
            geo_features["height"],
            geo_features["depth"],
            geo_features["aspect_ratio_xy"],
            geo_features["aspect_ratio_xz"],
            geo_features["linearity"],
            geo_features["planarity"],
            geo_features["sphericity"],
        ])

        # FPFH 특징
        fpfh_vector = self.extract_global_fpfh(pcd)

        # 결합
        return np.concatenate([geo_vector, fpfh_vector])

    @staticmethod
    def normalize_point_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Point Cloud를 단위 구에 정규화

        딥러닝 입력용 정규화

        Args:
            pcd: 입력 Point Cloud

        Returns:
            정규화된 Point Cloud
        """
        points = np.asarray(pcd.points)

        # 중심을 원점으로
        centroid = points.mean(axis=0)
        points_centered = points - centroid

        # 최대 거리로 스케일링
        max_dist = np.max(np.linalg.norm(points_centered, axis=1))
        points_normalized = points_centered / max_dist

        result = o3d.geometry.PointCloud()
        result.points = o3d.utility.Vector3dVector(points_normalized)

        if pcd.has_colors():
            result.colors = pcd.colors
        if pcd.has_normals():
            result.normals = pcd.normals

        return result

    @staticmethod
    def sample_points(pcd: o3d.geometry.PointCloud, n_points: int = 1024) -> np.ndarray:
        """고정 개수로 점 샘플링

        딥러닝 입력용 (PointNet 등)

        Args:
            pcd: 입력 Point Cloud
            n_points: 목표 점 개수

        Returns:
            (n_points, 3) 형태의 점 배열
        """
        points = np.asarray(pcd.points)
        n_current = len(points)

        if n_current >= n_points:
            # 랜덤 샘플링
            indices = np.random.choice(n_current, n_points, replace=False)
            return points[indices]
        else:
            # 부족하면 복제
            indices = np.random.choice(n_current, n_points, replace=True)
            return points[indices]
