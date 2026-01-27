"""Point Cloud 세그멘테이션

Phase 3: 세그멘테이션
- RANSAC 평면 추출
- DBSCAN 클러스터링
- Region Growing (영역 성장)
"""

import numpy as np
import open3d as o3d
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Plane:
    """추출된 평면 정보"""
    points: o3d.geometry.PointCloud  # 평면에 속한 점들
    equation: np.ndarray              # [a, b, c, d] where ax + by + cz + d = 0
    inlier_indices: np.ndarray        # 원본에서의 인덱스


@dataclass
class Cluster:
    """클러스터 정보"""
    points: o3d.geometry.PointCloud
    indices: np.ndarray
    label: int
    centroid: np.ndarray
    bbox: o3d.geometry.AxisAlignedBoundingBox


class Segmenter:
    """Point Cloud 세그멘테이션 클래스"""

    def __init__(self):
        # RANSAC 파라미터
        self.ransac_distance = 0.02   # 평면까지 최대 거리
        self.ransac_n = 3             # 평면 피팅에 사용할 점 수
        self.ransac_iterations = 1000  # 반복 횟수

        # 클러스터링 파라미터
        self.cluster_eps = 0.1         # DBSCAN epsilon (최대 이웃 거리)
        self.cluster_min_points = 10   # 최소 클러스터 크기

    def extract_plane(self,
                      pcd: o3d.geometry.PointCloud,
                      distance_threshold: float = None,
                      num_iterations: int = None) -> Tuple[Plane, o3d.geometry.PointCloud]:
        """RANSAC으로 평면 추출

        가장 많은 점이 속한 평면을 찾아 추출

        Args:
            pcd: 입력 Point Cloud
            distance_threshold: 평면에서 점까지 최대 거리
            num_iterations: RANSAC 반복 횟수

        Returns:
            (추출된 평면, 나머지 점들)
        """
        distance_threshold = distance_threshold or self.ransac_distance
        num_iterations = num_iterations or self.ransac_iterations

        # RANSAC 평면 피팅
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=num_iterations
        )

        # 평면 점들 추출
        plane_pcd = pcd.select_by_index(inliers)
        plane_pcd.paint_uniform_color([0.5, 0.5, 0.5])

        # 나머지 점들
        remaining_pcd = pcd.select_by_index(inliers, invert=True)

        plane = Plane(
            points=plane_pcd,
            equation=np.array(plane_model),
            inlier_indices=np.array(inliers)
        )

        a, b, c, d = plane_model
        print(f"평면 추출: {len(inliers):,}개 점")
        print(f"  방정식: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

        return plane, remaining_pcd

    def extract_multiple_planes(self,
                                pcd: o3d.geometry.PointCloud,
                                max_planes: int = 3,
                                min_points: int = 100) -> Tuple[List[Plane], o3d.geometry.PointCloud]:
        """여러 평면 순차 추출

        Args:
            pcd: 입력 Point Cloud
            max_planes: 최대 추출할 평면 수
            min_points: 평면당 최소 점 수

        Returns:
            (평면 리스트, 나머지 점들)
        """
        planes = []
        remaining = pcd

        for i in range(max_planes):
            if len(remaining.points) < min_points:
                break

            plane, remaining = self.extract_plane(remaining)

            if len(plane.points.points) < min_points:
                # 평면이 너무 작으면 중단
                remaining = remaining + plane.points
                break

            planes.append(plane)

        print(f"총 {len(planes)}개 평면 추출, 나머지 {len(remaining.points):,}개 점")
        return planes, remaining

    def cluster_dbscan(self,
                       pcd: o3d.geometry.PointCloud,
                       eps: float = None,
                       min_points: int = None) -> List[Cluster]:
        """DBSCAN 클러스터링

        밀도 기반 클러스터링으로 개별 객체 분리

        Args:
            pcd: 입력 Point Cloud
            eps: 이웃 판정 거리
            min_points: 클러스터 최소 점 수

        Returns:
            클러스터 리스트
        """
        eps = eps or self.cluster_eps
        min_points = min_points or self.cluster_min_points

        # DBSCAN 실행
        labels = np.array(pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points
        ))

        # 고유 레이블 (-1은 노이즈)
        unique_labels = set(labels)
        unique_labels.discard(-1)

        print(f"DBSCAN 클러스터링: {len(unique_labels)}개 클러스터 발견")

        # 각 클러스터 생성
        clusters = []
        colors = self._generate_colors(len(unique_labels))

        for i, label in enumerate(sorted(unique_labels)):
            indices = np.where(labels == label)[0]
            cluster_pcd = pcd.select_by_index(indices)
            cluster_pcd.paint_uniform_color(colors[i])

            points = np.asarray(cluster_pcd.points)
            centroid = points.mean(axis=0)
            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            bbox.color = colors[i]

            cluster = Cluster(
                points=cluster_pcd,
                indices=indices,
                label=label,
                centroid=centroid,
                bbox=bbox
            )
            clusters.append(cluster)

            print(f"  클러스터 {label}: {len(indices):,}개 점, 중심 ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")

        return clusters

    def euclidean_clustering(self,
                             pcd: o3d.geometry.PointCloud,
                             tolerance: float = 0.1,
                             min_size: int = 10,
                             max_size: int = 100000) -> List[Cluster]:
        """유클리디안 클러스터링

        DBSCAN과 유사하지만 더 간단한 거리 기반 클러스터링
        Open3D에서는 DBSCAN으로 구현
        """
        return self.cluster_dbscan(pcd, eps=tolerance, min_points=min_size)

    def segment_scene(self,
                      pcd: o3d.geometry.PointCloud) -> Tuple[List[Plane], List[Cluster]]:
        """전체 씬 세그멘테이션

        1. 평면 추출 (바닥, 벽 등)
        2. 나머지 점들 클러스터링 (객체)

        Args:
            pcd: 입력 Point Cloud

        Returns:
            (평면 리스트, 클러스터 리스트)
        """
        print("=" * 50)
        print("씬 세그멘테이션 시작")
        print("=" * 50)

        # 1. 평면 추출
        planes, remaining = self.extract_multiple_planes(pcd, max_planes=2)

        # 2. 나머지 클러스터링
        if len(remaining.points) > self.cluster_min_points:
            clusters = self.cluster_dbscan(remaining)
        else:
            clusters = []

        print("=" * 50)
        print(f"세그멘테이션 완료: {len(planes)}개 평면, {len(clusters)}개 객체")
        print("=" * 50)

        return planes, clusters

    def visualize_segmentation(self,
                               planes: List[Plane],
                               clusters: List[Cluster],
                               show_bbox: bool = True) -> None:
        """세그멘테이션 결과 시각화"""
        geometries = []

        # 평면 추가
        for plane in planes:
            geometries.append(plane.points)

        # 클러스터 추가
        for cluster in clusters:
            geometries.append(cluster.points)
            if show_bbox:
                geometries.append(cluster.bbox)

        o3d.visualization.draw_geometries(
            geometries,
            window_name="Segmentation Result",
            width=1024,
            height=768
        )

    @staticmethod
    def _generate_colors(n: int) -> List[List[float]]:
        """N개의 구분 가능한 색상 생성"""
        colors = [
            [1, 0, 0],      # 빨강
            [0, 1, 0],      # 초록
            [0, 0, 1],      # 파랑
            [1, 1, 0],      # 노랑
            [1, 0, 1],      # 마젠타
            [0, 1, 1],      # 시안
            [1, 0.5, 0],    # 주황
            [0.5, 0, 1],    # 보라
        ]

        while len(colors) < n:
            colors.append(np.random.rand(3).tolist())

        return colors[:n]
