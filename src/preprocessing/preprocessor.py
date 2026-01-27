"""Point Cloud 전처리 파이프라인

Phase 2: 전처리
- 다운샘플링 (Voxel Grid)
- 노이즈 제거 (Statistical Outlier Removal)
- 법선 벡터 추정
"""

import numpy as np
import open3d as o3d
from typing import Tuple


class Preprocessor:
    """Point Cloud 전처리 클래스

    전처리 순서:
    1. 다운샘플링 - 점 개수 감소
    2. 노이즈 제거 - 이상치 필터링
    3. 법선 추정 - 표면 방향 계산
    """

    def __init__(self):
        # 기본 파라미터
        self.voxel_size = 0.05        # 다운샘플링 복셀 크기
        self.nb_neighbors = 20         # 노이즈 제거: 이웃 점 수
        self.std_ratio = 2.0           # 노이즈 제거: 표준편차 비율
        self.normal_radius = 0.1       # 법선 추정 반경
        self.normal_max_nn = 30        # 법선 추정 최대 이웃

    def downsample(self,
                   pcd: o3d.geometry.PointCloud,
                   voxel_size: float = None) -> o3d.geometry.PointCloud:
        """Voxel Grid 다운샘플링

        공간을 voxel_size 크기의 격자로 나누고,
        각 격자 내의 점들을 하나의 대표점으로 축소

        Args:
            pcd: 입력 Point Cloud
            voxel_size: 복셀 크기 (클수록 점이 적어짐)

        Returns:
            다운샘플링된 Point Cloud
        """
        voxel_size = voxel_size or self.voxel_size

        before = len(pcd.points)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        after = len(pcd_down.points)

        print(f"다운샘플링: {before:,} → {after:,} ({100*after/before:.1f}%)")
        return pcd_down

    def remove_outliers(self,
                        pcd: o3d.geometry.PointCloud,
                        nb_neighbors: int = None,
                        std_ratio: float = None) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """Statistical Outlier Removal

        각 점의 이웃들과의 평균 거리를 계산하고,
        전체 평균에서 std_ratio * 표준편차 이상 벗어난 점 제거

        Args:
            pcd: 입력 Point Cloud
            nb_neighbors: 이웃 점 개수
            std_ratio: 표준편차 비율 (클수록 관대함)

        Returns:
            (정제된 Point Cloud, 인라이어 인덱스)
        """
        nb_neighbors = nb_neighbors or self.nb_neighbors
        std_ratio = std_ratio or self.std_ratio

        before = len(pcd.points)
        pcd_clean, inlier_idx = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        after = len(pcd_clean.points)
        removed = before - after

        print(f"노이즈 제거: {removed:,}개 이상치 제거 ({100*removed/before:.1f}%)")
        return pcd_clean, np.array(inlier_idx)

    def remove_radius_outliers(self,
                               pcd: o3d.geometry.PointCloud,
                               nb_points: int = 16,
                               radius: float = 0.1) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """Radius Outlier Removal

        지정된 반경 내에 최소 nb_points 이상의 이웃이 없는 점 제거

        Args:
            pcd: 입력 Point Cloud
            nb_points: 최소 이웃 점 수
            radius: 검색 반경

        Returns:
            (정제된 Point Cloud, 인라이어 인덱스)
        """
        before = len(pcd.points)
        pcd_clean, inlier_idx = pcd.remove_radius_outlier(
            nb_points=nb_points,
            radius=radius
        )
        after = len(pcd_clean.points)
        removed = before - after

        print(f"반경 기반 노이즈 제거: {removed:,}개 제거")
        return pcd_clean, np.array(inlier_idx)

    def estimate_normals(self,
                         pcd: o3d.geometry.PointCloud,
                         radius: float = None,
                         max_nn: int = None) -> o3d.geometry.PointCloud:
        """법선 벡터 추정

        각 점 주변의 이웃 점들로 평면을 피팅하고
        그 평면의 법선 벡터를 해당 점의 법선으로 설정

        Args:
            pcd: 입력 Point Cloud
            radius: 이웃 검색 반경
            max_nn: 최대 이웃 점 수

        Returns:
            법선이 추정된 Point Cloud (원본 수정됨)
        """
        radius = radius or self.normal_radius
        max_nn = max_nn or self.normal_max_nn

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius,
                max_nn=max_nn
            )
        )

        # 법선 방향 일관성 맞추기 (카메라 방향 기준)
        pcd.orient_normals_consistent_tangent_plane(k=15)

        print(f"법선 추정 완료: {len(pcd.normals):,}개 법선")
        return pcd

    def full_pipeline(self,
                      pcd: o3d.geometry.PointCloud,
                      voxel_size: float = None,
                      remove_noise: bool = True,
                      estimate_normals: bool = True) -> o3d.geometry.PointCloud:
        """전체 전처리 파이프라인 실행

        Args:
            pcd: 입력 Point Cloud
            voxel_size: 다운샘플링 크기
            remove_noise: 노이즈 제거 여부
            estimate_normals: 법선 추정 여부

        Returns:
            전처리된 Point Cloud
        """
        print("=" * 50)
        print("전처리 파이프라인 시작")
        print("=" * 50)

        result = pcd

        # 1. 다운샘플링
        result = self.downsample(result, voxel_size)

        # 2. 노이즈 제거
        if remove_noise:
            result, _ = self.remove_outliers(result)

        # 3. 법선 추정
        if estimate_normals:
            result = self.estimate_normals(result)

        print("=" * 50)
        print(f"전처리 완료: 최종 {len(result.points):,}개 포인트")
        print("=" * 50)

        return result

    @staticmethod
    def crop_box(pcd: o3d.geometry.PointCloud,
                 min_bound: tuple,
                 max_bound: tuple) -> o3d.geometry.PointCloud:
        """Bounding Box로 영역 자르기"""
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound,
            max_bound=max_bound
        )
        return pcd.crop(bbox)

    @staticmethod
    def transform(pcd: o3d.geometry.PointCloud,
                  rotation: np.ndarray = None,
                  translation: np.ndarray = None) -> o3d.geometry.PointCloud:
        """Point Cloud 변환 (회전, 이동)"""
        transform_matrix = np.eye(4)

        if rotation is not None:
            transform_matrix[:3, :3] = rotation

        if translation is not None:
            transform_matrix[:3, 3] = translation

        return pcd.transform(transform_matrix)
