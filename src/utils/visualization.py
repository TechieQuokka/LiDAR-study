"""시각화 유틸리티"""

import numpy as np
import open3d as o3d
from typing import List, Optional
import matplotlib.pyplot as plt


class Visualizer:
    """Point Cloud 시각화 유틸리티"""

    @staticmethod
    def show(pcd: o3d.geometry.PointCloud,
             window_name: str = "Point Cloud",
             point_size: float = 2.0) -> None:
        """단일 Point Cloud 표시"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1024, height=768)
        vis.add_geometry(pcd)

        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.array([0.1, 0.1, 0.1])

        vis.run()
        vis.destroy_window()

    @staticmethod
    def show_multiple(geometries: List,
                      window_name: str = "Point Clouds",
                      point_size: float = 2.0) -> None:
        """여러 지오메트리 표시"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1024, height=768)

        for geom in geometries:
            vis.add_geometry(geom)

        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.array([0.1, 0.1, 0.1])

        vis.run()
        vis.destroy_window()

    @staticmethod
    def show_comparison(pcd1: o3d.geometry.PointCloud,
                        pcd2: o3d.geometry.PointCloud,
                        title1: str = "Before",
                        title2: str = "After") -> None:
        """두 Point Cloud 비교 (나란히)"""
        # 두 번째를 오른쪽으로 이동
        points1 = np.asarray(pcd1.points)
        points2 = np.asarray(pcd2.points)

        offset = points1.max(axis=0)[0] - points2.min(axis=0)[0] + 1

        pcd2_shifted = o3d.geometry.PointCloud(pcd2)
        pcd2_shifted.translate([offset, 0, 0])

        print(f"좌측: {title1} ({len(pcd1.points):,}개)")
        print(f"우측: {title2} ({len(pcd2.points):,}개)")

        Visualizer.show_multiple([pcd1, pcd2_shifted], window_name=f"{title1} vs {title2}")

    @staticmethod
    def plot_histogram(pcd: o3d.geometry.PointCloud,
                       axis: int = 2,
                       bins: int = 50) -> None:
        """포인트 분포 히스토그램"""
        points = np.asarray(pcd.points)
        axis_names = ['X', 'Y', 'Z']

        plt.figure(figsize=(10, 4))
        plt.hist(points[:, axis], bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel(f'{axis_names[axis]} coordinate')
        plt.ylabel('Count')
        plt.title(f'Point Distribution along {axis_names[axis]}-axis')
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def create_coordinate_frame(size: float = 1.0,
                                origin: tuple = (0, 0, 0)) -> o3d.geometry.TriangleMesh:
        """좌표계 프레임 생성"""
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size,
            origin=origin
        )
        return frame

    @staticmethod
    def colorize_by_height(pcd: o3d.geometry.PointCloud,
                           axis: int = 2) -> o3d.geometry.PointCloud:
        """높이에 따라 색상 적용"""
        points = np.asarray(pcd.points)
        heights = points[:, axis]

        # 정규화
        h_min, h_max = heights.min(), heights.max()
        h_normalized = (heights - h_min) / (h_max - h_min + 1e-6)

        # 컬러맵 적용 (파랑 -> 빨강)
        colors = np.zeros((len(heights), 3))
        colors[:, 0] = h_normalized        # R
        colors[:, 2] = 1 - h_normalized    # B

        result = o3d.geometry.PointCloud(pcd)
        result.colors = o3d.utility.Vector3dVector(colors)
        return result
