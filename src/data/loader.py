"""Point Cloud 데이터 로딩 및 샘플 생성

Phase 1: Point Cloud 기초
- 파일 로딩 (PLY, PCD)
- 샘플 데이터 생성
- 기본 시각화
"""

import numpy as np
import open3d as o3d
from pathlib import Path


class PointCloudLoader:
    """Point Cloud 로딩 및 샘플 생성 클래스"""

    def load(self, filepath: str) -> o3d.geometry.PointCloud:
        """파일에서 Point Cloud 로드

        Args:
            filepath: PLY, PCD, XYZ 등 파일 경로

        Returns:
            Open3D PointCloud 객체
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")

        pcd = o3d.io.read_point_cloud(str(path))
        print(f"로드 완료: {len(pcd.points):,}개 포인트")
        return pcd

    def save(self, pcd: o3d.geometry.PointCloud, filepath: str) -> None:
        """Point Cloud를 파일로 저장"""
        o3d.io.write_point_cloud(filepath, pcd)
        print(f"저장 완료: {filepath}")

    def create_sample_box(self,
                          center: tuple = (0, 0, 0),
                          size: tuple = (1, 1, 1),
                          density: int = 1000) -> o3d.geometry.PointCloud:
        """샘플 박스 Point Cloud 생성

        Args:
            center: 박스 중심 (x, y, z)
            size: 박스 크기 (width, height, depth)
            density: 포인트 밀도
        """
        # 각 면에 점 생성
        points = []
        cx, cy, cz = center
        w, h, d = size

        # 6면에 균등하게 점 배치
        n = int(np.sqrt(density / 6))

        for face in range(6):
            u = np.random.uniform(-0.5, 0.5, n * n)
            v = np.random.uniform(-0.5, 0.5, n * n)

            if face == 0:    # 앞면
                pts = np.column_stack([u * w + cx, v * h + cy, np.full(n*n, cz + d/2)])
            elif face == 1:  # 뒷면
                pts = np.column_stack([u * w + cx, v * h + cy, np.full(n*n, cz - d/2)])
            elif face == 2:  # 왼쪽
                pts = np.column_stack([np.full(n*n, cx - w/2), v * h + cy, u * d + cz])
            elif face == 3:  # 오른쪽
                pts = np.column_stack([np.full(n*n, cx + w/2), v * h + cy, u * d + cz])
            elif face == 4:  # 위
                pts = np.column_stack([u * w + cx, np.full(n*n, cy + h/2), v * d + cz])
            else:            # 아래
                pts = np.column_stack([u * w + cx, np.full(n*n, cy - h/2), v * d + cz])

            points.append(pts)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.vstack(points))
        pcd.paint_uniform_color([0.3, 0.6, 0.9])  # 파란색
        return pcd

    def create_sample_sphere(self,
                             center: tuple = (0, 0, 0),
                             radius: float = 0.5,
                             n_points: int = 1000) -> o3d.geometry.PointCloud:
        """샘플 구 Point Cloud 생성"""
        # 균등 분포 구면 좌표
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        cos_theta = np.random.uniform(-1, 1, n_points)
        theta = np.arccos(cos_theta)

        x = radius * np.sin(theta) * np.cos(phi) + center[0]
        y = radius * np.sin(theta) * np.sin(phi) + center[1]
        z = radius * np.cos(theta) + center[2]

        points = np.column_stack([x, y, z])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.9, 0.3, 0.3])  # 빨간색
        return pcd

    def create_sample_plane(self,
                            center: tuple = (0, 0, 0),
                            size: tuple = (5, 5),
                            n_points: int = 2000) -> o3d.geometry.PointCloud:
        """샘플 평면 Point Cloud 생성 (바닥면)"""
        x = np.random.uniform(-size[0]/2, size[0]/2, n_points) + center[0]
        y = np.random.uniform(-size[1]/2, size[1]/2, n_points) + center[1]
        z = np.full(n_points, center[2])

        points = np.column_stack([x, y, z])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 회색
        return pcd

    def create_sample_scene(self, add_noise: bool = True) -> o3d.geometry.PointCloud:
        """학습용 샘플 씬 생성 (바닥 + 여러 객체)

        Returns:
            바닥과 여러 물체가 포함된 Point Cloud
        """
        # 바닥
        floor = self.create_sample_plane(center=(0, 0, 0), size=(6, 6), n_points=3000)

        # 물체들
        box1 = self.create_sample_box(center=(-1, -1, 0.5), size=(1, 1, 1), density=800)
        box2 = self.create_sample_box(center=(1.5, 0.5, 0.3), size=(0.6, 0.6, 0.6), density=500)
        sphere1 = self.create_sample_sphere(center=(0.5, -1.5, 0.4), radius=0.4, n_points=600)
        sphere2 = self.create_sample_sphere(center=(-1.5, 1, 0.3), radius=0.3, n_points=400)

        # 합치기
        all_points = np.vstack([
            np.asarray(floor.points),
            np.asarray(box1.points),
            np.asarray(box2.points),
            np.asarray(sphere1.points),
            np.asarray(sphere2.points),
        ])

        all_colors = np.vstack([
            np.asarray(floor.colors),
            np.asarray(box1.colors),
            np.asarray(box2.colors),
            np.asarray(sphere1.colors),
            np.asarray(sphere2.colors),
        ])

        # 노이즈 추가
        if add_noise:
            noise = np.random.normal(0, 0.01, all_points.shape)
            all_points += noise

            # 이상치(outlier) 추가
            n_outliers = int(len(all_points) * 0.02)
            outlier_points = np.random.uniform(-3, 3, (n_outliers, 3))
            outlier_colors = np.tile([1, 1, 0], (n_outliers, 1))  # 노란색

            all_points = np.vstack([all_points, outlier_points])
            all_colors = np.vstack([all_colors, outlier_colors])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)

        print(f"샘플 씬 생성: {len(pcd.points):,}개 포인트")
        return pcd

    @staticmethod
    def visualize(pcd: o3d.geometry.PointCloud,
                  window_name: str = "Point Cloud") -> None:
        """Point Cloud 시각화"""
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=window_name,
            width=1024,
            height=768
        )

    @staticmethod
    def visualize_multiple(geometries: list,
                           window_name: str = "Point Clouds") -> None:
        """여러 Point Cloud 동시 시각화"""
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            width=1024,
            height=768
        )

    @staticmethod
    def get_info(pcd: o3d.geometry.PointCloud) -> dict:
        """Point Cloud 정보 반환"""
        points = np.asarray(pcd.points)
        info = {
            "포인트 수": len(points),
            "최소 좌표": points.min(axis=0).tolist(),
            "최대 좌표": points.max(axis=0).tolist(),
            "중심": points.mean(axis=0).tolist(),
            "색상 여부": pcd.has_colors(),
            "법선 여부": pcd.has_normals(),
        }
        return info
