"""
LiDAR Point Cloud 처리 데모

설치 후 이 스크립트로 기본 기능을 테스트합니다.
"""

from src.data import PointCloudLoader
from src.preprocessing import Preprocessor
from src.segmentation import Segmenter
from src.utils import Visualizer


def main():
    print("=" * 60)
    print("LiDAR Point Cloud 처리 데모")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1] 샘플 씬 생성...")
    loader = PointCloudLoader()
    scene = loader.create_sample_scene(add_noise=True)

    print("\n원본 씬 시각화 (창을 닫으면 계속)")
    Visualizer.show(scene, "1. Original Scene (with noise)")

    # 2. 전처리
    print("\n[2] 전처리 파이프라인...")
    preprocessor = Preprocessor()
    scene_clean = preprocessor.full_pipeline(
        scene,
        voxel_size=0.03,
        remove_noise=True,
        estimate_normals=True
    )

    print("\n전처리 후 시각화 (창을 닫으면 계속)")
    Visualizer.show(scene_clean, "2. Preprocessed Scene")

    # 3. 세그멘테이션
    print("\n[3] 씬 세그멘테이션...")
    segmenter = Segmenter()
    planes, clusters = segmenter.segment_scene(scene_clean)

    print("\n세그멘테이션 결과 시각화")
    segmenter.visualize_segmentation(planes, clusters, show_bbox=True)

    print("\n" + "=" * 60)
    print("데모 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
