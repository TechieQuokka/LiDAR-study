# LiDAR 센서 알고리즘 학습 프로젝트

Point Cloud 처리와 3D 객체 인식을 단계별로 학습하는 프로젝트입니다.

## 설치

```bash
pip install -r requirements.txt
```

## 프로젝트 구조

```
src/
├── data/           # 데이터 로딩
├── preprocessing/  # 전처리 (다운샘플링, 노이즈 제거)
├── segmentation/   # 세그멘테이션 (평면 추출, 클러스터링)
├── recognition/    # 객체 인식
└── utils/          # 유틸리티

notebooks/          # 실습 노트북
```

## 학습 단계

| Phase | 주제 | 노트북 |
|-------|------|--------|
| 1 | Point Cloud 기초 | `01_basics.ipynb` |
| 2 | 전처리 파이프라인 | `02_preprocessing.ipynb` |
| 3 | 세그멘테이션 | `03_segmentation.ipynb` |
| 4 | 객체 인식 | `04_recognition.ipynb` |

## 빠른 시작

```python
from src.data import PointCloudLoader
from src.preprocessing import Preprocessor
from src.segmentation import Segmenter

# 데이터 로드
loader = PointCloudLoader()
pcd = loader.create_sample_scene()

# 전처리
preprocessor = Preprocessor()
pcd_clean = preprocessor.full_pipeline(pcd)

# 세그멘테이션
segmenter = Segmenter()
planes, objects = segmenter.segment_scene(pcd_clean)
```
