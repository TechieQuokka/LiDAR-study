# LiDAR 센서 알고리즘 학습 프로젝트 아키텍처

## 1. 프로젝트 개요

### 1.1 목적
LiDAR(Light Detection and Ranging) 센서 데이터 처리 및 3D 객체 인식 알고리즘을 학습하기 위한 프로젝트이다. 실제 센서 없이 시뮬레이션 환경과 공개 데이터셋을 활용하여 Point Cloud 처리의 기초부터 딥러닝 기반 객체 인식까지 단계별로 학습한다.

### 1.2 학습 목표
- Point Cloud 데이터 구조와 처리 방법 이해
- 3D 스캐닝 데이터의 전처리 파이프라인 구축
- 세그멘테이션 및 클러스터링 알고리즘 구현
- 딥러닝 기반 3D 객체 인식 모델 활용

### 1.3 대상 독자
- 프로그래밍 경험이 있는 개발자
- LiDAR 및 Point Cloud 처리를 처음 접하는 학습자
- 3D 비전 및 객체 인식에 관심 있는 연구자

---

## 2. 기술 스택

### 2.1 프로그래밍 언어
**Python**을 주 언어로 사용한다. 빠른 프로토타이핑과 풍부한 라이브러리 생태계를 활용할 수 있으며, 추후 성능 최적화가 필요한 경우 C++로 전환을 고려한다.

### 2.2 핵심 라이브러리

| 라이브러리 | 역할 | 선정 이유 |
|-----------|------|----------|
| Open3D | Point Cloud 처리 | 현대적 API, 풍부한 문서, Python 친화적 |
| NumPy | 수치 연산 | 고성능 배열 연산, 표준 과학 계산 도구 |
| PyTorch | 딥러닝 모델 | 동적 계산 그래프, 연구용으로 적합 |
| scikit-learn | 전통적 머신러닝 | 분류/클러스터링 알고리즘 제공 |
| Matplotlib | 시각화 | 2D 그래프 및 분석 결과 표시 |

### 2.3 보조 도구

| 도구 | 용도 |
|------|------|
| CloudCompare | Point Cloud 시각화 및 수동 분석 (GUI) |
| Blender | 합성 데이터 생성 및 3D 모델링 |
| Jupyter Notebook | 실험 및 학습 기록 |

---

## 3. 학습 단계별 아키텍처

### 3.1 Phase 1: Point Cloud 기초

**목표**: Point Cloud 데이터의 구조를 이해하고 기본 연산을 수행한다.

**학습 내용**
- Point Cloud의 정의: 3차원 공간상의 점들의 집합 (x, y, z 좌표)
- 데이터 포맷: PLY, PCD, XYZ, LAS 등
- 기본 연산: 로딩, 저장, 시각화, 좌표 변환
- 속성 정보: 색상(RGB), 강도(Intensity), 법선 벡터(Normal)

**핵심 개념**
- 점(Point): 3D 공간의 단일 측정값
- 점군(Point Cloud): 점들의 비정형 집합
- 복셀(Voxel): 3D 공간을 격자로 분할한 단위

---

### 3.2 Phase 2: 전처리 파이프라인

**목표**: 노이즈가 포함된 원본 데이터를 분석에 적합한 형태로 가공한다.

**전처리 단계**

```
원본 데이터
    │
    ▼
┌─────────────────┐
│  다운샘플링      │  ← 데이터 양 감소, 계산 효율 향상
│  (Voxel Grid)   │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  노이즈 제거     │  ← 이상치(Outlier) 필터링
│  (Statistical)  │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  법선 추정       │  ← 각 점의 표면 방향 계산
│  (Normal Est.)  │
└─────────────────┘
    │
    ▼
정제된 데이터
```

**주요 알고리즘**
- Voxel Grid Downsampling: 공간을 격자로 나누고 각 격자 내 점들을 하나로 대표
- Statistical Outlier Removal: 통계적 기법으로 고립된 점 제거
- Radius Outlier Removal: 특정 반경 내 이웃 점 수 기준으로 필터링
- Normal Estimation: k-NN 또는 반경 기반 이웃 점으로 법선 벡터 계산

---

### 3.3 Phase 3: 세그멘테이션

**목표**: Point Cloud를 의미 있는 영역으로 분할한다.

**세그멘테이션 유형**

| 유형 | 설명 | 활용 |
|------|------|------|
| 기하학적 세그멘테이션 | 평면, 원기둥 등 기본 도형 추출 | 바닥/벽 분리 |
| 클러스터 기반 | 공간적 근접성으로 그룹화 | 개별 객체 분리 |
| 의미론적 세그멘테이션 | 각 점에 클래스 레이블 부여 | 객체 종류 구분 |

**주요 알고리즘**
- RANSAC (Random Sample Consensus): 평면 등 모델 피팅, 이상치에 강건
- Euclidean Clustering: 거리 기반 클러스터링, 간단하고 빠름
- DBSCAN: 밀도 기반 클러스터링, 다양한 형태의 클러스터 검출
- Region Growing: 시드 점에서 유사한 이웃으로 영역 확장

**처리 흐름**

```
전처리된 데이터
    │
    ├──→ RANSAC ──→ 평면 추출 (바닥, 벽)
    │
    └──→ 나머지 점들
              │
              ▼
         클러스터링 ──→ 개별 객체 후보
```

---

### 3.4 Phase 4: 객체 인식

**목표**: 분리된 Point Cloud 클러스터를 특정 객체 클래스로 분류한다.

**접근 방법 비교**

| 구분 | 전통적 방법 | 딥러닝 방법 |
|------|------------|------------|
| 특징 | 수작업 설계 디스크립터 | 자동 학습 특징 |
| 대표 기법 | FPFH + SVM | PointNet, PointNet++ |
| 장점 | 해석 가능, 적은 데이터 | 높은 성능, 복잡한 패턴 학습 |
| 단점 | 복잡한 형상에 한계 | 많은 데이터 필요 |

**전통적 파이프라인**

```
Point Cloud 클러스터
    │
    ▼
┌─────────────────┐
│  특징 추출       │  ← FPFH, SHOT 등
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  분류기          │  ← SVM, Random Forest
└─────────────────┘
    │
    ▼
객체 클래스 예측
```

**딥러닝 파이프라인 (PointNet)**

```
Point Cloud (N x 3)
    │
    ▼
┌─────────────────┐
│  Input Transform │  ← 3x3 변환 행렬 학습
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Shared MLP     │  ← 점별 특징 추출
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Max Pooling    │  ← 전역 특징 집계 (순서 불변성)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  FC Layers      │  ← 분류
└─────────────────┘
    │
    ▼
객체 클래스 예측
```

---

## 4. 데이터 아키텍처

### 4.1 사용 데이터셋

| 데이터셋 | 규모 | 특징 | 학습 단계 |
|----------|------|------|----------|
| ModelNet40 | 12,311개 CAD 모델 | 40개 카테고리, 깨끗한 데이터 | Phase 4 입문 |
| ShapeNet | 51,300개 모델 | 55개 카테고리, 파트 레이블 | Phase 4 심화 |
| ScanNet | 1,513개 씬 | 실내 RGB-D 스캔 | Phase 3 실습 |
| S3DIS | 6개 구역 | 실내 Point Cloud, 13개 클래스 | Phase 3-4 |

### 4.2 데이터 디렉토리 구조

```
data/
├── raw/                    # 원본 다운로드 데이터
│   ├── modelnet40/
│   ├── shapenet/
│   └── scannet/
├── processed/              # 전처리 완료 데이터
│   ├── train/
│   ├── val/
│   └── test/
└── samples/                # 학습용 샘플 데이터
```

### 4.3 데이터 포맷

**Point Cloud 파일 포맷**
- PLY (Polygon File Format): 가장 범용적, ASCII/Binary 지원
- PCD (Point Cloud Data): PCL 표준 포맷
- NPY/NPZ: NumPy 배열, Python에서 빠른 로딩

**데이터 표현**
- 기본: (N, 3) 배열 - N개 점의 x, y, z 좌표
- 확장: (N, 6) 배열 - 좌표 + 법선 벡터
- 컬러: (N, 6) 배열 - 좌표 + RGB 색상

---

## 5. 프로젝트 구조

```
LiDAR-test/
├── docs/                   # 문서
│   └── architecture.md     # 본 문서
├── notebooks/              # Jupyter 실험 노트북
│   ├── 01_point_cloud_basics.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_segmentation.ipynb
│   └── 04_object_recognition.ipynb
├── src/                    # 소스 코드
│   ├── data/               # 데이터 로딩 및 처리
│   ├── preprocessing/      # 전처리 모듈
│   ├── segmentation/       # 세그멘테이션 모듈
│   ├── recognition/        # 객체 인식 모듈
│   └── utils/              # 유틸리티 함수
├── data/                   # 데이터 디렉토리
├── models/                 # 학습된 모델 저장
├── configs/                # 설정 파일
├── tests/                  # 테스트 코드
├── requirements.txt        # 의존성 목록
└── README.md               # 프로젝트 설명
```

---

## 6. 학습 로드맵 상세

### 6.1 단계별 학습 목표

| 단계 | 주제 | 핵심 산출물 |
|------|------|------------|
| Phase 1 | Point Cloud 기초 | 데이터 시각화 노트북 |
| Phase 2 | 전처리 파이프라인 | 재사용 가능한 전처리 모듈 |
| Phase 3 | 세그멘테이션 | 실내 씬 분할 데모 |
| Phase 4 | 객체 인식 | ModelNet40 분류기 |

### 6.2 각 단계 학습 내용

**Phase 1 상세**
1. Open3D 설치 및 환경 구성
2. PLY/PCD 파일 로딩
3. 3D 시각화 (인터랙티브 뷰어)
4. 좌표 변환 (회전, 이동, 스케일)
5. 색상 및 속성 매핑

**Phase 2 상세**
1. Voxel Grid 다운샘플링 구현
2. Statistical Outlier Removal 적용
3. 법선 벡터 추정
4. 파이프라인 클래스로 통합

**Phase 3 상세**
1. RANSAC 평면 추출
2. DBSCAN 클러스터링
3. Region Growing 알고리즘
4. ScanNet 데이터로 실습

**Phase 4 상세**
1. FPFH 특징 디스크립터 계산
2. SVM 분류기 학습
3. PointNet 구조 이해
4. ModelNet40 학습 및 평가

---

## 7. 참고 자료

### 7.1 논문
- PointNet: Deep Learning on Point Sets (Qi et al., 2017)
- PointNet++: Deep Hierarchical Feature Learning (Qi et al., 2017)
- VoxNet: A 3D Convolutional Neural Network (Maturana & Scherer, 2015)

### 7.2 온라인 자료
- Open3D 공식 문서 및 튜토리얼
- PCL (Point Cloud Library) 튜토리얼
- Stanford 3D Scanning Repository

### 7.3 도서
- 3D Point Cloud Processing and Learning for Autonomous Driving
- Computer Vision: Algorithms and Applications (Szeliski)

---

## 8. 향후 확장 방향

### 8.1 고급 주제
- 다중 스캔 정합 (Multi-scan Registration)
- 실시간 SLAM (Simultaneous Localization and Mapping)
- 3D Object Detection (자율주행 분야)
- Instance Segmentation

### 8.2 실제 센서 연동
- ROS (Robot Operating System) 연동
- 저가형 LiDAR 센서 (RPLidar, Livox 등)
- RGB-D 카메라 (Intel RealSense, Azure Kinect)

---

*문서 작성일: 2026-01-27*
*마지막 수정: 2026-01-27*
