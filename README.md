# legacy-work (ML 위주의 프로젝트)
- 백엔드와 MLOPS 작업은 회사 깃헙에서만 작업하였다 보니 자료가 없어서 포함하지 못하였습니다 (포트폴리오 PPT를 참고해주세요)

## catch-app
- 머신러닝 적용 이전 OpenCV와 기하 계산만으로 가설/실험 등을 수행한 안드로이드 앱 프로젝트입니다
- face 관련 모델은 외부 모델 (센스타임)

## auto-catchnet
- 연구 초기 단계에 시선 추적 자료들을 조사하고 실험했던 프로젝트입니다
- 자주 쓰는 파이썬 기능 들에 대한 헬퍼 모음 (AC 폴더)
- 공개 데이터셋 전처리, 생성, 변환, 데이터 로더 (DS 폴더)
- 기하 연산, 타원 피팅, 교점 계산, 최적화, 카메라 fov 등등의 수학 헬퍼 함수와 알고리즘 폴더 (AL 폴더)
- eNAS (neural architecture search), iris landmark, face landmark, unity-eye 생성 이미지 훈련 등의 머신러닝 폴더 (ai 폴더)
- 외부 시선 추적 관련 연구 자료 모음 (pps 폴더)
- 기록 노트 혹은 쥬피터 노트북으로 실험, 코드 스케치 등의 작성 (notes 폴더)
  
## chai
- (leo)latent embedding optimization, hypernetwork, MAML++, 3dmm, 3d dense face alignment 등의 실험을 수행했던 프로젝트입니다
- chai/models/tf/[leo,maml] (torch 폴더에 있는 것은 제가 작성한 것이 아닌 오픈소스 원본 구현체입니다)
- 기본적 개발 환경이 세팅되어 있는 도커파일 작성 (tools 폴더)

