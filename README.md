# LostArk Buff Eye

로스트아크 버프 아이콘과 남은 시간을 인식해 Windows 데스크톱 오버레이로 표시하는 Python/OpenCV/PySide6 앱입니다. 창 모드와 전체 창 모드를 대상으로 합니다.

이 저장소에는 현재 앱 소스, 실행 및 빌드 스크립트, 현재 설정과 필수 인식·표시 에셋만 포함합니다. 개발 기록, 원본 화면 캡처, 실험 자료, 실행 로그, 의존성 설치 폴더와 EXE 빌드 결과는 제외합니다. `datasets/calibration-3440`에는 실행에 필요한 파일만 있습니다.

## 실행

Windows x64와 Python 3.12 환경에서 PowerShell로 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
.\.venv\Scripts\python.exe -m buff_eye.app
```

설정창에서 인식 영역과 표시할 효과를 조정합니다. 사용자 설정과 등록 이미지는 `%LOCALAPPDATA%\LostArkBuffEye`에 저장됩니다.

## 단일 EXE 빌드

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.venv\Scripts\python.exe scripts/build_exe.py
```

결과는 `dist/LostArkBuffEye.exe`입니다.

## 현재 제한

독점 전체 화면은 지원하지 않습니다. 작은 아이콘의 시간 판독, 강한 반사로 가려진 최초 아이콘 인식, 다른 PC에서의 장시간 사용은 추가 검증이 필요합니다. 미검출은 버프 미적용을 의미하지 않습니다.
