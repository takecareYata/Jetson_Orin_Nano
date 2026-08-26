# main.py
import numpy as np
import time
import cv2
import pycuda.autoinit  # PyCUDA 컨텍스트 자동 관리 초기화용

# 작성한 커스텀 모듈에서 가속 엔진 래퍼 클래스 로드
from trt_module import TRTInferenceEngine

# 타겟 설정 변수
ENGINE_PATH = 'RPS_MobileNetV2.engine'
IMG_SIZE = 224

# 가속 엔진 인스턴스 생성 (선언과 동시에 VRAM 및 자원 세팅 완결)
trt_engine = TRTInferenceEngine(ENGINE_PATH)

ansToText = {0: 'scissors', 1: 'rock', 2: 'paper'}
colorList = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

def processImage(frame):
    # OpenCV 가공 이미지 전처리
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)
    inp = np.expand_dims(img, 0) # 배치 차원 추가: (1, 224, 224, 3)

    # 🚀 모듈 내부의 최신 비동기 추론 메서드 단 한 줄로 제어
    output_host = trt_engine.infer(inp)

    # 출력 벡터 해석 및 화면 연출
    output_data = output_host.reshape(-1)
    ans = int(np.argmax(output_data))
    text = ansToText.get(ans, str(ans))
    cv2.putText(frame, text, (180, 50), cv2.FONT_HERSHEY_PLAIN, 2, colorList[ans], 2)

# V4L2 카메라 캡처 인터페이스 설정
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('cam', 320+40, 240+60)

startTime = time.time()
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret: 
        break

    processImage(frame)

    # 정확한 실시간 처리량(FPS) 모니터링 연산
    curTime = time.time()
    fps = 1 / (curTime - startTime)
    startTime = curTime
    cv2.putText(frame, f'FPS: {fps:.1f}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    cv2.imshow('cam', frame)
    if cv2.waitKey(10) == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()