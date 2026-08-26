# # ========================================================
# # [강제 조치] 다른 어떤 모듈보다 먼저 CUDA 컨텍스트를 초기화합니다.
# import pycuda.driver as cuda
# import pycuda.autoinit  # pylint: disable=unused-import
# # ========================================================

import os
import time
import cv2
import numpy as np
import pycuda.autoinit  # pylint: disable=unused-import


# 사용자 정의 TensorRT 추론 모듈 로딩
from trt_module import TRTInferenceEngine

# TensorRT 엔진 파일 경로
ENGINE_PATH = 'rps_yolo11n.engine'
# ENGINE_PATH = 'best.engine'

# 바운딩 박스 텍스트 및 색상 정의
ansToText = {0: 'scissors', 1: 'rock', 2: 'paper'}
colorList = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# 설정값
IMG_SIZE = 320
NUM_CLASSES = len(ansToText)
CONF_TH = 0.5
IOU_TH = 0.45

# --- 1. 엔진 로딩 및 컨텍스트 초기화 ---
trt_engine = TRTInferenceEngine(ENGINE_PATH)
input_tensor = trt_engine.tensors[trt_engine.input_name]
output_tensor = trt_engine.tensors[trt_engine.output_name]
output_shape = tuple(output_tensor['shape'])

print(f"엔진 파일: {ENGINE_PATH}")
print(f"입력 텐서: {trt_engine.input_name} - shape: {tuple(input_tensor['shape'])} - dtype: {trt_engine.engine.get_tensor_dtype(trt_engine.input_name)}")
print(f"출력 텐서: {trt_engine.output_name} - shape: {output_shape} - dtype: {trt_engine.engine.get_tensor_dtype(trt_engine.output_name)}")


def letterbox(img, new_shape=(320, 320), color=(114, 114, 114)):
    """이미지 비율을 유지하면서 패딩을 추가하는 전처리 함수"""
    h, w = img.shape[:2]
    nh, nw = new_shape
    r = min(nw / w, nh / h)

    new_w, new_h = int(w * r), int(h * r)
    resized = cv2.resize(img, (new_w, new_h))

    pad_w = nw - new_w
    pad_h = nh - new_h
    pad_x = pad_w // 2
    pad_y = pad_h // 2
    right = pad_w - pad_x
    bottom = pad_h - pad_y

    padded = cv2.copyMakeBorder(
        resized,
        pad_y, bottom,
        pad_x, right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded, r, pad_x, pad_y


def nms(boxes, scores, iou_th):
    """넘파이 배열 기반의 고속 Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_th)[0]
        order = order[inds + 1]

    return keep


def processImage(frame):
    H, W = frame.shape[:2]

    # 1. 이미지 전처리
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_lb, r, pad_x, pad_y = letterbox(img_rgb, (IMG_SIZE, IMG_SIZE))
    img = img_lb.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.ascontiguousarray(img)

    # 2. 배치 차원 추가 후 추론 실행
    input_data = np.expand_dims(img, axis=0)
    raw_output = trt_engine.infer(input_data) #.copy()  # GPU에서 CPU로 결과 복사
    # print(f"원시 출력 shape: {raw_output.shape} - dtype: {raw_output.dtype}")

    # 3. 중요: YOLOv11 원본 출력 구조 복원 및 전치 (1, 7, 2100) -> (2100, 7)
    raw = raw_output.reshape(output_shape)[0]
    detections = raw.T  # 각 행: [cx, cy, w, h, score0, score1, score2]

    # 4. 넘파이 벡터화 연산 기반의 Confidence 필터링
    cx, cy, w, h = detections[:, 0], detections[:, 1], detections[:, 2], detections[:, 3]
    cls_scores = detections[:, 4:4 + NUM_CLASSES]

    class_ids = np.argmax(cls_scores, axis=1)
    scores = np.max(cls_scores, axis=1)

    # CONF_TH를 만족하는 인덱스 추출
    keep_idx = np.where(scores >= CONF_TH)[0]
    if len(keep_idx) == 0:
        return  # 검출 객체 없음 시 조기 종료

    # 필터링 적용
    cx, cy, w, h = cx[keep_idx], cy[keep_idx], w[keep_idx], h[keep_idx]
    scores = scores[keep_idx]
    class_ids = class_ids[keep_idx]

    # 5. 중심점 좌표 변환 및 원본 해상도 매핑 역연산
    x1 = np.clip((cx - w / 2.0 - pad_x) / r, 0, W).astype(np.int32)
    y1 = np.clip((cy - h / 2.0 - pad_y) / r, 0, H).astype(np.int32)
    x2 = np.clip((cx + w / 2.0 - pad_x) / r, 0, W).astype(np.int32)
    y2 = np.clip((cy + h / 2.0 - pad_y) / r, 0, H).astype(np.int32)

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # 6. NMS 필터링
    keep = nms(boxes, scores, IOU_TH)

    # 7. 검출된 최종 객체 시각화 (화면 드로잉)
    for i in keep:
        bx1, by1, bx2, by2 = boxes[i]
        cid = class_ids[i]
        sc = scores[i]

        if cid not in ansToText:
            continue

        color = colorList[cid % len(colorList)]
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        text = f"{ansToText[cid]} {sc:.2f}"
        cv2.putText(frame, text, (bx1, max(0, by1 - 7)), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)


# --- 2. 카메라 설정 및 비디오 루프 ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('cam', 320 + 40, 240 + 60)

startTime = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 처리 및 객체 추론 호출
    processImage(frame)

    # 실시간 FPS 연산
    curTime = time.time()
    fps = 1 / (curTime - startTime)
    startTime = curTime
    cv2.putText(frame, f'FPS: {fps:.1f}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    # 결과 화면 출력
    cv2.imshow('cam', frame)

    # 'q' 키 입력 시 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()