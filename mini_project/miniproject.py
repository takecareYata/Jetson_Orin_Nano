import os
import time
import cv2
import numpy as np
import pycuda.autoinit  # pylint: disable=unused-import


# 사용자 정의 TensorRT 추론 모듈 로딩
try:
    from trt_module import TRTInferenceEngine
except ImportError:
    print("오류: 'trt_module.py' 파일을 찾을 수 없습니다.")
    exit(1)

# TensorRT 엔진 파일 경로
ENGINE_PATH = 'minipj_5.engine'

# 가위바위보 클래스 정의 (id -> 텍스트)
RPS_MAP = {
    0: {'en': 'scissors', 'ko': '가위'},
    1: {'en': 'rock', 'ko': '바위'},
    2: {'en': 'paper', 'ko': '보'}
}

# BGR 색상 정의
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_MAGENTA = (255, 0, 255)

colorList = [COLOR_RED, COLOR_GREEN, COLOR_BLUE]

# 승패 결과 정의
RESULT_MAP = {
    'WIN': {'text': 'WIN!', 'color': COLOR_YELLOW},
    'LOSE': {'text': 'LOSE', 'color': COLOR_RED},
    'DRAW': {'text': 'DRAW', 'color': COLOR_MAGENTA}
}

# 임계값 설정
IMG_SIZE = 320
NUM_CLASSES = 3
CONF_TH = 0.5
IOU_TH = 0.45

# --- 1. 엔진 로딩 및 초기화 ---
if not os.path.exists(ENGINE_PATH):
    print(f"오류: 엔진 파일 '{ENGINE_PATH}'을 찾을 수 없습니다.")
    exit(1)

trt_engine = TRTInferenceEngine(ENGINE_PATH)
input_tensor = trt_engine.tensors[trt_engine.input_name]
output_tensor = trt_engine.tensors[trt_engine.output_name]
output_shape = tuple(output_tensor['shape'])

print(f"엔진 파일: {ENGINE_PATH}")
print(f"입력 텐서 shape: {tuple(input_tensor['shape'])}")
print(f"출력 텐서 shape: {output_shape}")


def letterbox(img, new_shape=(320, 320), color=(114, 114, 114)):
    """이미지 비율을 유지하며 패딩을 추가하는 전처리 함수"""
    h, w = img.shape[:2]
    nh, nw = new_shape
    r = min(float(nw) / w, float(nh) / h)

    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

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
    """NumPy 1.x 기반 고속 Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0].astype(np.float32)
    y1 = boxes[:, 1].astype(np.float32)
    x2 = boxes[:, 2].astype(np.float32)
    y2 = boxes[:, 3].astype(np.float32)
    
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


def get_game_result(p1_id, p2_id):
    """가위바위보 승패 판정 함수 (0: 가위, 1: 바위, 2: 보)"""
    if p1_id == p2_id:
        return 'DRAW'
    elif (p1_id == 0 and p2_id == 2) or \
         (p1_id == 1 and p2_id == 0) or \
         (p1_id == 2 and p2_id == 1):
        return 'WIN'
    else:
        return 'LOSE'


def processImage(frame):
    """이미지 전처리 및 TRT 추론, 결과 데이터 추출"""
    H, W = frame.shape[:2]

    # 1. 이미지 전처리 및 C-contiguous 메모리 배치 명시
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_lb, r, pad_x, pad_y = letterbox(img_rgb, (IMG_SIZE, IMG_SIZE))
    img = img_lb.astype(np.float32) / 255.0
    img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)

    # 2. 배치 차원 확장 및 추론
    input_data = np.expand_dims(img, axis=0)
    raw_output = trt_engine.infer(input_data)

    # 3. 출력 차원 재구성 (1, 7, 2100) -> (2100, 7)
    raw = np.array(raw_output, copy=False).reshape(output_shape)[0]
    detections = raw.T

    # 4. 필터링 및 Bounding Box 연산
    cx, cy, w, h = detections[:, 0], detections[:, 1], detections[:, 2], detections[:, 3]
    cls_scores = detections[:, 4:4 + NUM_CLASSES]

    class_ids = np.argmax(cls_scores, axis=1)
    scores = np.max(cls_scores, axis=1)

    keep_idx = np.where(scores >= CONF_TH)[0]
    if len(keep_idx) == 0:
        return []

    cx, cy, w, h = cx[keep_idx], cy[keep_idx], w[keep_idx], h[keep_idx]
    scores = scores[keep_idx]
    class_ids = class_ids[keep_idx]

    # 좌표 복원 및 클리핑
    x1 = np.clip((cx - w / 2.0 - pad_x) / r, 0, W).astype(np.int32)
    y1 = np.clip((cy - h / 2.0 - pad_y) / r, 0, H).astype(np.int32)
    x2 = np.clip((cx + w / 2.0 - pad_x) / r, 0, W).astype(np.int32)
    y2 = np.clip((cy + h / 2.0 - pad_y) / r, 0, H).astype(np.int32)

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # 5. NMS 적용
    keep = nms(boxes, scores, IOU_TH)
    if not keep:
        return []

    det_list = []
    for i in keep:
        det_list.append({
            'box': boxes[i],
            'score': float(scores[i]),
            'class_id': int(class_ids[i]),
            'area': int((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
        })

    # 바운딩 박스 크기 순으로 최대 2개만 선택 후 X좌표 기준(좌->우) 정렬
    det_list = sorted(det_list, key=lambda x: x['area'], reverse=True)[:2]
    det_list = sorted(det_list, key=lambda x: x['box'][0])

    return det_list


def draw_game_results(frame, det_list):
    """화면에 Player1, Player2 박스 및 승/패/무승부(DRAW) 판정 결과 시각화"""
    H, W = frame.shape[:2]

    if len(det_list) == 2:
        p1, p2 = det_list[0], det_list[1]
        p1_result = get_game_result(p1['class_id'], p2['class_id'])

        if p1_result == 'WIN':
            p2_result = 'LOSE'
        elif p1_result == 'LOSE':
            p2_result = 'WIN'
        else:
            p2_result = 'DRAW'

        p1['result'] = p1_result
        p2['result'] = p2_result

        # 상단 오버레이 영역
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, 90), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "VS", (W // 2 - 20, 55), cv2.FONT_HERSHEY_DUPLEX, 1.2, COLOR_WHITE, 2)

    else:
        cv2.putText(frame, "Waiting for 2 Hands...", (W // 2 - 130, H // 2),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, COLOR_YELLOW, 2)
        p1 = det_list[0] if len(det_list) == 1 else None
        p2 = None

    for i, p in enumerate([p1, p2]):
        if p is None:
            continue

        label = f"PLAYER{i+1}"
        box_color = colorList[p['class_id'] % len(colorList)]
        bx1, by1, bx2, by2 = p['box']
        rps_text = RPS_MAP[p['class_id']]['en'].upper()

        # 바운딩 박스 그려주기
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2)
        cv2.putText(frame, f"{label}: {rps_text} ({p['score']:.2f})",
                    (bx1, max(0, by1 - 10)), cv2.FONT_HERSHEY_PLAIN, 1.2, box_color, 2)

        if 'result' in p:
            res_info = RESULT_MAP[p['result']]
            res_text = res_info['text']
            res_color = res_info['color']

            text_x = 20 if i == 0 else W // 2 + 50
            cv2.putText(frame, f"{label}: {rps_text}", (text_x, 35), cv2.FONT_HERSHEY_PLAIN, 1.3, COLOR_WHITE, 2)
            cv2.putText(frame, res_text, (text_x + 10, 75), cv2.FONT_HERSHEY_DUPLEX, 1.5, res_color, 3)
            cv2.putText(frame, res_text, (bx1 + 10, by1 + 35), cv2.FONT_HERSHEY_DUPLEX, 1.2, res_color, 2)


# --- 2. 비디오 루프 실행 ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("오류: 카메라를 연결할 수 없습니다.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow('RPS Game', cv2.WINDOW_NORMAL)
cv2.resizeWindow('RPS Game', 740, 580)

startTime = time.time()
frame_count = 0
fps = 0.0

print("가위바위보 게임 시작 ('q' 누르면 종료)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    det_list = processImage(frame)
    draw_game_results(frame, det_list)

    # FPS 수치 연산
    frame_count += 1
    if frame_count % 10 == 0:
        curTime = time.time()
        fps = 10.0 / (curTime - startTime)
        startTime = curTime

    H, W = frame.shape[:2]
    cv2.putText(frame, f'FPS: {fps:.1f}', (W - 110, H - 15), cv2.FONT_HERSHEY_PLAIN, 1.3, COLOR_GREEN, 2)

    cv2.imshow('RPS Game', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()