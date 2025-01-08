import mediapipe as mp
import cv2
import numpy as np
import time
import torch

# Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# YOLOv5 모델 로드
# torch.hub에서 YOLOv5 모델을 다운로드하거나 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5s 모델 사용
model.classes = [0]  # 'person' 클래스만 사용 (손을 특정하려면 커스텀 모델 필요)

def detect_hands_with_yolo(image):
    """
    YOLO를 사용하여 손 탐지
    """
    results = model(image)
    detections = results.xyxy[0].cpu().numpy()  # 바운딩 박스 좌표
    hands = []
    for *box, conf, cls in detections:
        # YOLO의 바운딩 박스 좌표
        x_min, y_min, x_max, y_max = map(int, box)
        hands.append((x_min, y_min, x_max, y_max))
    return hands

def run():
    cap = cv2.VideoCapture(0)  # 웹캠 입력
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("카메라 입력을 읽을 수 없습니다.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_hands = detect_hands_with_yolo(frame_rgb)  # YOLO로 손 탐지

            for (x_min, y_min, x_max, y_max) in detected_hands:
                # 탐지된 손 영역을 Crop
                hand_image = frame_rgb[y_min:y_max, x_min:x_max]

                # Mediapipe로 손 랜드마크 분석
                results = hands.process(hand_image)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # 랜드마크를 Crop된 이미지가 아닌 원본 프레임에 그리기
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                        )
                        # 손 바운딩 박스 표시
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # 프레임 출력
            cv2.imshow('YOLO + Mediapipe', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
