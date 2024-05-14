import cv2
import numpy as np
import tensorflow as tf

# 감정 카테고리 정의
emotions = ["Angry", "Fear", "Happy", "Sad", "Surprise"]

# 모델 로드
model = tf.keras.models.load_model("emotion_detection_model.h5")

# 웹캠 열기
cap = cv2.VideoCapture(0)

while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 크기 변경 및 전처리
    resized_frame = cv2.resize(frame, (48, 48))
    input_frame = np.expand_dims(resized_frame, axis=0) / 255.0

    # 감정 예측
    prediction = model.predict(input_frame)
    predicted_emotion = emotions[np.argmax(prediction)]

    # 프레임에 감정 텍스트 표시
    cv2.putText(frame, predicted_emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 프레임 출력
    cv2.imshow('Emotion Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 윈도우 종료
cap.release()
cv2.destroyAllWindows()
