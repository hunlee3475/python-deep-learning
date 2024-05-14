import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 감정 카테고리 배열 형태로 정의
emotions = ["Angry", "Fear", "Happy", "Sad", "Surprise"]

# 데이터셋 전처리
X_train, y_train = [], []  # 이미지 데이터와 레이블을 저장할 리스트
for emotion in emotions:
    for i in range(1, 451):  # 각 감정에 대해 450개의 이미지를 반복적으로 처리
        # 이미지 로드
        img = cv2.imread(f"./{emotion}/img{i}.jpg", cv2.IMREAD_COLOR)
        # 이미지 크기 조정 및 정규화
        img = cv2.resize(img, (48, 48))
        img = img.astype(np.float32) / 255.0
        # 데이터 및 레이블 추가
        X_train.append(img)
        y_train.append(emotions.index(emotion))

X_train = np.array(X_train)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(emotions))  # one-hot encoding

# 데이터셋 섞기
idx = np.random.permutation(len(X_train))
X_train, y_train = X_train[idx], y_train[idx]

# 감정 분석을 위한 CNN 모델 생성
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(emotions), activation='softmax')
])

#'accuracy'를 사용하여 정확도를 모니터링합니다.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#75번 반복하여 학습합니다.
#10%를 검증 데이터로 사용합니다
history = model.fit(X_train, y_train, epochs=75, validation_split=0.1)
#학습된 모델을 파일로 저장합니다
model.save("emotion_detection_model.h5")

# 테스트 이미지 예측 및 시각화
test_predictions = []
for i in range(1, 11):
    test_img = cv2.imread(f"./test_images/test_image{i}.jpg", cv2.IMREAD_COLOR)  # 컬러 이미지로 로드
    test_img = cv2.resize(test_img, (48, 48))
    test_img = test_img.astype(np.float32) / 255.0  # float32로 변환 및 정규화
    test_img = np.expand_dims(test_img, axis=0)  # 배치 차원 추가
    prediction = model.predict(test_img)
    test_predictions.append(prediction)

# 예측 결과 시각화
num_test_images = len(test_predictions)
fig, axs = plt.subplots(2, num_test_images // 2, figsize=(15, 6))
axs = axs.ravel()
for i in range(num_test_images):
    axs[i].imshow(cv2.cvtColor(cv2.imread(f"./test_images/test_image{i + 1}.jpg", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    axs[i].set_title(emotions[np.argmax(test_predictions[i])])
    axs[i].axis('off')

plt.show()
