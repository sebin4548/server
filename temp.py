from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
import os, cv2
import numpy as np
from sklearn.cluster import HDBSCAN
from collections import Counter
from math import sqrt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# 폴더 설정
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# YOLO 모델 로드
MODEL_PATH = "/Users/kwonsebin/Documents/khu/4-1/capstone/runs/detect/train13/weights/best.pt"
detection_model = YOLO(MODEL_PATH)

# 루트 경로 - index.html 반환
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    # 동영상 파일 저장
    video_file = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(video_path)

    try:
        # YOLO 모델 실행
        detection_Ball_Glove = detection_model.track(video_path, save=True)

        # 분석 시작
        ball, glove = [], []
        for result in detection_Ball_Glove:
            for obj in result.boxes:
                if obj.cls == 2.0:  # 공
                    x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
                    ball.append([x1, y1])
                if obj.cls == 8.0:  # 글러브
                    x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
                    glove.append([x1, y1])

        # HDBSCAN 클러스터링
        # hdbBall = HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=10)
        # hdbBall.fit(ball)
        # hdbGlove = HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=10)
        # hdbGlove.fit(glove)

        # most_common_ball = Counter(hdbBall.labels_).most_common(1)[0][0]
        # most_common_glove = Counter(hdbGlove.labels_).most_common(1)[0][0]

        # 각 프레임 정보 저장
        frame = {}
        for i, result in enumerate(detection_Ball_Glove):
            info = {}
            for obj in result.boxes:
                if obj.cls == 2.0:  # 공
                    x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
                    # if hdbBall.labels_[ball.index([x1, y1])] == most_common_ball:
                        # info["ball"] = (x1, y1, x2, y2)
                    info["ball"] = (x1, y1, x2, y2)
                if obj.cls == 8.0:  # 글러브
                    x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
                    # if hdbGlove.labels_[glove.index([x1, y1])] == most_common_glove:
                        # info["glove"] = (x1, y1, x2, y2)
                    info["glove"] = (x1, y1, x2, y2)
            frame[i] = info

        # 거리 계산
        distances = {}
        for frame_number, objects in frame.items():
            if "ball" in objects and "glove" in objects:
                ball_coords = objects["ball"]
                glove_coords = objects["glove"]
                ball_center = ((ball_coords[0] + ball_coords[2]) / 2, (ball_coords[1] + ball_coords[3]) / 2)
                glove_center = ((glove_coords[0] + glove_coords[2]) / 2, (glove_coords[1] + glove_coords[3]) / 2)
                distance = sqrt((ball_center[0] - glove_center[0]) ** 2 + (ball_center[1] - glove_center[1]) ** 2)
                distances[frame_number] = distance

        # 선형 회귀
        X = np.array(list(distances.keys())).reshape(-1, 1)
        y = np.array(list(distances.values()))
        model = LinearRegression()
        model.fit(X, y)

        answer = int(-model.intercept_ / model.coef_[0])
        # answer = 50

        # 특정 프레임 저장
        frame_image_path = os.path.join(UPLOAD_FOLDER, 'result_frame.jpg')


        img = detection_Ball_Glove[answer].orig_img  # YOLO 결과 원본 이미지
        annotated_image = img.copy()
        # img.show()
        success = cv2.imwrite(frame_image_path, annotated_image)
        if success:
            print(f"Image saved successfully at: {frame_image_path}")
        else:
            print("Failed to save the image.")
            return jsonify({'error': 'Failed to save the image.'}), 500

        # 결과 반환
        return jsonify({
            'message': 'Video processed successfully',
            'frame': answer,
            'image_url': '/uploads/result_frame.jpg'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    full_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}")
        return jsonify({'error': 'File not found'}), 404
    print(f"Serving file from: {full_path}")
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(port=8081)
