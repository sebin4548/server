import subprocess
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import os, cv2
import numpy as np
from sklearn.cluster import HDBSCAN
from collections import Counter
from math import sqrt
from sklearn.linear_model import LinearRegression
import mediapipe as mp
from numpy import dot
from numpy.linalg import norm
from IsTriangle import is_triangle_line_in_box
from Average import average
from cos import cosx
# from findFoot import find_foot

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 폴더 설정
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# YOLO 모델 로드
MODEL_PATH = "/Users/kwonsebin/Documents/khu/4-1/capstone/runs/detect/train13/weights/best.pt"
MODEL_PATH1 = "/Users/kwonsebin/Documents/khu/4-1/capstone/runs/detect/train82/weights/best.pt"

detection_model = YOLO(MODEL_PATH)
detection_model1 = YOLO(MODEL_PATH1)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False)

# 저장 폴더 생성
output_folder = "uploads"
os.makedirs(output_folder, exist_ok=True)

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

    
    def process_and_stream():
        try:
            socketio.emit('progress', {'message': 'Starting YOLO processing...'})            
        # YOLO 분석 완료 후 결과 처리
            detection_Ball_Glove = detection_model.track(video_path, save=False)
            detection_results3 = detection_model1.track(video_path, save=False)

            # find_foot(detection_results3)
            

            
                



            # YOLO 모델 실행
            # detection_Ball_Glove = detection_model.track(video_path, save=True,callback=progress_callback)

            # 분석 시작
            ball, glove = [], []
            for i, result in enumerate(detection_Ball_Glove):
                frame_number = i + 1
                total_frames = len(detection_Ball_Glove)
                # message = f"Processing frame {frame_number}/{total_frames}"
                # print(message)  # 서버 로그 출력
                # socketio.emit('progress', {'message': message})  # WebSocket으로 전송
                
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


            R0x1, R0y1, R0x2, R0y2, R3x1, R3y1, R3x2, R3y2 = average(detection_results3)
            # print(R0x1, R0y1, R0x2, R0y2, R3x1, R3y1, R3x2, R3y2)
            result_dict = {}

            for (i, result) in enumerate(detection_results3):
                i_dict = {}
                img = result.orig_img  # YOLO 결과 원본 이미지
                annotated_image = img.copy()
                print(i, result.boxes.cls)
                num_base = sum(1 for obj in result.boxes if obj.cls == 1.)
                Two_base = False
                Two_base_first = True
                if num_base >1:
                    Two_base = True
                num_glove = sum(1 for obj in result.boxes if obj.cls == 8.)
                Two_glove = False
                Two_glove_first = True
                if num_glove >1:
                    Two_glove = True
                # num_player = sum(1 for obj in result.boxes if obj.cls == 0. or obj.cls == 3.)
                num_Runner = sum(1 for obj in result.boxes if obj.cls == 0.)
                num_Baseman = sum(1 for obj in result.boxes if obj.cls == 3.)
                over_1_runner = False
                cos_list = []
                obj_list = []
                if num_Runner >1 and num_Baseman >= 1:
                    over_1_runner = True
                    for obj in result.boxes:
                        x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
                        
                        vec = np.array([x1, y1, x2, y2])
                        vec0B = np.array([R0x1, R0y1, R0x2, R0y2])
                        cos0 = cosx(vec, vec0B)
                        cos_list.append(cos0)
                        obj_list.append(obj)
                    max_cos_index = cos_list.index(max(cos_list))
                    obj = obj_list[max_cos_index]





                count = 0
                for obj in (result.boxes):
                    if over_1_runner:
                        x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
                        vec = np.array([x1, y1, x2, y2])
                        vec0B = np.array([R0x1, R0y1, R0x2, R0y2])
                        cos0 = cosx(vec, vec0B)
                        if cos0 != max(cos_list):
                            continue
                        # ----------------------------------

                    if obj.cls == 0. or obj.cls == 3.:
                        x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
                        vec = np.array([x1, y1, x2, y2])
                        vec0B = np.array([R0x1, R0y1, R0x2, R0y2])
                        vec3B = np.array([R3x1, R3y1, R3x2, R3y2])
                        cos0 = dot(vec, vec0B)/(norm(vec)*norm(vec0B))
                        cos3 = dot(vec, vec3B)/(norm(vec)*norm(vec3B))

                        if cos0 < cos3  : 
                        # if obj.cls == 3.:
                            cv2.putText(annotated_image, "Baseman", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            cropped_person_image = img[y1:y2, x1:x2]

                            i_dict[count]=(3, (x1, y1, x2, y2))
                            count += 1
                            # cv2.imwrite(f"pose_results/cropped_person_image{i}.jpg", cropped_person_image)
                            cropped_rgb = cv2.cvtColor(cropped_person_image, cv2.COLOR_BGR2RGB)
                            result_pose = pose.process(cropped_rgb)
                            # MediaPipe의 포즈 추정 결과를 이미지에 그리기
                            mp.solutions.drawing_utils.draw_landmarks(
                                annotated_image[y1:y2, x1:x2], result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        elif cos0 > cos3 :
                        # if obj.cls == 0.:  # 클래스가 0인 경우에만 실행
                            
                            
                            x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
                            
                            
                            vecOR = np.array([x1, y1, x2, y2])
                            vecRB = np.array([R0x1, R0y1, R0x2, R0y2])
                            cos = dot(vecOR, vecRB)/(norm(vecOR)*norm(vecRB))
                            if cos > 0.99:
                                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
                            else:
                                print("cos : ", cos)
                                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255,255,255), 2)
                                # cv2.putText(annotated_image, "correct", (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            i_dict[count]=(0, (x1, y1, x2, y2))
                            count += 1
                            
                            cropped_person_image = img[y1:y2, x1:x2]
                            # cv2.imwrite(f"pose_results/cropped_lg{i}.jpg", cropped_person_image)

                            cropped_rgb = cv2.cvtColor(cropped_person_image, cv2.COLOR_BGR2RGB)
                            result_pose = pose.process(cropped_rgb)

                            if result_pose.pose_landmarks:
                                landmark_27 = result_pose.pose_landmarks.landmark[27]
                                landmark_29 = result_pose.pose_landmarks.landmark[29]
                                landmark_31 = result_pose.pose_landmarks.landmark[31]
                                landmark_28 = result_pose.pose_landmarks.landmark[28]
                                landmark_30 = result_pose.pose_landmarks.landmark[30]
                                landmark_32 = result_pose.pose_landmarks.landmark[32]

                                x27, y27 = int(landmark_27.x * (x2 - x1)) + x1, int(landmark_27.y * (y2 - y1)) + y1
                                x29, y29 = int(landmark_29.x * (x2 - x1)) + x1, int(landmark_29.y * (y2 - y1)) + y1
                                x31, y31 = int(landmark_31.x * (x2 - x1)) + x1, int(landmark_31.y * (y2 - y1)) + y1
                                x28, y28 = int(landmark_28.x * (x2 - x1)) + x1, int(landmark_28.y * (y2 - y1)) + y1
                                x30, y30 = int(landmark_30.x * (x2 - x1)) + x1, int(landmark_30.y * (y2 - y1)) + y1
                                x32, y32 = int(landmark_32.x * (x2 - x1)) + x1, int(landmark_32.y * (y2 - y1)) + y1
                                
                                
                                cv2.line(annotated_image, (x27, y27), (x29, y29), (0, 255, 0), 2)
                                cv2.line(annotated_image, (x29, y29), (x31, y31), (0, 255, 0), 2)
                                cv2.line(annotated_image, (x31, y31), (x27, y27), (0, 255, 0), 2)

                                cv2.line(annotated_image, (x28, y28), (x30, y30), (0, 255, 0), 2)
                                cv2.line(annotated_image, (x30, y30), (x32, y32), (0, 255, 0), 2)
                                cv2.line(annotated_image, (x32, y32), (x28, y28), (0, 255, 0), 2)
                                
                                for idx in range(27, 33):
                                    landmark = result_pose.pose_landmarks.landmark[idx]
                                    x, y = int(landmark.x * (x2 - x1)) + x1, int(landmark.y * (y2 - y1)) + y1
                                    cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)


                                triangle1 = [(x27, y27), (x29, y29), (x31, y31)]
                                triangle2 = [(x28, y28), (x30, y30), (x32, y32)]


                    
                    # if (obj.cls == 8.) or (obj.cls == 3.) or (obj.cls == 2.) or (obj.cls == 1.) or (obj.cls == 0.):
                        # if Two_glove and Two_glove_first:
                        #     # pass
                        #     Bx1, By1, Bx2, By2 = map(int, obj.xyxy[0].tolist())
                        #     Two_base_first = False
                        # elif Two_glove and not Two_glove_first:
                        #     Ax1, Ay1, Ax2, Ay2 = Bx1, By1, Bx2, By2
                        #     Bx1, By1, Bx2, By2 = map(int, obj.xyxy[0].tolist())
                        #     # 두 개의 사각형을 포함하는 사각형 계산
                        #     Cx1, Cy1 = min(Ax1, Bx1), min(Ay1, By1)
                        #     Cx2, Cy2 = max(Ax2, Bx2), max(Ay2, By2)
                        #     cv2.rectangle(annotated_image, (Cx1, Cy1), (Cx2, Cy2), (0, 255, 255), 2)

                        # elif not Two_glove:
                        #     Bx1, By1, Bx2, By2 = map(int, obj.xyxy[0].tolist())
                        #     cv2.rectangle(annotated_image, (Bx1, By1), (Bx2, By2), (0, 255, 0), 2)

                        # center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        # cv2.putText(annotated_image, f"{obj.cls.item():.0f}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        # x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
                        # cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    if obj.cls == 8.:
                        x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
                        cv2.putText(annotated_image, "Glove", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 0), 2)
                    
                    
                    if obj.cls == 1.:
                        if Two_base and Two_base_first:
                            # pass
                            Bx1, By1, Bx2, By2 = map(int, obj.xyxy[0].tolist())
                            Two_base_first = False
                        elif Two_base and not Two_base_first:
                            Ax1, Ay1, Ax2, Ay2 = Bx1, By1, Bx2, By2
                            Bx1, By1, Bx2, By2 = map(int, obj.xyxy[0].tolist())
                            # 두 개의 사각형을 포함하는 사각형 계산
                            Bx1, By1 = min(Ax1, Bx1), min(Ay1, By1)
                            Bx2, By2 = max(Ax2, Bx2), max(Ay2, By2)
                            cv2.putText(annotated_image, "base", (Bx1, By1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            cv2.rectangle(annotated_image, (Bx1, By1), (Bx2, By2), (0, 255, 255), 2)

                        elif not Two_base:
                            Bx1, By1, Bx2, By2 = map(int, obj.xyxy[0].tolist())
                            cv2.putText(annotated_image, "Base", (Bx1, By1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            cv2.rectangle(annotated_image, (Bx1, By1), (Bx2, By2), (0, 255, 255), 2)


                    

                result_dict[i] = i_dict
                            # 삼각형 선분이 직사각형 경계와 교차하는지 확인
                if is_triangle_line_in_box(triangle1, (Bx1, By1, Bx2, By2)) or is_triangle_line_in_box(triangle2, (Bx1, By1, Bx2, By2)):
                    print("베이스를 밟았습니다")
                    cv2.putText(annotated_image, "On base", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                
                frame_image_path = os.path.join(UPLOAD_FOLDER, 'result_frame.jpg')
                if i == abs(answer):
                    # output_path = os.path.join(output_folder, f"pose_detection_{i}.jpg")
                    # cv2.imwrite(output_path, annotated_image)
                    # img.show()
                    success = cv2.imwrite(frame_image_path, annotated_image)
                    print("answer : ", answer)
                    answer = abs(answer)
                    if success:
                        print(f"Image saved successfully at: {frame_image_path}")
                    else:
                        print("Failed to save the image.")
                        return jsonify({'error': 'Failed to save the image.'}), 500







            # img = detection_Ball_Glove[answer].orig_img  # YOLO 결과 원본 이미지
            # annotated_image = img.copy()
            # # img.show()
            # success = cv2.imwrite(frame_image_path, annotated_image)
            # print("answer : ", answer)
            # answer = abs(answer)
            # # if success:
            # #     print(f"Image saved successfully at: {frame_image_path}")
            # # else:
            # #     print("Failed to save the image.")
            # #     return jsonify({'error': 'Failed to save the image.'}), 500
            socketio.emit('progress', {'message': 'Processing complete!'})
            socketio.emit('result', {
                'frame': answer,
                'image_url': '/uploads/result_frame.jpg'
                # 'image_url': f'/uploads/pose_detection_{answer}.png'
            })

        # 비동기로 YOLO 처리 시작
        
        except Exception as e:
            error_message = f"Error during processing: {str(e)}"
            print(error_message)
            socketio.emit('progress', {'message': error_message})
        # 비동기 작업 시작 메시지 반환
    socketio.start_background_task(process_and_stream)
    return jsonify({'message': 'Video processing started'})

    
    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    full_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}")
        return jsonify({'error': 'File not found'}), 404
    print(f"Serving file from: {full_path}")
    return send_from_directory(UPLOAD_FOLDER, filename)

@socketio.on('connect')
def handle_connect():
    print("Client connected")


@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")


if __name__ == '__main__':
    app.run(port=8081)






# import subprocess
# from flask import Flask, request, jsonify, render_template, send_from_directory
# from flask_socketio import SocketIO, emit
# from ultralytics import YOLO
# import os, cv2
# import numpy as np
# from sklearn.cluster import HDBSCAN
# from collections import Counter
# from math import sqrt
# from sklearn.linear_model import LinearRegression

# app = Flask(__name__)
# socketio = SocketIO(app, cors_allowed_origins="*")

# # 폴더 설정
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # YOLO 모델 로드
# MODEL_PATH = "/Users/kwonsebin/Documents/khu/4-1/capstone/runs/detect/train13/weights/best.pt"
# detection_model = YOLO(MODEL_PATH)

# # 루트 경로 - index.html 반환
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/process', methods=['POST'])
# def process_video():
#     if 'video' not in request.files:
#         return jsonify({'error': 'No video file provided'}), 400

#     # 동영상 파일 저장
#     video_file = request.files['video']
#     video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
#     video_file.save(video_path)

    
#     def process_and_stream():
#         try:
#             socketio.emit('progress', {'message': 'Starting YOLO processing...'})            
#         # YOLO 분석 완료 후 결과 처리
#             detection_Ball_Glove = detection_model.track(video_path, save=False)
#             # YOLO 모델 실행
#             # detection_Ball_Glove = detection_model.track(video_path, save=True,callback=progress_callback)

#             # 분석 시작
#             ball, glove = [], []
#             for i, result in enumerate(detection_Ball_Glove):
#                 frame_number = i + 1
#                 total_frames = len(detection_Ball_Glove)
#                 # message = f"Processing frame {frame_number}/{total_frames}"
#                 # print(message)  # 서버 로그 출력
#                 # socketio.emit('progress', {'message': message})  # WebSocket으로 전송
                
#                 for obj in result.boxes:
#                     if obj.cls == 2.0:  # 공
#                         x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
#                         ball.append([x1, y1])
#                     if obj.cls == 8.0:  # 글러브
#                         x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
#                         glove.append([x1, y1])

#             # HDBSCAN 클러스터링
#             # hdbBall = HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=10)
#             # hdbBall.fit(ball)
#             # hdbGlove = HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=10)
#             # hdbGlove.fit(glove)

#             # most_common_ball = Counter(hdbBall.labels_).most_common(1)[0][0]
#             # most_common_glove = Counter(hdbGlove.labels_).most_common(1)[0][0]

#             # 각 프레임 정보 저장
#             frame = {}
#             for i, result in enumerate(detection_Ball_Glove):
#                 info = {}
#                 for obj in result.boxes:
#                     if obj.cls == 2.0:  # 공
#                         x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
#                         # if hdbBall.labels_[ball.index([x1, y1])] == most_common_ball:
#                             # info["ball"] = (x1, y1, x2, y2)
#                         info["ball"] = (x1, y1, x2, y2)
#                     if obj.cls == 8.0:  # 글러브
#                         x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
#                         # if hdbGlove.labels_[glove.index([x1, y1])] == most_common_glove:
#                             # info["glove"] = (x1, y1, x2, y2)
#                         info["glove"] = (x1, y1, x2, y2)
#                 frame[i] = info

#             # 거리 계산
#             distances = {}
#             for frame_number, objects in frame.items():
#                 if "ball" in objects and "glove" in objects:
#                     ball_coords = objects["ball"]
#                     glove_coords = objects["glove"]
#                     ball_center = ((ball_coords[0] + ball_coords[2]) / 2, (ball_coords[1] + ball_coords[3]) / 2)
#                     glove_center = ((glove_coords[0] + glove_coords[2]) / 2, (glove_coords[1] + glove_coords[3]) / 2)
#                     distance = sqrt((ball_center[0] - glove_center[0]) ** 2 + (ball_center[1] - glove_center[1]) ** 2)
#                     distances[frame_number] = distance

#             # 선형 회귀
#             X = np.array(list(distances.keys())).reshape(-1, 1)
#             y = np.array(list(distances.values()))
#             model = LinearRegression()
#             model.fit(X, y)

#             answer = int(-model.intercept_ / model.coef_[0])
#             # answer = 50

#             # 특정 프레임 저장
#             frame_image_path = os.path.join(UPLOAD_FOLDER, 'result_frame.jpg')


#             img = detection_Ball_Glove[answer].orig_img  # YOLO 결과 원본 이미지
#             annotated_image = img.copy()
#             # img.show()
#             success = cv2.imwrite(frame_image_path, annotated_image)
#             # if success:
#             #     print(f"Image saved successfully at: {frame_image_path}")
#             # else:
#             #     print("Failed to save the image.")
#             #     return jsonify({'error': 'Failed to save the image.'}), 500
#             socketio.emit('progress', {'message': 'Processing complete!'})
#             socketio.emit('result', {
#                 'frame': answer,
#                 'image_url': '/uploads/result_frame.jpg'
#             })

#         # 비동기로 YOLO 처리 시작
        
#         except Exception as e:
#             error_message = f"Error during processing: {str(e)}"
#             print(error_message)
#             socketio.emit('progress', {'message': error_message})
#         # 비동기 작업 시작 메시지 반환
#     socketio.start_background_task(process_and_stream)
#     return jsonify({'message': 'Video processing started'})

    
    
# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     full_path = os.path.join(UPLOAD_FOLDER, filename)
#     if not os.path.exists(full_path):
#         print(f"File not found: {full_path}")
#         return jsonify({'error': 'File not found'}), 404
#     print(f"Serving file from: {full_path}")
#     return send_from_directory(UPLOAD_FOLDER, filename)

# @socketio.on('connect')
# def handle_connect():
#     print("Client connected")


# @socketio.on('disconnect')
# def handle_disconnect():
#     print("Client disconnected")


# if __name__ == '__main__':
#     app.run(port=8081)
