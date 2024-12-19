주제
-> 1루 베이스 포스 아웃/세이프 상황에 대한 비디오 판독 자동화

YOLO V11을 통한 데이터 학습
['Runner', 'Base', 'Baseball_ball', 'Fielder1', 'Home_base', 'Fielder2', 'Umpire1', 'Umpire2', 'Glove'] 기준으로 라벨링 되어있는 데이터 활용
학습에 활용하는 데이터 출처 https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71726

기능 요약

1. 서버를 통해 영상 입력 (영상은 공과 주자 등 움직임을 명확하게 인식할 수 있어야 함)
2. YOLO V11 통해 학습된 모델 1로 공과 글러브 인식, 이후 포구 시점 파악
3. YOLO V11 통해 학습된 모델 2로 베이스와 주자 인식, 이후 MediaPipe를 통해 주자의 발 좌표 인식
4. 포구시점 시 주자의 발과 베이스 위치를 파악, 주자가 베이스를 밟았으면 safe, 밟지 못했으면 out으로 판정 가능

사용 모델 및 언어
YOLO V11, MediaPipe
Flask
Python


<h3>Example Video</h3>
<iframe width="560" height="315" 
    src="https://youtu.be/YuXrAXy7xXE" 
    frameborder="0" allowfullscreen>
</iframe>

