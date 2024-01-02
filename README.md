# object-detection-yolov8-python
-copy folder train and change to valid
- train data in CLI using command below
  yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=20 imgsz=416
-python code is for object detection in video.mp4 in real-time.

 Note:
The trained model or the weight output or the best.pt classify 1 object only and it is the name:['king']
So basically this is object detection in chessboard game which detect king only.![output](https://github.com/ljfanxi/object-detection-yolov8-python/assets/61730377/690ed788-7231-410c-b384-9da81aa6bc62)
