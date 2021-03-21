# References
# https://deep-eye.tistory.com/6
# https://bong-sik.tistory.com/16

import cv2
import numpy as np
import pafy


def show_YOLO_detection():
    
    # Youtube 비디오 불러오기

    url = "https://www.youtube.com/watch?v=8DyziWtkfBw&ab_channel=RedHotChiliPeppers"
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")

    VideoSignal = cv2.VideoCapture()
    VideoSignal.open(best.url)

    # YOLO 가중치 파일과 CFG 파일 로드
    YOLO_net = cv2.dnn.readNet("yolov2-tiny.weights","yolov2-tiny.cfg")

    # YOLO NETWORK 재구성
    classes = []
    with open("yolo.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = YOLO_net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in YOLO_net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    cnt = 0

    while True:
        
        # 프레임 받아오기
        ret, frame = VideoSignal.read()
        h, w, c = frame.shape

        # 성능을 위해 프레임 생략
        cnt += 1
        if cnt % 4 != 0:
            continue

        # YOLO 입력
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        YOLO_net.setInput(blob)
        outs = YOLO_net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:

            for detection in out:

                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    dw = int(detection[2] * w)
                    dh = int(detection[3] * h)
                    
                    # Rectangle coordinate
                    x = int(center_x - dw / 2)
                    y = int(center_y - dh / 2)
                    boxes.append([x, y, dw, dh])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)


        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                score = confidences[i]
                color = colors[i]

                # 경계상자와 클래스 정보 이미지에 입력
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, color, 1)

        cv2.imshow("YOLOv3", frame)

        k = cv2.waitKey(100) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindowss()


if __name__ == '__main__':
    show_YOLO_detection()
