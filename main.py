import numpy as np
import cv2

def detect(input_path, output_path, true_confidence=0.2):
    image = cv2.imread(input_path)
    net = cv2.dnn.readNetFromCaffe("prototxt.txt", "model.caffemodel")

    objects = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    object_color = np.array(([128,128,0], [244,164,96], [255,69,0], [255,99,71], [128,128,128], [255,192,203], [75,0,130], [0,100,0], [189,183,107], [139,69,19], [188,143,143], [220,20,60], [199,21,133], [148,0,211], [0,0,139], [0,0,0], [46,139,87], [255,255,224], [245,222,179], [255,0,0], [255,69,0]))

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > true_confidence:
            obj = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x0, y0, x1, y1) = box.astype("int")

            cv2.rectangle(image, (x0, y0), (x1, y1), tuple([int(x) for x in object_color[obj]]), 2)

            text = "{}: {:.2f}%".format(objects[obj], confidence * 100)
            y = y0 - 15 if y0 - 15 > 15 else y0 + 15
            cv2.putText(image, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple([int(x) for x in object_color[obj]]), 2)

    cv2.imwrite(output_path, image)

def detect_faces(input_path, output_path, true_confidence=0.2):
    image = cv2.imread(input_path)
    net = cv2.dnn.readNetFromCaffe("face.prototxt.txt", "face.caffemodel")
    
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)

    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > true_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x0, y0, x1, y1) = box.astype("int")

            crop_img = image[y0:y1, x0:x1]
            cv2.imwrite("face_"+str(i)+".jpg", crop_img)

            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)

            text = "{:.2f}%".format(confidence * 100)
            y = y0 - 10 if y0 - 10 > 10 else y0 + 10
            cv2.putText(image, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imwrite(output_path, image)

detect("input.jpg", "out_detect.jpg")

detect_faces("input.jpg", "out_face.jpg")
