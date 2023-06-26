import cv2

image = cv2.imread('C:\\Users\\Amira\\Desktop\\opencv\\photo\\car2.jpg')

class_name = []
class_file = "C:\\Users\\Amira\\Desktop\\opencv\\file\\thing.txt"

with open(class_file, "rt") as f:
    class_name = f.read().rstrip('\n').split('\n')

p = 'file/frozen_inference_graph.pb'
v = 'file/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

# Scanner les fichiers p et v et nommer les objets dans l'image
net = cv2.dnn_DetectionModel(p, v)
net.setInputSize((320, 230))
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

class_ids, confs, bbox = net.detect(image, confThreshold=0.5)

for class_id, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
    cv2.rectangle(image, box, color=(0, 255, 0), thickness=3)
    cv2.putText(image, class_name[class_id - 1], (box[0] + 10, box[1] + 20),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness=2)

cv2.imshow("programme", image)
cv2.waitKey(0)


