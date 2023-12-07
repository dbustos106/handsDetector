import handTracker as ht
import cv2
import os

folder_name = "E"
folder_path = "C:/Users/julia/Desktop/Manos/data" + "/" + folder_name

if not os.path.exists(folder_path):
    print("Carpeta creada", folder_path)
    os.makedirs(folder_path)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cont = 0

detector = ht.HandsDetector(confDetection=0.9)

while(True):
    ret, frame = cap.read()
    frame = detector.findHands(frame, draw=False)
    lista, bbox, numHands = detector.findPosition(frame, numHand=0, drawPoints=False, drawBox=False, color=[0, 255, 0])

    if numHands == 1:
        xmin, ymin, xmax, ymax = bbox
        screenshot = frame[ymin:ymax, xmin:xmax]
        screenshot = cv2.resize(screenshot, (640, 640), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(folder_path + "/E_{}.jpg".format(cont), screenshot)
        cont += 1

    cv2.imshow("Manos", frame)
    k = cv2.waitKey(1)

    if k == 27 or cont == 100:
        break

cap.release()
cv2.destroyAllWindows()