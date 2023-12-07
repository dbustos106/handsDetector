import mediapipe as mp
import math
import cv2

class HandsDetector():

    def __init__(self, mode=False, maxHands=2, modelComplexity=1, confDetection=0.5, confTracking=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.confDetection = confDetection
        self.confTracking = confTracking

        self.model = mp.solutions.hands
        self.modelHands = self.model.Hands(self.mode, self.maxHands, self.modelComplexity, self.confDetection, self.confTracking)
        self.drawing = mp.solutions.drawing_utils
        self.tip = [4, 8, 12, 16, 20]

    def findHands(self, frame, draw=True):
        frameColor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.modelHands.process(frameColor)

        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.drawing.draw_landmarks(frame, hand, self.model.HAND_CONNECTIONS)

        return frame

    def findPosition(self, frame, numHand=0, drawPoints=True, drawBox=True, color=[]):
        self.listPoints = []
        xListPoints = []
        yListPoints = []

        bbox = ()
        numHands = 0
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[numHand]
            numHands = len(self.results.multi_hand_landmarks)

            for id, lm in enumerate(myHand.landmark):
                height, width, _ = frame.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                xListPoints.append(cx)
                yListPoints.append(cy)
                self.listPoints.append([id, cx, cy])

                if drawPoints:
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 0), cv2.FILLED)
                
            xmin, xmax = min(xListPoints) - 40, max(xListPoints) + 40
            ymin, ymax = min(yListPoints) - 40, max(yListPoints) + 40
            bbox = xmin, ymin, xmax, ymax
            if drawBox:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        
        return self.listPoints, bbox, numHands
    
    def fingersUp(self):
        fingers = []

        if self.listPoints[self.tip[0]][1] > self.listPoints[self.tip[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.listPoints[self.tip[id]][2] > self.listPoints[self.tip[id]-1][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def distance(self, p1, p2, frame, draw=True, r=15, t=3):
        x1, y1 = self.listPoints[p1][1:]
        x2, y2 = self.listPoints[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)

        return length, frame, [x1, y1, x2, y2, cx, cy] 


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    detector = HandsDetector()

    while (True):
        ret, frame = cap.read()
        
        frame = detector.findHands(frame)
        lista, bbox, player = detector.findPosition(frame)

        cv2.imshow("Manos", frame)
        k = cv2.waitKey(1)

        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
