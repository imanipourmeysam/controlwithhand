import cv2 as cv 
import mediapipe as mp
import pyautogui as mouse
from scipy.interpolate import interp1d

MP_DrawingModule = mp.solutions.drawing_utils
MP_HandModule = mp.solutions.hands

class HandDetector:

    def __init__(self, model_complexity = 1, max_num_hands = 2, min_detection_confidence = 0.5, min_tracking_confidence = 0.5):
            self.hands = MP_HandModule.Hands(model_complexity = model_complexity, max_num_hands = max_num_hands, min_detection_confidence = min_detection_confidence,
                                             min_tracking_confidence = min_tracking_confidence)

    def getHandPoints(self, frame, handnumber = 0, draw = False):
        result = self.hands.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB))
        PointList = []
        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[handnumber]
            for id, landmark in enumerate(landmarks.landmark):
                mirror = interp1d([1.2, -0.001], [-0.001, 1.2])
                xcoor, ycoor = mirror(landmark.x) * 1920, landmark.y * 1080 
                PointList.append([id, xcoor, ycoor])
            if  draw:
                MP_DrawingModule.draw_landmarks(frame, landmarks, MP_HandModule.HAND_CONNECTIONS)
        return PointList


cap = cv.VideoCapture(0)

handobj = HandDetector()

ans   = 0
horef = 0
justonce = True
#while cap.isOpened():
while True:
    success, frame = cap.read()
    cv.imshow("test", frame)    
    if cv.waitKey(1) == 27:
        break
    handpoints = handobj.getHandPoints(frame, draw = False)
    cur_x, cur_y = mouse.position()
    if  handpoints:
        if justonce:
            mouse.moveTo(handpoints[8][1], handpoints[8][2])
            justonce = False;
        if(abs(handpoints[8][1] - cur_x) > 1 and abs(handpoints[8][2] - cur_y) > 1):  
            mouse.moveTo(handpoints[8][1], handpoints[8][2])
            dis = handpoints[5][1] - handpoints[4][1]
            if dis < 120:
                mouse.click()
                print("clicked")

cv.destroyAllWindows()
cap.release()

