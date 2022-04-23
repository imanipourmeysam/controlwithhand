import cv2 as cv
import mediapipe as mp

drawingModule = mp.solutions.drawing_utils
handModule = mp.solutions.hands

vidcap = cv.VideoCapture("/dev/vidiieo2")

with handModule.Hands(static_image_mode = False, min_detection_confidence = 0.7, min_tracking_confidence = 0.7) as hands:
    while(True):
        ret_bool, npframe = vidcap.read()
        if(ret_bool != True):
            print("error while captureing video via webcam")
        RGBnpframe = cv.cvtColor(npframe,cv.COLOR_BGR2RGB)
        detected_hands = hands.process(RGBnpframe)
        if detected_hands.multi_hand_landmarks != None:
            for handLandmarks in detected_hands.multi_hand_landmarks:
                drawingModule.draw_landmarks(npframe, handLandmarks, handModule.HAND_CONNECTIONS)
        cv.imshow("test hand", npframe)
        if cv.waitKey(1) == 27:
            break


cv.destroyAllWindows()
vidcap.release()
