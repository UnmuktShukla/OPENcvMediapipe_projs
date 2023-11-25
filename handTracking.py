import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw =mp.solutions.drawing_utils

cTime=0
pTime=0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks :
        for HandLMS in results.multi_hand_landmarks:
            for id , lm in enumerate(HandLMS.landmark):
                #print(id , lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)


                #if id==5:
                #    cv2.putText(img, "hello", (cx,cy), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 255), 3)
                    #cv2.circle(img, (cx,cy) ,5,(255,0,255),cv2.FILLED)


            mpDraw.draw_landmarks(img , HandLMS, mpHands.HAND_CONNECTIONS)
    cTime=time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img, str(int(fps)), (10,70) , cv2.FONT_HERSHEY_DUPLEX , 3, (255,0,255) ,3 )
    cv2.imshow("image",img)
    cv2.waitKey(1)
