import cv2
import numpy as np
import time
import handTracmodule as htm
import math
import osascript

cap = cv2.VideoCapture(0)
ptime=0

###############
wcam , hcam =1280, 720
################

min_vol=0
max_vol=100
vol=0
volbar =0
cap.set(3,wcam)
cap.set(4,hcam)
detector =htm.handDetector(min_detec_confi=0.5 ,MaxHands=1)

while True:
    success , img = cap.read()
    img=(detector.findHands(img))
    lmlist =detector.findPosition(img , draw=False)
    if len(lmlist)!=0:
        cv2.circle(img , (lmlist[4][1],lmlist[4][2]) ,15 ,(25,25,25), cv2.FILLED)
        cv2.circle(img, (lmlist[8][1], lmlist[8][2]), 15, (25, 25, 25), cv2.FILLED)
        cv2.circle(img, ((lmlist[4][1]+lmlist[8][1])//2, (lmlist[4][2]+lmlist[8][2])//2), 10, (25, 25, 25), cv2.FILLED)
        cv2.line(img,(lmlist[4][1],lmlist[4][2]), (lmlist[8][1],lmlist[8][2]), (25,25,25),3  )

        length = math.hypot(lmlist[8][1] - lmlist[4][1]  , lmlist[8][2]- lmlist[4][2])
        #print(length)
        if length<50 or length>350:
            cv2.circle(img, ((lmlist[4][1] + lmlist[8][1]) // 2, (lmlist[4][2] + lmlist[8][2]) // 2), 10, (255, 25, 25),
                       cv2.FILLED)

        vol = np.interp(length, [50, 350], [min_vol, max_vol])
        volbar = np.interp(length, [50, 350], [400,100])

        cv2.rectangle(img, (50,100), (85, 400) ,(299,0,0) , 3)
        cv2.rectangle(img, (50, int(volbar)), (85, 400), (299, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(vol)}%', (40,50), cv2.FONT_HERSHEY_PLAIN ,3,(299,0,0) ,3)

        vol1 = "set volume output volume " + str(vol)
        osascript.osascript(vol1)


    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    # for lm in lmlist:
    #     cv2.putText(img , str(lm[0]),(lm[1],lm[2]),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)

    cv2.putText(img, str(int(fps)), (10,70),cv2.FONT_HERSHEY_PLAIN, 3, (23,23,23), 2)

    cv2.imshow("video", img)
    cv2.waitKey(1)