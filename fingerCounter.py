import cv2
import os
import time
import handTracmodule as htm

cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

ptime= 0
path= "FingerImages"
myList =os.listdir(path)

overlayLis=[]
for imPath in sorted(myList):
    image = cv2.imread(f'{path}/{imPath}')
    overlayLis.append(image)

detector=htm.handDetector(MaxHands=1)
tipIDs =[4,8,12,16,20]

while True:
    success, img = cap.read()
    img=detector.findHands(img)
    lmlist =detector.findPosition(img, draw=False)


    if len(lmlist)!=0:
        fingers=[]

       # print(lmlist[4][1], lmlist[8][1])

        #thumb
        if lmlist[tipIDs[0]][1] >  lmlist[tipIDs[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #fingers
        for id in range(1,5):
            if lmlist[tipIDs[id]][2] < lmlist[tipIDs[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        total_fingers=fingers.count(1)
        print(total_fingers)

        h,w,c = overlayLis[total_fingers ].shape
        img[0:h, 0:w] = overlayLis[total_fingers]
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime

    cv2.putText(img,f'FPS: {int(fps)}', (10,50), cv2.FONT_HERSHEY_PLAIN, 3, (126,32,25), 2)

    cv2.imshow("video",img)
    cv2.waitKey(1)