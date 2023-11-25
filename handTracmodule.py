import cv2
import mediapipe as mp
import time
class handDetector():
    def __init__(self, mode=False, MaxHands=2, cmplxt=1, min_detec_confi=0.5, min_track_confi=0.5):
        self.mode = mode
        self.MaxHands = MaxHands
        self.cmplxt = cmplxt
        self.min_detec_confi = min_detec_confi
        self.min_track_confi = min_track_confi

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,
                                        self.MaxHands,
                                        self.cmplxt,
                                        self.min_detec_confi,
                                        self.min_track_confi)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for HandLMS in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(img, HandLMS, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNO=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNO]
            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id,cx,cy])
                if draw:
                    # cv2.putText(img, "hello", (cx,cy), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 255), 3)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return lmlist


def main():
    cTime = 0
    pTime = 0
    detector = handDetector()
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist=  detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
