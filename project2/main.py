import cvzone
import cv2
import numpy as np
import math
import random
from cvzone.HandTrackingModule import HandDetector


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=(0.8),maxHands=1)

class SnakeGameClass:
    def __init__(self,pathFood):
        #Khai bao cac attributes khac nhau cua object
        self.points = []  #all points of the snake
        self.lengths = [] #distance between each point
        self.currentLength = 0 #total length of the snake
        self.allowedLength = 150 #total allowed Length
        self.previousHead = 0, 0 #pevious head point

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.imgFood = cv2.resize(self.imgFood,(75,75))

        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()

        self.score = 0
        self.gameOver = False


    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)


    def update(self,imgMain, currentHead):

        if self.gameOver:
            cvzone.putTextRect(imgMain, "Game Over", [300, 400],
                               scale= 7,thickness= 5, offset= 20)
            cvzone.putTextRect(imgMain, f'Score: {self.score}', [300, 550],
                               scale= 7,thickness= 5, offset= 20)
        else:
            px, py = self.previousHead
            cx, cy = currentHead

            self.points.append([cx, cy]) # add element into final list
            distance = math.hypot(cx - px, cy - py)
            self.lengths.append(distance)
            self.currentLength += distance
            self.previousHead = cx, cy

            #Length Reduction
            if self.currentLength > self.allowedLength:
                for i, length in enumerate(self.lengths):
                    self.currentLength -= length
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.currentLength < self.allowedLength:
                        break

            #check if snake  ate the Food
            rx,ry = self.foodPoint
            if rx - self.wFood//2 <cx<rx + self.wFood//2   and\
                    ry - self.hFood//2<cy<ry+ self.hFood//2 :
                self.randomFoodLocation()
                self.allowedLength += 50
                self.score +=1
                print(self.score)

            #Draw snake
            if self.points:
                for i, point in enumerate(self.points):
                    if i != 0:
                            cv2.line(imgMain, self.points[i-1], self.points[i], (0, 0 ,255), 10)
                cv2.circle(imgMain, self.points[-1], 20, (200, 0, 200), cv2.FILLED)  # Vẽ đầu rắn màu tím

            #Draw Food
            imgMain = cvzone.overlayPNG(imgMain, self.imgFood,
                                        (rx - self.wFood // 2, ry - self.hFood // 2)) # hàm của thư viện cvzone để chèn ảnh PNG có nền trong suốt
            cvzone.putTextRect(imgMain, f'Score: {self.score}', [50, 80],
                               scale= 3,thickness= 3, offset= 10)  # giúp viết chữ lên ảnh kèm theo khung chữ nhật nền

            #check for Collision
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(imgMain, [pts], False, (0,200,0), 3)
            minDist =  cv2.pointPolygonTest(pts,(cx,cy),True)

            if -1 <= minDist <= 1:
                self.gameOver = True
                self.points = []  # all points of the snake
                self.lengths = []  # distance between each point
                self.currentLength = 0  # total length of the snake
                self.allowedLength = 200  # total allowed Length
                self.previousHead = 0, 0  # pevious head point
                self.randomFoodLocation()



        return imgMain
game = SnakeGameClass("venv/food6.png")
while True:
    success, img = cap.read() #doc img tu camera

    img = cv2.flip(img,1)
    hands, img = detector.findHands(img, flipType= False)  #hàm trong thư viện Mediapipe/ phat hien hand

    if hands:
        lmList = hands[0]['lmList']
        poinIndex = lmList[8][0:2] #set toa do diem cho ngon tro
        img = game.update(img, poinIndex)
    cv2.imshow("image",img)
    key  = cv2.waitKey(1)
    if key == ord('r'):
        game.gameOver = False
        game.score = 0
