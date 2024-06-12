import cv2
import numpy as np

img = cv2.imread("c5.jpg")
inst = cv2.imread("instruct.jpg")

hsvFrame = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

red_lower1=np.array([0,100,20])
red_upper1 = np.array([8,255,255])

red_lower2=np.array([171,100,20])
red_upper2 = np.array([179,255,255])
red_mask1=cv2.inRange(hsvFrame,red_lower1,red_upper1)
red_mask2 = cv2.inRange(hsvFrame,red_lower2,red_upper2)
red_mask=red_mask1+red_mask2

orange_lower = np.array([10,100,20],np.uint8)
orange_upper = np.array([22,255,255],np.uint8)
orange_mask = cv2.inRange(hsvFrame,orange_lower,orange_upper)

black_lower = np.array([0,0,0],np.uint8)
black_upper = np.array([180,255,30],np.uint8)
black_mask = cv2.inRange(hsvFrame,black_lower,black_upper)

white_lower = np.array([0,0,200],np.uint8)
white_upper = np.array([180,255,255],np.uint8)
white_mask = cv2.inRange(hsvFrame,white_lower,white_upper)

grey_lower = np.array([0,0,31],np.uint8)
grey_upper = np.array([0,0,199],np.uint8)
grey_mask = cv2.inRange(hsvFrame,grey_lower,grey_upper)

blue_lower = np.array([85,50,20],np.uint8)
blue_upper = np.array([128,255,255],np.uint8)
blue_mask = cv2.inRange(hsvFrame,blue_lower,blue_upper)

yellow_lower = np.array([23,100,20],np.uint8)
yellow_upper = np.array([34,255,255],np.uint8)
yellow_mask = cv2.inRange(hsvFrame,yellow_lower,yellow_upper)

green_lower = np.array([35,100,20],np.uint8)
green_upper = np.array([84,255,255],np.uint8)
green_mask = cv2.inRange(hsvFrame,green_lower,green_upper)

purple_lower = np.array([130,100,20],np.uint8)
purple_upper = np.array([141,255,255],np.uint8)
purple_mask = cv2.inRange(hsvFrame,purple_lower,purple_upper)

pink_lower = np.array([142,100,20],np.uint8)
pink_upper = np.array([170,255,255],np.uint8)
pink_mask = cv2.inRange(hsvFrame,pink_lower,pink_upper)


cv2.imshow("Original",img)
cv2.imshow("Instruction",inst)

while True:
    key=cv2.waitKey(1)
    if key ==ord('r'):
        output = cv2.bitwise_and(img, img, mask=red_mask)
        cv2.imshow("Red",output)
        output2 = img.copy()
        contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                output2 = cv2.rectangle(output2, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(output2, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        cv2.imshow("Red2",output2)
        cv2.waitKey(0)
        cv2.destroyWindow("Red")
        cv2.destroyWindow("Red2")

    if key == ord('o'):
        output = cv2.bitwise_and(img,img,mask=orange_mask)
        cv2.imshow("Orange",output)
        output2 = img.copy()
        contours, hierarchy = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                output2 = cv2.rectangle(output2, (x, y), (x + w, y + h), (0, 128, 255), 2)
                cv2.putText(output2, "Orange Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255))

        cv2.imshow("Orange2", output2)
        cv2.waitKey(0)
        cv2.destroyWindow("Orange")
        cv2.destroyWindow("Orange2")

    if key == ord('y'):
        output = cv2.bitwise_and(img,img,mask=yellow_mask)
        cv2.imshow("Yellow",output)
        output2 = img.copy()
        contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                output2 = cv2.rectangle(output2, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(output2, "Yellow Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 204, 204))

        cv2.imshow("Yellow2",output2)
        cv2.waitKey(0)
        cv2.destroyWindow("Yellow")
        cv2.destroyWindow("Yellow2")

    if key == ord('g'):
        output = cv2.bitwise_and(img,img,mask=green_mask)
        cv2.imshow("Green",output)
        output2 = img.copy()
        contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                output2 = cv2.rectangle(output2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(output2, "Green Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.imshow("Green2",output2)
        cv2.waitKey(0)
        cv2.destroyWindow("Green")
        cv2.destroyWindow("Green2")

    if key == ord('b'):
        output = cv2.bitwise_and(img,img,mask=blue_mask)
        cv2.imshow("Blue",output)
        output2 = img.copy()
        contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                output2 = cv2.rectangle(output2, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(output2, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        cv2.imshow("Blue2",output2)
        cv2.waitKey(0)
        cv2.destroyWindow("Blue")
        cv2.destroyWindow("Blue2")

    if key == ord('p'):
        output = cv2.bitwise_and(img,img,mask=purple_mask)
        cv2.imshow("Purple",output)
        output2 = img.copy()
        contours, hierarchy = cv2.findContours(purple_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                output2 = cv2.rectangle(output2, (x, y), (x + w, y + h), (255, 51, 153), 2)
                cv2.putText(output2, "Purple Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 51, 153))
        cv2.imshow("Purple2",output2)
        cv2.waitKey(0)
        cv2.destroyWindow("Purple")
        cv2.destroyWindow("Purple2")

    if key == ord('k'):
        output = cv2.bitwise_and(img,img,mask=pink_mask)
        cv2.imshow("Pink",output)
        output2 = img.copy()
        contours, hierarchy = cv2.findContours(pink_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                output2 = cv2.rectangle(output2, (x, y), (x + w, y + h), (255, 102, 255), 2)
                cv2.putText(output2, "Pink Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 102, 255))
        cv2.imshow("Pink2",output2)
        cv2.waitKey(0)
        cv2.destroyWindow("Pink")
        cv2.destroyWindow("Pink2")

    if key == ord('w'):
        output = cv2.bitwise_and(img,img,mask=white_mask)
        cv2.imshow("White",output)
        output2 = img.copy()
        contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                output2 = cv2.rectangle(output2, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(output2, "White Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv2.imshow("White2",output2)
        cv2.waitKey(0)
        cv2.destroyWindow("White")
        cv2.destroyWindow("White2")

    if key == ord('h'):
        output = cv2.bitwise_and(img,img,mask=black_mask)
        cv2.imshow("Black",output)
        output2 = img.copy()
        contours, hierarchy = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                output2 = cv2.rectangle(output2, (x, y), (x + w, y + h), (0, 0, 0), 2)
                cv2.putText(output2, "Black Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv2.imshow("Black2",output2)
        cv2.waitKey(0)
        cv2.destroyWindow("Black")
        cv2.destroyWindow("Black2")

    if key == ord('l'):
        output = cv2.bitwise_and(img,img,mask=grey_mask)
        cv2.imshow("Grey",output)
        output2 = img.copy()
        contours, hierarchy = cv2.findContours(grey_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                output2 = cv2.rectangle(output2, (x, y), (x + w, y + h), (128, 128, 128), 2)
                cv2.putText(output2, "Grey Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv2.imshow("Grey2",output2)
        cv2.waitKey(0)
        cv2.destroyWindow("Grey")
        cv2.destroyWindow("Grey2")

    if key == ord('q'):
        break

cv2.destroyAllWindows()

