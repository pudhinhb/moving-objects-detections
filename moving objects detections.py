import cv2 #open cv
import imutils #resize

cam=cv2.VideoCapture(0) #initialize cam id
firstframe=None
area=500

while True:
    _,img = cam.read() #Read from cam
    text="Normal"

    img = imutils.resize(img, width=500)#resize

    grayImg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #color to grayscale img

    gaussianImg = cv2.GaussianBlur (grayImg,(21,21),0) #smoothend

    if firstframe is None:
        firstframe = gaussianImg #capturing firstframe
        continue
    imgDiff= cv2.absdiff(firstframe, gaussianImg) #absolute diffference

    threshImg = cv2.threshold(imgDiff, 25,255, cv2. THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations =2) #left overs erotion

    cnts= cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #make complete contour

    cnts= imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < area: #to cover all area
            continue
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle (img,(x,y), (x+w, y+h),(0,255,0),2)
        text = "Moving object detected"

    print (text)
    cv2.putText(img, text, (10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    cv2.imshow("cameraFeed",img)

    key= cv2.waitKey(10)
    print(key)
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()