import cv2

thres = 0.45
# Threshold to detect object
cap = cv2.VideoCapture(0)
 cap = cv2.VideoCapture(r"C:\Users\meets\Downloads\test2.avi")
cap = cv2.VideoCapture(r"C:\Users\meets\Downloads\videoplayback.mp4")


cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

classNames = []
classFile = r"C:\Users\meets\Downloads\Object_Detection_Files\coco.names"
with open(classFile, 'rt') as f:
    classNames = [line.rstrip() for line in f]

configPath = r"C:\Users\meets\Downloads\Object_Detection_Files\ssd_mobilenet_v3_l arge_coco_2020_01_14.pbtxt"
weightsPath = r"C:\Users\meets\Downloads\Object_Detection_Files\frozen_inference_ graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:

success, img = cap.read()
classIds, confs, bbox = net.detect(img, confThreshold=thres)
print(classIds, bbox)
if len(classIds) != 0:
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox): if
        (classNames[classId - 1] == "person"):
cv2.rectangle(img, box, color=(255, 255, 0), thickness=2)  # print(classNames[classId-1])

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    diff = cv2.absdiff(frame1,
                       frame2)
    # finding the absoulute differece between two pixcels of two image arrays gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # converting to gray scale image blur = cv2.GaussianBlur(gray, (5, 5), 0) # (input,kernal value, sigmax)
_, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
# finding the threshold value
dilated = cv2.dilate(thresh, None,
                     iterations=3)
# fill the empty pixles contours,_=cv2.findContours(dilated,cv2.RETR_TREE, v2.CHAIN_APPROX_SIMPLE) #

for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    if cv2.contourArea(contour) < 1000:
        continue
# cv2.rectangle(frame1, box, color=(255, 255, 0), thickness=2)
cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.putText(frame1, "Status:	{}".format("Movement"), (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 0, 255), 3)
# cv2.drawContours(frame1,contours,-1,(0,255,0),2) cv2.imshow("Feed", img)
cv2.waitKey()

cv2.imshow("Output", frame1)
frame1 = frame2
ret, frame2 = cap.read()

if cv2.waitKey(40) == 27:
    break

'''cv2.putText(img, classNames[classId-1].upper(), (box[0] +10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)



cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)'''

cv2.destroyAllWindows()
cap.release()
