import cv2

haar_cascade = cv2.CascadeClassifier('haar_face.xml')
people =  ["Anoushka G", "Nihal J", "Noyonika G"]

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("trained_faces2.0.yml")

imgname= "1.jpg"
img = cv2.imread(imgname)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 5, 15, 10)
#cv2.imshow(f"BLur{index}", gray)
face_rect = haar_cascade.detectMultiScale(gray, 1.05, 5,  minSize = [30,30])
for (x,y,w,h) in face_rect:
    face_roi = gray[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(face_roi)
    if(confidence>70 or confidence== 0):
        print(f"Name = {people[label]} with confidence {confidence} in {imgname}")
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, people[label], (x+2,y+h//8), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 1)
        cv2.putText(img, f"Confidence: {confidence}", (x+2,y+h-h//16), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,255,0), 1)
    else:
        print(f"Unrecognized because confidence is {confidence} towards {people[label]}")
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(img, "Unrecognized", (x+2,y+h//8), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1)
#cv2.imshow(f"Detected in  image", img)
#cv2.waitKey(0)

