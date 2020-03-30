import cv2

face_cascade = cv2.CascadeClassifier('DATA/haarcascade_frontalface_alt2.xml')


def detect_face(img):

    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(
        face_img,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
    )

    for(x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x+w, y+h), (255, 255, 0), 3)

    return face_img


cap = cv2.imread('IMG-20180401-WA0009.jpg')     # copy image src address
cap1 = cap.copy()                               # copy previous img into a variable


while True:

        frame = detect_face(cap)
        frame_gray = detect_face(cap1)
        gray_img = cv2.cvtColor(frame_gray,cv2.COLOR_BGR2GRAY)      # convert img into gray-scaling img by this function

        cv2.imshow('Image Face Detect', frame)
        cv2.imshow('Gray Scaling Image Face Detect', gray_img)      # show conversion img

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cap.release()
cap1.release()
cv2.destroyAllWindows()
