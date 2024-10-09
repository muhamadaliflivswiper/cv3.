import cv2


face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
glasses_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

video_capture = cv2.VideoCapture(0)

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(vid, "Wajah", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        
        roi_gray = gray_image[y:y + h, x:x + w]
        roi_color = vid[y:y + h, x:x + w]

        
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            cv2.putText(roi_color, "Mata", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        
        smiles = smile_classifier.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            cv2.putText(roi_color, "Senyum", (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        
        glasses = glasses_classifier.detectMultiScale(roi_gray)
        for (gx, gy, gw, gh) in glasses:
            cv2.rectangle(roi_color, (gx, gy), (gx + gw, gy + gh), (255, 255, 0), 2)
            cv2.putText(roi_color, "Kacamata", (gx, gy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    return faces

while True:
    result, video_frame = video_capture.read()  
    if not result:
        break  

    
    video_frame = cv2.flip(video_frame, 1)

    detect_bounding_box(video_frame)  

    cv2.imshow("My Face Detection Project", video_frame) 

    
    if cv2.getWindowProperty("My Face Detection Project", cv2.WND_PROP_VISIBLE) < 1:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()