import cv2
import numpy as np

# 自带人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)


def is_wearing_mask(face_roi):
    # 转灰度图 → 看亮度（最稳定，不受肤色影响）
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # 取人脸下半部（嘴巴位置）
    h, w = gray.shape
    mouth_region = gray[int(h*0.5):h, :]  # 只取下半部分
    
    # 计算亮度
    mean_val = cv2.mean(mouth_region)[0]

    # 逻辑：
    # 没口罩 → 亮 → mean_val 高
    # 有口罩 → 暗 → mean_val 低
    return mean_val < 80  # 暗 = 戴口罩

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        
        if is_wearing_mask(face):
            label = "Mask"
            color = (0, 255, 0)
        else:
            label = "No Mask"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()