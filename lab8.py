import cv2
import numpy as np

# зашумление фото
template = cv2.imread('variant-5.jpg', 0)
noise = np.random.normal(0, 1, template.shape).astype(np.uint8)
noise_template = cv2.add(template, noise)

# загрузка видео и метода ORB, видео не смог загрузить, т.к. его размер слишком большой
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(noise_template, None)
capt = cv2.VideoCapture('lab8.mp4')

# обработка кадров
while True:
    ret, frame = capt.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(gray_frame, None)

    kp_ch = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = kp_ch.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # дальнейшая обработка, если достаточно совпадений
    if len(matches) > 160:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            h, w = noise_template.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 2, cv2.LINE_AA)

            center = np.mean(dst, axis=0).flatten()

            # смена цвета при попадании в углы (границы зон я увеличил,
            # т.к. понятия не имею как уместить метку в зону 50 на 50)
            if center[0] < 200 and center[1] < 200:
                color = (255, 0, 0)  # Синий
            elif center[0] > frame.shape[1] - 200 and center[1] > frame.shape[0] - 200:
                color = (0, 0, 255)  # Красный
            else:
                color = (0, 255, 0)  # Зеленый

            frame = cv2.polylines(frame, [np.int32(dst)], True, color, 2, cv2.LINE_AA)

    # отображение кадра
    cv2.imshow('Для выхода нажмите q', frame)

    # выход по нажатию q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capt.release()
cv2.destroyAllWindows()
# код получился сильноусложнённым, но только в таком виде он работает
# более менее нормально
