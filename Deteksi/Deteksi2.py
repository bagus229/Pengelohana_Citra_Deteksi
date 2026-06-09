import cv2
import imutils

# Inisialisasi HOG Descriptor dan SVM Detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Membaca video
cap = cv2.VideoCapture("vid.mp4")

while cap.isOpened():

    # Membaca frame video
    ret, image = cap.read()

    if not ret:
        break

    # Resize frame
    image = imutils.resize(
        image,
        width=min(400, image.shape[1])
    )

    # Deteksi pejalan kaki
    (regions, _) = hog.detectMultiScale(
        image,
        winStride=(4, 4),
        padding=(4, 4),
        scale=1.05
    )

    # Gambar kotak pada pejalan kaki yang terdeteksi
    for (x, y, w, h) in regions:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

    # Tampilkan hasil
    cv2.imshow("Deteksi Pejalan Kaki", image)

    # Tekan Q untuk keluar
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()