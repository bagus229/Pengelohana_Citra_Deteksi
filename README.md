# Pengelohana_Citra_Deteksi

## Deteksi Gambar Pejalan Kaki
Kode:
```python
import cv2
import imutils

# Inisialisasi HOG Descriptor dan SVM Detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Membaca gambar
image = cv2.imread("img.png")

# Resize gambar agar pemrosesan lebih cepat
image = imutils.resize(image, width=min(400, image.shape[1]))

# Deteksi pejalan kaki
(regions, _) = hog.detectMultiScale(
    image,
    winStride=(4, 4),
    padding=(4, 4),
    scale=1.05
)

# Gambar kotak di sekitar pejalan kaki yang terdeteksi
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
cv2.waitKey(0)
cv2.destroyAllWindows()
```
Penjelasan: Kode tersebut digunakan untuk mendeteksi pejalan kaki pada gambar menggunakan OpenCV. Program mendeteksi manusia dengan metode HOG + SVM, kemudian ketika berhasil diidentifikasi ditandai dengan kotak merah pada setiap pejalan kaki dan ditampilkan dalam bentuk gambar.

## Deteksi Video Pejalan Kaki
Kode:
```python
import cv2
import imutils
# Menginisialisasi orang HOG
# detektor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# Membaca Gambar
image = cv2.imread('img.png')
# Mengubah ukuran gambar
image = imutils.resize(image, width=min(400, image.shape[1]))
# Mendeteksi semua wilayah di
# Gambar yang memiliki pejalan kaki di dalamnya
(regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)
# Menggambar wilayah dalam Gambar
for (x, y, w, h) in regions:
    cv2.rectangle(image, (x, y),
                    (x + w, y + h),
                    (0, 0, 255), 2)
# Menampilkan Gambar keluaran
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
Penjelasan: Program diatas digunakan untuk mendeteksi pejalan kaki dalam video format mp4. Menggunakan OpenCV dengan menggunaakn metode HOG + SVM. Setiap pejalan kaki yang terdeteksi akan ditandai kotak merah dan hasilnya ditampilkan secara real-time. Program akan terus berjalan hingga video selesai. 

## Deteksi Plat Nomor
Kode:
```python
import cv2
import pytesseract
import matplotlib.pyplot as plt

# Jalur ke tesseract yang dapat dieksekusi
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def detect_plate_number(image_path):
    # Muat gambar
    image = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Konversi ke skala abu-abu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Terapkan Gaussian Blur untuk menghilangkan noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Deteksi tepi (Canny) untuk menyorot kontur pelat
    edges = cv2.Canny(blurred, 100, 200)

    # Temukan kontur untuk menemukan plat nomor
    contours, _ = cv2.findContours(
        edges.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Urutkan kontur berdasarkan area (urutan menurun)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    plate_contour = None
    for contour in contours:
        # Perkiraan kontur ke poligon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Periksa apakah kontur memiliki 4 simpul
        # (yang mungkin persegi panjang, khas untuk pelat)
        if len(approx) == 4:
            plate_contour = approx
            break

    if plate_contour is not None:
        # Gambar kotak pembatas di sekitar plat nomor yang terdeteksi
        x, y, w, h = cv2.boundingRect(plate_contour)
        plate_image = gray[y:y + h, x:x + w]

        # Terapkan ambang batas untuk membinarisasi area pelat
        _, thresh = cv2.threshold(
            plate_image, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Lakukan OCR pada area pelat yang terdeteksi
        # Perlakukan sebagai satu kata
        plate_number = pytesseract.image_to_string(thresh, config='--psm 8')
        return plate_number.strip()
    else:
        return "License plate not detected"


# Berikan jalur gambar
image_path = "car.jpg"  # Ganti dengan jalur gambar Anda

# Deteksi dan cetak nomor plat
plate_number = detect_plate_number(image_path)
print("Detected Plate Number:", plate_number)
```
Penjelasan: program tesebut berguna untuk mendeteksi dan membaca plat nomor kendaraan dengan menggunakan OpenCV dan Tesseract OCR. program membaca gambar kendaraan. kemudian, mengubahnya menjadi grayscale, mengurangi noise dengan Gaussian Blur, lalu mendeteksi tepinya. area plat yang terdeteksi diporses menggunakan Tesseract OCR yang nantinya hasil akan berupa teks sesuai dengan plat nomor pada gambar.
