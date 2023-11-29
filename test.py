import cv2
import numpy as np

# Đọc ảnh
image = cv2.imread('images/21.jpg', 0)  # Đọc ảnh dưới dạng ảnh xám

# Tạo kernel cho erosion
kernel = np.ones((5, 5), np.uint8)  # Kernel kích thước 5x5, tùy chỉnh kích thước nếu cần

# Thực hiện erosion
eroded_image = cv2.erode(image, kernel, iterations=1)

# Hiển thị ảnh gốc và ảnh sau khi erosion
cv2.imshow('Original Image', image)
cv2.imshow('Eroded Image', eroded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
