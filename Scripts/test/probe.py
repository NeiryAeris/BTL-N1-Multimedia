import cv2

image = cv2.imread("sample_query.jpg")
height, width = image.shape[:2]
print(f"Width: {width}, Height: {height}")