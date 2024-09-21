from ultralytics import YOLO
import cv2

# import matplotlib
# matplotlib.use('gtk4agg',force=True)

image_path = "/media/raul/d1964fe0-512e-4389-b8f7-b1bd04e829612/Projects/Free/image-object-manage/data/images/cats/4.jpg"
image = cv2.imread(image_path)

# model = YOLO("yolov8s.pt")
# result = model.track(image, persist=True)
# plot = result[0].plot()

# cv2.imwrite(
#     "/media/raul/d1964fe0-512e-4389-b8f7-b1bd04e829612/Projects/Free/image-object-manage/data/images/generate/yolo.jpg",
#     plot,
# )

model_seg = YOLO("yolov8m-seg.pt")

