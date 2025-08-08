from ultralytics import YOLO
import cv2

yolo_model_v10 = r"C:\Users\kesav_u5svehq\Downloads\bone_fracture_detection_main\weights\yv10.pt"

model = YOLO(yolo_model_v10)

img = cv2.imread(r"C:\Users\kesav_u5svehq\Downloads\bone_fracture_detection_main\testing_images\WhatsApp Image 2024-11-04 at 17.51.46.jpeg")

result = model(img)


detection_image = result[0].plot()

save_path = r"C:\Users\kesav_u5svehq\Downloads\bone_fracture_detection_main\testing_images\detection_output.jpeg"

cv2.imwrite(save_path, detection_image)

print(f"Detection image saved at {save_path}")