from ultralytics import YOLO
import os

image_path = os.path.join(os.path.dirname(__file__), "image.png")

## Importing the necessary Model
## We can Either import small,large or extra learge or medium model
model = YOLO("yolo11n.pt")

#results = model.predict(source=image_path,save = True)
results = model.predict(source="./Day2 - Exploring Ultralytics/image.png",save = True)

for result in results:
    print(result.boxes.xyxy)