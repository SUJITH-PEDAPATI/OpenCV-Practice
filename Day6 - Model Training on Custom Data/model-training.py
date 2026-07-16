import os
from ultralytics import YOLO

# Resolve absolute path to data.yaml
script_dir = os.path.dirname(os.path.abspath(__file__))
data_yaml_path = os.path.join(script_dir, "data.yaml")

model = YOLO(r"F:\D Drive\Linkedin Learning\Open CV, Hands on AI\Basics of Open CV\Day6 - Model Training on Custom Data\dataset\train\weights\best.pt")

video_path = os.path.join(script_dir, "video.mov")
model.predict(source = video_path, show = True, line_width = 2)

# if __name__ == '__main__': ## Helps not to re-initalize the model for evry epcohs
#     model.train(
#         data = data_yaml_path,
#         batch = 16,
#         workers = 1,
#         epochs = 100,
#         project = os.path.join(script_dir, "dataset"),
#         name = "train",
#         exist_ok = True
#     )