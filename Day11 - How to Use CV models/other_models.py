# from ultralytics import YOLO

# model = YOLO("yolo11.pt")
# from ultralytics import SAM
# ## Load the Model
# model = SAM("sam2.1_b.pt")

# # Run Inference
# model.predict(
#     source = r"F:\D Drive\Linkedin Learning\Open CV, Hands on AI\Basics of Open CV\Day11 - How to Use CV models\image.png",
#     save = True,
# )

from ultralytics import FastSAM
source = r"F:\D Drive\Linkedin Learning\Open CV, Hands on AI\Basics of Open CV\Day11 - How to Use CV models\image.png",

model = FastSAM("FastSAM-x.pt")

everything_results = model.predict(
    source = source,
    device = 'cuda',
    retina_masks = True,
    imgsz = 1024,
    conf = 0.4,
    iou = 0.9,
    save = True
)