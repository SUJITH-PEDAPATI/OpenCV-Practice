from ultralytics import YOLO

# model = YOLO("yolo26n.pt")

# model.export(
#     format = "onnx", ## widely used format for the exporting
#     simplify = True, ## Simplyfy = True can be used only using the ONNX file format
#     batch = 2,
# )

model = YOLO("yolo26n-pose.pt")

model.export(
    format = "torchscript",
    batch = 2,
    imgsz = 416
)
