from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-seg.pt")  # pretrained YOLO26n model

# Run batched inference on a list of images
#  results = model.predict(source = "./Day5- Tasks  and Modes/video.mov",show = True)  # return a list of Results objects
results = model.export(format = "torchscript")  # export the model to TorchScript format

# Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk