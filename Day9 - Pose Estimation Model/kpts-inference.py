from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

model.predict(
    source = r"F:\D Drive\Linkedin Learning\Open CV, Hands on AI\Basics of Open CV\Day9 - Pose Estimation Model\video.mov",
    line_width = 2,
    show = True,
    conf = 0.5 ## This ensures that any prediction will have less than 50% of confidence will not be displayedd
    ## In case of the IOU , If we have different objects then we use the IOU!
)