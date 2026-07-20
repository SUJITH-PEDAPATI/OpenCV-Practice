from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.track(
    source = r"F:\D Drive\Linkedin Learning\Open CV, Hands on AI\Basics of Open CV\Day12 - How to predict and track detectec objects\video.mov",
    conf = 0.6, ## 60% Confidence Score
    line_width = 4,
    device = 'cuda',
    persist = True,
    # max_det = 2,
    show = True, ## Helps in Visualizing the videos
    save = True
)