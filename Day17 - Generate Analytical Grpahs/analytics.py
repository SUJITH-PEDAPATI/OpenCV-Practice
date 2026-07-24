import cv2
from ultralytics import solutions

cap = cv2.VideoCapture(r"F:\D Drive\Linkedin Learning\Open CV, Hands on AI\Basics of Open CV\Day17 - Generate Analytical Grpahs\video.mov")

w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH,
              cv2.CAP_PROP_FRAME_HEIGHT,
              cv2.CAP_PROP_FPS))
video_writer = (
    cv2.VideoWriter(
        "result.avi",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (1280, 720)))
 ## Instead of fps,(w,h) we can also use fps,(1280,720) because the video is of 1280x720 resolution - To create the analytical graphs, we need to have a fixed resolution of 1280x720, so we can use this resolution instead of the original video resolution

# Analytics Module Initialization
solution = solutions.Analytics(
    model=r"F:\D Drive\Linkedin Learning\Open CV, Hands on AI\Basics of Open CV\Day17 - Generate Analytical Grpahs\new-model.pt",
    analytics_type="area",
    device="cpu",
    show=True,
)

frame_count = 0
# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    frame_count += 1
    results = solution(im0, frame_count)
    video_writer.write(results.plot_im)
cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy windows

# from ultralytics import YOLO

# model = YOLO("yolo26n.pt")

# model.predict(source=r"F:\D Drive\Linkedin Learning\Open CV, Hands on AI\Basics of Open CV\Day17 - Generate Analytical Grpahs\video.mov",
#               show=True)