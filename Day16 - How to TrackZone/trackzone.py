import cv2
from ultralytics import solutions

cap = cv2.VideoCapture("")
w,h,fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH,
              cv2.CAP_PROP_FRAME_HEIGHT,
              cv2.CAP_PROP_FPS)
)

video_writer = (
    cv2.VideoWriter(
        "results.avi",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,(w,h)
    )
)

solution = solutions.TrackZone(
    model = r"F:\D Drive\Linkedin Learning\Open CV, Hands on AI\Basics of Open CV\Day16 - How to TrackZone\best-detect.pt",
    show = True,
    show_in = True,
    show_out = True,
    
)
while cap.isOpened():
    success,im0 = cap.read()
    if not success:
        break
    results = solution(im0)
    video_writer.write(results.plot_im)
cap.release()
video_writer.release()
cv2.destroyAllWindows()