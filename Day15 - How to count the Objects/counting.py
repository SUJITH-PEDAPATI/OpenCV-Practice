import cv2
from ultralytics import solutions

# cap = cv2.VideoCapture(r"F:\D Drive\Linkedin Learning\Open CV, Hands on AI\Basics of Open CV\Day15 - How to count the Objects\video.mp4")
cap = cv2.VideoCapture(r"F:\D Drive\Linkedin Learning\Open CV, Hands on AI\Basics of Open CV\Day15 - How to count the Objects\video1.mp4")

w,h,fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH,
              cv2.CAP_PROP_FRAME_HEIGHT,
              cv2.CAP_PROP_FPS)
)
## Object Counting Module

## In this case, we are not passing the coordinates, so it by default takes the default coordinates
solution = solutions.ObjectCounter(
    # model = r"F:\D Drive\Linkedin Learning\Open CV, Hands on AI\Basics of Open CV\Day15 - How to count the Objects\best-detect.pt", ## This is trained only on the Apples video

    ## So we use the YOLO Model
    model = "yolo26n.pt",
    classes = [39],
    show_in = True,
    show_out = True,
    show = True,
    region = [(640,0),(640,720)],
    line_width = 4
)

video_writer = (
    cv2.VideoWriter(
        "result.avi",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,(w,h)
    )
)


## Processing the Video
while cap.isOpened():
    success,im0 = cap.read()
    if not success:
        break
    results = solution(im0)
    video_writer.write(results.plot_im)
cap.release()
video_writer.release()
cv2.destroyAllWindows()