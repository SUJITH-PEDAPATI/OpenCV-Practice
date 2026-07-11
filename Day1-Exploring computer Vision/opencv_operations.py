import cv2

image = cv2.imread("Day-1/image.jpg")

blurredImage = cv2.blur(image,(50,50))

# cv2.imshow("Blurred Image", blurredImage)
# cv2.waitKey(0)

cv2.imwrite("./Day-1/test.png",blurredImage)

## Crop[ing the Image
slicedImage = image[240:450,600:900]
# cv2.imshow("CroppedImage",slicedImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## Reading and Processing the video file
videoCapture = cv2.VideoCapture("./Day-1/video.mov")
while videoCapture.isOpened():
    success,frame = videoCapture.read()
    if success:
        cv2.imshow("Test Video",frame)
        if cv2.waitKey(2) & 0xFF == ord("q"):
            break
    else:
        break


