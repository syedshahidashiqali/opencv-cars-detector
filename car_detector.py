import cv2


# Our pre-trianed car classifier
classifier_file = "car-detector.xml"

# create our classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# Create a VideoCapture object and read from input file
# If the input is taken from the camera, pass 0 instead of the video file name.
video = cv2.VideoCapture('../Tesla Autopilot Dashcam Compilation 2021 Version-d4L1Pte7zVc.mp4')

# Runs until car stops
while True:
    
    # read the current frame
    (read_successful, frame) = video.read()
    
    if read_successful:
        gray_scaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    cars = car_tracker.detectMultiScale(gray_scaled_frame)
    # display img
    for(x, y, w, h) in cars:
        cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (255,0,0),
        2
        )
    cv2.imshow("Shahid Car detector img", frame)

    # Dont autoclose (wait for key press)
    cv2.waitKey(1)