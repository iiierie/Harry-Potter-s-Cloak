import cv2
import numpy as np

#start camera
cap = cv2.VideoCapture(0)
do_nothing = lambda x: None
#Trackbars
cv2.namedWindow("HSV Adjuster")
cv2.createTrackbar("Upper H", "HSV Adjuster", 110, 180,do_nothing)
cv2.createTrackbar("Lower H", "HSV Adjuster", 68, 180,do_nothing)

cv2.createTrackbar("Upper S", "HSV Adjuster", 255, 255,do_nothing)
cv2.createTrackbar("Lower S", "HSV Adjuster", 55, 255,do_nothing)

cv2.createTrackbar("Upper V", "HSV Adjuster", 255, 255,do_nothing)
cv2.createTrackbar("Lower V", "HSV Adjuster", 55, 255,do_nothing)



# initial frame
while True:
    cv2.waitKey(1500)
    done, initial_frame = cap.read()
    if done:
        break

while True:
    ret, frame = cap.read()
    # Convert BGR color space of captured frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get HSV values from the trackbars
    upper_hue = cv2.getTrackbarPos("Upper H", "HSV Adjuster")
    upper_saturation = cv2.getTrackbarPos("Upper S", "HSV Adjuster")
    upper_value = cv2.getTrackbarPos("Upper V", "HSV Adjuster")
    lower_hue = cv2.getTrackbarPos("Lower H", "HSV Adjuster")
    lower_saturation = cv2.getTrackbarPos("Lower S", "HSV Adjuster")
    lower_value = cv2.getTrackbarPos("Lower V", "HSV Adjuster")

    # Define a kernel for dilation
    dilation = np.ones((3, 3), np.uint8)

    upper_hsv = np.array([upper_hue, upper_saturation, upper_value])
    lower_hsv = np.array([lower_hue, lower_saturation, lower_value])

    # Create a mask to isolate the chosen sheet color
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
    mask = cv2.medianBlur(mask, 3)
    mask_inv = 255 - mask
    mask = cv2.dilate(mask, dilation, 7)

    # Apply the cloak effect
    b = frame[:, :, 0]
    g = frame[:, :, 1]
    r = frame[:, :, 2]
    b = cv2.bitwise_and(mask_inv, b)
    g = cv2.bitwise_and(mask_inv, g)
    r = cv2.bitwise_and(mask_inv, r)
    frame_inv = cv2.merge((b, g, r))

    b = initial_frame[:, :, 0]
    g = initial_frame[:, :, 1]
    r = initial_frame[:, :, 2]
    b = cv2.bitwise_and(b, mask)
    g = cv2.bitwise_and(g, mask)
    r = cv2.bitwise_and(r, mask)
    sheet = cv2.merge((b, g, r))

    final_frame = cv2.bitwise_or(frame_inv, sheet)

    # Display the final frame with the cloak effect
    cv2.imshow("Magic Cloak", final_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(3) == ord('q'):
        break


cv2.destroyAllWindows()
cap.release()
