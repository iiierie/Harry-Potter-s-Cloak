## Magic Cloak with OpenCV and Python

This is a Python script using OpenCV to create a "magic cloak" effect, where a chosen color (in HSV color space) is replaced with a background image or the initial frame captured from the camera. This creates an illusion of objects or people being invisible under the chosen color.

## Dependencies

- [Python](https://www.python.org/): The programming language used for the script.
- [OpenCV](https://opencv.org/): Open Source Computer Vision Library for image and video processing.

## Installation

You can install the required packages using pip:

```bash
pip install opencv-python numpy
```




## Explanation of Code
```
# Import the necessary libraries
import cv2
import numpy as np


# Start the camera capture
cap = cv2.VideoCapture(0)
```
The code begins by importing the required libraries, OpenCV (cv2) for image and video processing and NumPy (np) for numerical operations.

cv2.VideoCapture(0) initializes the camera capture. The argument 0 typically represents the default camera, but you can replace it with the camera index if you have multiple cameras connected.
```
# Define a function to do nothing inline
do_nothing = lambda x: None
```
This line defines a simple inline function called do_nothing using a lambda expression. It takes a single argument x and does nothing with it. This function is used as a placeholder callback for trackbars that don't need to perform any specific action.
```
# Create a GUI window for adjusting HSV values
cv2.namedWindow("HSV Adjuster")
```
cv2.namedWindow("HSV Adjuster") creates a graphical user interface (GUI) window titled "HSV Adjuster." This window will be used to adjust the HSV values for color selection.
```
# Create trackbars for adjusting upper and lower HSV values
cv2.createTrackbar("Upper H", "HSV Adjuster", 110, 180, do_nothing)
cv2.createTrackbar("Lower H", "HSV Adjuster", 68, 180, do_nothing)

cv2.createTrackbar("Upper S", "HSV Adjuster", 255, 255, do_nothing)
cv2.createTrackbar("Lower S", "HSV Adjuster", 55, 255, do_nothing)

cv2.createTrackbar("Upper V", "HSV Adjuster", 255, 255, do_nothing)
cv2.createTrackbar("Lower V", "HSV Adjuster", 55, 255, do_nothing)
```
These lines create trackbars within the "HSV Adjuster" window. Each trackbar corresponds to an HSV (Hue, Saturation, Value) component (H, S, V) and allows the user to adjust the upper and lower bounds of that component. The trackbars are initialized with default values and use the do_nothing function as a callback when the values change.
```# Capture an initial frame for background subtraction
while True:
    cv2.waitKey(1500)
    done, initial_frame = cap.read()
    if done:
        break
```
This code captures an initial frame from the camera to be used as a background. It waits for 1500 milliseconds (1.5 seconds) to allow the camera to adjust to lighting conditions. The captured frame is stored in the initial_frame variable.
```
# Main loop for video processing
while True:
    ret, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
```
This loop continuously captures frames from the camera and converts them from the BGR color space to the HSV color space. The converted frame is stored in the hsv_frame variable.

```
    # Get HSV values from the trackbars
    upper_hue = cv2.getTrackbarPos("Upper H", "HSV Adjuster")
    upper_saturation = cv2.getTrackbarPos("Upper S", "HSV Adjuster")
    upper_value = cv2.getTrackbarPos("Upper V", "HSV Adjuster")
    lower_hue = cv2.getTrackbarPos("Lower H", "HSV Adjuster")
    lower_saturation = cv2.getTrackbarPos("Lower S", "HSV Adjuster")
    lower_value = cv2.getTrackbarPos("Lower V", "HSV Adjuster")
```
This section of code retrieves the current HSV values from the trackbars for upper and lower bounds of Hue (H), Saturation (S), and Value (V). These values define the color range to be made invisible.
```
    # Define a kernel for dilation
    dilation = np.ones((3, 3), np.uint8)

    upper_hsv = np.array([upper_hue, upper_saturation, upper_value])
    lower_hsv = np.array([lower_hue, lower_saturation, lower_value])
```
A kernel for dilation is defined as a 3x3 matrix.

upper_hsv and lower_hsv arrays are created to store the upper and lower bounds of the HSV color range based on the trackbar values.
```
    # Create a mask to isolate the chosen color
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
    mask = cv2.medianBlur(mask, 3)
    mask_inv = 255 - mask
    mask = cv2.dilate(mask, dilation, 7)
```
A mask is created using cv2.inRange to isolate the chosen color within the specified HSV range.

The mask is then smoothed using cv2.medianBlur with a kernel size of 3.

An inverted mask mask_inv is created to represent the opposite of the mask.

The mask is further processed using dilation to enhance the detected color region.
```
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
```
The code applies the "cloak" effect by masking out the chosen color from the live frame and overlaying it with either the initial_frame or a background frame.

The cv2.bitwise_and operation is used to combine the masks and the color channels (B, G, R) of the frames.

The resulting frames are stored in frame_inv and sheet.

```
    # Combine the frames to create the final output
    final_frame = cv2.bitwise_or(frame_inv, sheet)

    # Display the final frame with the cloak effect
    cv2.imshow("Magic Cloak", final_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(3) == ord('q'):
        break

# Close all OpenCV windows, release the camera, and exit
cv2.destroyAllWindows()
cap.release()
```
The final frame is created by combining frame_inv (frame with the color region removed) and sheet (either the background or initial frame).

The cv2.imshow function displays the final frame with the "Magic Cloak" effect in a GUI window.

The loop continues until the user presses the 'q' key, at which the loop exists and program terminates.


