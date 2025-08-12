import cv2
import numpy as np
import cvui

def maskApply(image, lower, upper, lower2, upper2, imgB, maskB, rectB, dilateSize, areaSize):
    if(dilateSize < 1):
        dilateSize = 1

    if(areaSize < 50):
        areaSize = 50
    
    # Convert BGR to HSV colorspace
    hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # define masks
    color_mask = cv2.inRange(hsvFrame, lower, upper)
    color_mask2 = cv2.inRange(hsvFrame, lower2, upper2)

    # create kernal
    kernal =  np.ones((dilateSize, dilateSize), dtype=np.uint8)

    # Dilating selected color  pixels
    color_mask = cv2.dilate(color_mask, kernal)
    color_mask2 = cv2.dilate(color_mask2, kernal)
    mask = cv2.bitwise_or(color_mask, color_mask2)

    
    if (imgB and maskB):
        result = cv2.bitwise_and(image, image, mask=mask)
    elif (imgB):
        result = image.copy()
    elif (maskB):
        result = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
    else:
        result = np.zeros(image.shape, dtype=np.uint8)
        

    # Draw rectangle around the object
    if(rectB):
        # Creating contour to track red color
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > areaSize):
                x, y, w, h = cv2.boundingRect(contour)
                result = cv2.rectangle(result, (x, y),(x + w, y + h),(0, 255, 255), 2)
                # Write middle point of rectangle 
                cv2.putText(result, "X:"+ str(int(x + w/2)) + " Y:" + str(int(y + h/2)), (int(x), int(y-5)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)


    return result

WINDOW_NAME = 'Object Detection by Color'

# Load your image
img = cv2.imread('img/2.jpg')
if img is None:
    raise FileNotFoundError("Image not found.")

# Create a frame to draw the UI
frameWidth = 1024
frameHeight = 768
frame = np.zeros((frameHeight, frameWidth, 3), np.uint8)

# Initial HSV range
h_min, s_min, v_min = [162], [105], [71]
h_max = [179]
h2_min, h2_max = [0], [7]

# Which image show (Radio buttons booleans)
img1B, img2B, img3B, img4B, img5B = [True, True], [False, False], [False, False], [False, False], [False, False] #second element is old status
imageChanged = False

# Additional settings vars
imgB, maskB, rectB, = [True], [False], [True]
dilateSize = [5]
areaSize = [8000]

# Margin and position values
xTrackBars = 20
yTrackBars = 40
trackBarsSize = 200
bottomMargin = 75
sideMargin = 25
windowWidth = frameWidth - xTrackBars*2
windowHeight = frameHeight- yTrackBars-bottomMargin*2 - xTrackBars

# Option for some trackbars
options = cvui.TRACKBAR_HIDE_SEGMENT_LABELS

# Initialize cvui
cvui.init(WINDOW_NAME)

while True:
    # Load proper img
    if(img1B[0]):
        img = cv2.imread("img/1.jpg")
    if(img2B[0]):
        img = cv2.imread("img/2.jpg")
    if(img3B[0]):
        img = cv2.imread("img/3.jpg")
    if(img4B[0]):
        img = cv2.imread("img/4.jpg")
    if(img5B[0]):
        img = cv2.imread("img/5.jpg")

    # Fill background
    frame[:] = (49, 52, 49)

    # Draw sliders
    xPos = xTrackBars
    cvui.text(frame, xPos+(8), yTrackBars+(-20), "First Lower Hue")
    cvui.trackbar(frame, xPos, yTrackBars, trackBarsSize, h_min, 0, 179, 4, '%.0Lf')     # Hue min
    cvui.text(frame, xPos+(8), yTrackBars+bottomMargin+(-20), "First Upper Hue")
    cvui.trackbar(frame, xPos, yTrackBars+bottomMargin, trackBarsSize, h_max, 0, 179, 4, '%.0Lf') # Hue max

    xPos = xTrackBars+ trackBarsSize + sideMargin - 30
    cvui.text(frame, xPos+(8), yTrackBars+(-20), "Second Lower Hue")
    cvui.trackbar(frame, xPos, yTrackBars, trackBarsSize, h2_min, 0, 179, 4, '%.0Lf')     # Hue min
    cvui.text(frame, xPos+(8), yTrackBars+bottomMargin+(-20), "Second Upper Hue")
    cvui.trackbar(frame, xPos, yTrackBars+bottomMargin, trackBarsSize, h2_max, 0, 179, 4, '%.0Lf') # Hue max

    xPos = xTrackBars + trackBarsSize*2 + sideMargin*2 -60
    cvui.text(frame, xPos+(8), yTrackBars+(-20), "Lower Saturation")
    cvui.trackbar(frame, xPos, yTrackBars, trackBarsSize, s_min, 0, 255, 3, '%.0Lf', options)  # Sat min
    cvui.text(frame, xPos+(8), yTrackBars+bottomMargin+(-20), "Lower Value")
    cvui.trackbar(frame, xPos, yTrackBars+bottomMargin, trackBarsSize, v_min, 0, 255, 3, '%.0Lf', options) # Val min

    # Image change radio boxes
    xPos = xTrackBars + trackBarsSize*3 + sideMargin*3 - 80
    cvui.checkbox(frame, xPos, yTrackBars-15, 'Image 1', img1B)
    cvui.checkbox(frame, xPos, yTrackBars-15+30, 'Image 2', img2B)
    cvui.checkbox(frame, xPos, yTrackBars-15+30*2, 'Image 3', img3B)
    cvui.checkbox(frame, xPos, yTrackBars-15+30*3, 'Image 4', img4B)
    cvui.checkbox(frame, xPos, yTrackBars-15+30*4, 'Image 5', img5B)
    # Make this CheckBox to Radio Button
    if(img1B[0] != img1B[1] or img2B[0] != img2B[1] or img3B[0] != img3B[1] or img4B[0] != img4B[1] or img5B[0] != img5B[1]):
        imageChanged = True
    if(imageChanged):
        if(img1B[1] == True):
            img1B[0] = False
            img1B[1] = False
        if(img2B[1] == True):
            img2B[0] = False
            img2B[1] = False
        if(img3B[1] == True):
            img3B[0] = False
            img3B[1] = False
        if(img4B[1] == True):
            img4B[0] = False
            img4B[1] = False
        if(img5B[1] == True):
            img5B[0] = False
            img5B[1] = False

        if(img1B[0] == True):
            img1B[1] = True
        if(img2B[0] == True):
            img2B[1] = True
        if(img3B[0] == True):
            img3B[1] = True
        if(img4B[0] == True):
            img4B[1] = True
        if(img5B[0] == True):
            img5B[1] = True
        imageChanged = False

    xPos = int(xTrackBars + trackBarsSize*3.5 + sideMargin*3 - 90)
    cvui.checkbox(frame, xPos, yTrackBars-15, 'Show Image', imgB)
    cvui.checkbox(frame, xPos, yTrackBars-15+30, 'Show Mask', maskB)
    cvui.checkbox(frame, xPos, yTrackBars-15+30*2, 'Draw Rectangles', rectB)
    
    cvui.text(frame, xPos, yTrackBars-15+30*3, "Dilate Size")
    cvui.counter(frame, int(xTrackBars + trackBarsSize*3.45 + sideMargin*3),  yTrackBars-20+30*3, dilateSize)

    cvui.text(frame, xPos, yTrackBars-15+30*4, "Selected Area Size")
    cvui.counter(frame, int(xTrackBars + trackBarsSize*3.7 + sideMargin*3),  yTrackBars-20+30*4, areaSize, 50)


    # Tweaking masks :D
    lower = np.array([int(h_min[0]), int(s_min[0]), int(v_min[0])])
    upper = np.array([int(h_max[0]),255, 255])
    lower2 = np.array([int(h2_min[0]), int(s_min[0]), int(v_min[0])])
    upper2 = np.array([int(h2_max[0]),255, 255])

    result = maskApply(img, lower, upper, lower2, upper2, imgB[0], maskB[0], rectB[0], dilateSize[0], areaSize[0])

    # Show result preview inside the same window
    preview = cv2.resize(result, (windowWidth, windowHeight))
    startY = int(yTrackBars+bottomMargin+bottomMargin)
    startX = xTrackBars
    frame[startY:startY+windowHeight, startX:startX+windowWidth] = preview

    cvui.update()
    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(20) == 27:  # ESC to quit
        break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
