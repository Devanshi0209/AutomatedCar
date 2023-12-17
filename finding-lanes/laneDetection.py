import cv2
import numpy as np
import matplotlib.pyplot as plt




# canny function for boundary detection
#RGB->BW->Gaussian Blur->Canny
def canny_algo(image):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    return cv2.Canny(blur,50,150)


#give co-ordinates of area to be masked
def region_of_interest(image):
    height= image.shape[0]
    roi_triangle =np.array([[(240,690),(585,554),(711,550),(945,690)]])
    return roi_triangle

#turns co0rdinates to mask
def mask(triangle, image):
    mask = np.zeros_like(image) #create array of zeros of size of image
    cv2.fillPoly(mask, triangle ,255)  # fill polygon with 1s for mask
    # return mask
    return cv2.bitwise_and(image, mask)  # applies mask by doing bitwise AND

#create lines
def hough_lines(masked_image):
    lines=cv2.HoughLinesP(masked_image, 3, np.pi/180,100, np.array([]),minLineLength=10, maxLineGap=5)
    #(image,precison , precision( in radian),threshold, place holder array,minLineLength,maxLineGap=5 ) keep precison low for better result
    return lines

#takes in raw image and lines and display/return lines image
def display_lines(image,lines):
    lines_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lines_image, (x1,y1), (x2,y2), (22.4,100,7.8), 10)
    combined_image= cv2.addWeighted(image, 1, lines_image, 1, 1)
    return combined_image


#takes in intercept and slope values, and returns coordinates value
def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    y1 = image.shape[0]
    y2=int(y1*77/100)
    x1= int((y1-intercept)/slope)
    x2= int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])


def average_slope_intercept(image, lines) :
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4) #making array inear and assigning coordinate values
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        # print(parameters,"parameters")
        #polyfit  fits 1st degree(3rd parameter) polynomial in given point and return y itercept and slope.
        slope=parameters[0]
        intercept=parameters[1]
        #lines on left will be slanted towards right and will have a negative slope. Similarly with right side lines
        # **** can change on camera rotation, will have to modify logic
        if slope<0:
            left_fit.append((slope,intercept))
        else :
            right_fit.append((slope,intercept))
    left_fit_average= np.average(left_fit,axis=0)
    right_fit_average= np.average(right_fit, axis=0)
    left_line= make_coordinates(image, left_fit_average)
    right_line= make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

# orignal_image=cv2.imread('test4.jpg')
#
# copy_image = np.copy(orignal_image) #making a copy of image to pass
# canny = canny_algo(copy_image)
# mask_area = region_of_interest(canny)
# masked_image = mask(mask_area, canny)
# lane_lines =  hough_lines(masked_image)
# averaged_lane_lines= average_slope_intercept(copy_image,lane_lines)
# line_image = display_lines(copy_image, averaged_lane_lines)


cap= cv2.VideoCapture("videotest.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny = canny_algo(frame)
    mask_area = region_of_interest(canny)
    masked_image = mask(mask_area, canny)
    lane_lines =  hough_lines(masked_image)
    try:
        averaged_lane_lines= average_slope_intercept(frame,lane_lines)
        line_image = display_lines(frame, averaged_lane_lines)
    except Exception:
        line_image = display_lines(frame, lane_lines)
    output = line_image
    cv2.imshow('review2',output)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
        #shows the result image for specified amount of milliseconds- if given as 0 it shows till we press a key on keyboard , 1 cuz we wait 1 ms between frames of video


cap.release()
cv2.destroyAllWindows()


# output = line_im  age

# displaying image
# plt.imshow(output)
# plt.show()
# cv2.imshow('aisehikoochbhi',output)
# cv2.waitKey(0)
