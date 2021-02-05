import cv2
import cv2.aruco as aruco
import numpy as np


def closest_point(point, point_list):
    point_list = np.asarray(point_list)
    deltas = point_list - point
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def mark_point(point, frame):
    return cv2.circle(frame, (int(point[0]),int(point[1])), radius=3, color=(0, 0, 255), thickness=3)


def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
   
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
  
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

""" def mark_square(pt_1,pt_2, frame):
    return cv2.rectangle(frame, (int(pt_1[0]),int(pt_1[1])), (int(pt_2[0]),int(pt_2[1])), color=(0, 0, 255), thickness=3)
 """
##########################################################################################################################################
frame = cv2.imread("test_img_3/File_000.jpeg")

frame = cv2.resize(frame, (0,0), None,0.25,0.25)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_1000)
arucoParameters = aruco.DetectorParameters_create()

corners, ids, rejectedImgPoints = aruco.detectMarkers(
    gray, aruco_dict, parameters=arucoParameters)
#print(corners)
#frame = aruco.drawDetectedMarkers(frame, corners)

#print(len(corners))

# square size 700x700mm outer dimension

# get the square

all_cords = []

for square in corners:
    all_cords.extend(square[0].tolist())

height, width, _ = frame.shape

top_left_point = all_cords[closest_point([0,0], all_cords)]
top_right_point = all_cords[closest_point([width, 0], all_cords)]

bottom_left_point = all_cords[closest_point([0,height], all_cords)]
bottom_right_point = all_cords[closest_point([width, height], all_cords)]

closest_2_middle = []
for square in corners:
    closest_2_middle.append(square[0][closest_point([width*0.5, height*0.5], square[0])].tolist())

""" 
for pt in closest_2_middle:
    frame = mark_point(pt, frame) 
"""

ref_pts = np.array([top_left_point, top_right_point, bottom_left_point, bottom_right_point])
pts = np.array(closest_2_middle)
warped = four_point_transform(frame, pts)   
size_ref = four_point_transform(frame, ref_pts)   
#warped = cv2.resize(warped, (0,0), None,0.25,0.25)


# find markers to get a size ref
size_ref_gray = cv2.cvtColor(size_ref, cv2.COLOR_BGR2GRAY)
size_ref_arucoParameters = aruco.DetectorParameters_create()
size_ref_arucoParameters.minDistanceToBorder = 0
size_ref_arucoParameters.adaptiveThreshWinSizeMax = 400
size_ref_aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_50)

size_ref_corners, _, _ = aruco.detectMarkers(
    size_ref_gray, size_ref_aruco_dict, parameters=size_ref_arucoParameters)

size_ref = aruco.drawDetectedMarkers(size_ref, size_ref_corners)

cv2.imshow("size ref",size_ref)

#calc pixel to mm^2 ratio
tot_pixel_area_markers = 0
marker_mm_area = 50*50

for corner in size_ref_corners:
    tot_pixel_area_markers += cv2.contourArea(corner)

average_marker_area = tot_pixel_area_markers / len(size_ref_corners)
pixels_per_mm_sq = average_marker_area / marker_mm_area


# find areas with similar color 
frame_hsv = cv2.cvtColor(warped, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(frame_hsv)


# code for shifting
#s = (s - 30) % 180



""" 
funkar ganska bra
s = cv2.GaussianBlur(s, (7, 7), 0)
cv2.imshow("s", s)
canny_s = cv2.Canny(s, 100, 50) 

"""
""" h = cv2.GaussianBlur(h, (7, 7), 0)
cv2.imshow("h", h)
canny_h = cv2.Canny(h, 100, 50) 
canny_h[:, :] = 0
cv2.imshow("canny_h", canny_h) 
"""
canny_h = h
canny_h[:, :] = 0

v = cv2.GaussianBlur(v, (7, 7), 0)
cv2.imshow("v", v)
canny_v = cv2.Canny(v, 200, 100) 
cv2.imshow("canny_v", canny_v)

s = cv2.GaussianBlur(s, (7, 7), 0)
cv2.imshow("s", s)
canny_s = cv2.Canny(s, 100, 50)
cv2.imshow("canny_s", canny_s)

#canny_s = cv2.Canny(s,255, 255/3)
canny_s = cv2.dilate(canny_s, None, iterations=1)
canny_s = cv2.erode(canny_s, None, iterations=1)

canny_v = cv2.dilate(canny_v, None, iterations=1)
canny_v = cv2.erode(canny_v, None, iterations=1)


canny_all = cv2.merge([canny_h,canny_s, canny_v])

canny_all_gray = cv2.cvtColor(canny_all, cv2.COLOR_BGR2GRAY)

canny_all_gray = cv2.dilate(canny_all_gray, None, iterations=1)
canny_all_gray = cv2.erode(canny_all_gray, None, iterations=1)
cv2.imshow("canny_all", canny_all_gray)


contours, hierarchy  = cv2.findContours(canny_all_gray.copy(), cv2.RETR_TREE ,
	cv2.CHAIN_APPROX_SIMPLE)


""" contours_v, hierarchy_v  = cv2.findContours(canny_v.copy(), cv2.RETR_TREE ,
	cv2.CHAIN_APPROX_SIMPLE)


contours_s, hierarchy_s  = cv2.findContours(canny_s.copy(), cv2.RETR_TREE ,
	cv2.CHAIN_APPROX_SIMPLE)

contours = np.concatenate((contours_v, contours_s), axis=None)
hierarchy = np.concatenate((hierarchy_v, hierarchy_s), axis=None)

 """
for i in range(len(contours)):
    contour = contours[i]
    area = cv2.contourArea(contour)
    #ignore small contours
    if area < 1000:
        continue
    #ignore outer contours
    if hierarchy[0][i][3] < 0:
        continue

    cm_2_area = (area/pixels_per_mm_sq)/100
    cv2.drawContours(warped, contour, -1, (0, 0, 255), 2)
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(warped,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(warped,
                "{:.1f}".format(cm_2_area),
                (int(x+w/4),int(y+h/2)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                .5,(255,255,255),1)


    

cv2.imshow("warped", warped)


#cv2.imshow("image", frame)

cv2.waitKey()
cv2.destroyAllWindows()