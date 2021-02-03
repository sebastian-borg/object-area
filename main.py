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
frame = aruco.drawDetectedMarkers(frame, corners)

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

for pt in closest_2_middle:
    frame = mark_point(pt, frame)


pts = np.array(closest_2_middle)
warped = four_point_transform(frame, pts)   
#warped = cv2.resize(warped, (0,0), None,0.25,0.25)

gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

edged = cv2.Canny(gray, 30, 100)
edged = cv2.dilate(edged, None, iterations=3)
edged = cv2.erode(edged, None, iterations=3)

contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if cv2.contourArea(contour) > 1000:
        print(cv2.contourArea(contour))
        warped = cv2.drawContours(warped, contour, -1, (0, 0, 255), 2)

cv2.imshow("warped", warped)

warp_height, warp_width, _ = warped.shape

print(warp_height)
print(warp_width)




cv2.imshow("detection", edged)


cv2.imshow("image", frame)

cv2.waitKey()
cv2.destroyAllWindows()