import cv2
import numpy as np

# 读取门杆区域的图片，并提取每根门杆的坐标和方向
img = cv2.imread("door.jpg")
mask = cv2.imread("mask.jpg", 0)
door_part = cv2.bitwise_and(img, img, mask=mask)
gray = cv2.cvtColor(door_part, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
door_lines = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if h > 2*w:  # 门杆的高度应该大于2倍宽度，避免误检
        continue
    line = ((x+w//2, y), (x+w//2, y+h))
    door_lines.append(line)

# 对每根门杆进行投影变换或极坐标变换
margin = 10
for line in door_lines:
    pt1, pt2 = line
    width = int(np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2) + margin*2)
    height = max(width, pt2[1]-pt1[1]+margin*2)
    if pt1[0] > img.shape[1]//2:  # 如果门杆在右侧，使用极坐标变换
        center = (pt2[0]-margin, pt2[1]+margin)
        radius = height//2
        polar = cv2.linearPolar(gray[pt1[1]-margin:pt2[1]+margin, pt2[0]-width//2:pt2[0]+width//2],
                                center, radius, cv2.WARP_FILL_OUTLIERS)
        slope = np.mean(polar)
    else:  # 如果门杆在左侧，使用透视变换
        src_points = np.float32([[pt1[0]-margin, pt1[1]-margin], [pt1[0]-margin, pt2[1]+margin],
                                 [pt2[0]+margin, pt1[1]-margin], [pt2[0]+margin, pt2[1]+margin]])
        dst_points = np.float32([[pt1[0]-margin, pt1[1]-margin], [pt1[0]-margin, pt1[1]-margin+height],
                                 [pt2[0]+margin, pt2[1]-margin], [pt2[0]+margin, pt2[1]-margin+height]])
        M = cv2.getPerspectiveTransform(src_points, dst_points)


        
        cropped = gray[pt1[1]-margin:pt2[1]+margin, pt1[0]-margin:pt2[0]+margin]
        warped = cv2.warpPerspective(cropped, M, (width, height))
        edges = cv2.Canny(warped, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        slope_sum = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2-y1) / (x2-x1+1e-5)
            slope_sum += slope
        slope = slope_sum / len(lines)
    if abs(slope) < 0.1:
        print("门杆是直的")
    else:
        print("门杆是歪的")