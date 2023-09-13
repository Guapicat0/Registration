import cv2
import numpy as np

# 读取图像
image1 = cv2.imread("F:/Registering/data/DroneRGBT/Train/RGB/1000.jpg")
image2 = cv2.imread("F:/Registering/data/DroneRGBT/Train/Infrared/1000R.jpg")
image1=cv2.resize(image1,(512,512))
image2=cv2.resize(image2,(512,512))
# 创建ORB对象
# 创建SIFT对象
sift = cv2.xfeatures2d.SIFT_create()

# 检测关键点并计算描述符
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# 特征点匹配
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

# 距离比率测试筛选匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 计算匹配准确率
num_matches = len(good_matches)
num_keypoints1 = len(keypoints1)
matching_accuracy = num_matches / num_keypoints1

# 计算精度
total_distance = 0.0

# 使用RANSAC算法筛选特征点
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
imgOut = cv2.warpPerspective(image2,H,(image1.shape[1],image1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)


filtered_matches = [m for i, m in enumerate(good_matches) if mask[i]]


pt_base = np.float32([keypoints1[m.queryIdx].pt for m in filtered_matches])
pt_Needregistering = np.float32([keypoints2[m.trainIdx].pt for m in filtered_matches])
transforms_pts = cv2.perspectiveTransform(pt_Needregistering.reshape(-1,1,2),H)
pt_registering =[]
for pts in transforms_pts:
    pt_x,pt_y = pts[0]
    print(pt_x,pt_y)
    pt_registering.append(pts[0])
pt_registering = np.array(pt_registering)
RMSE_distance = 0
count = 0
for x,y,z in zip(pt_base,pt_Needregistering,pt_registering):
    count += 1
    print('第%i个正确匹配点（内点）在参考图像的位置为(%i,%i)' % (count, x[0], x[1]),
          '在待配准图像的位置为(%i,%i)'%(y[0],y[1]),
          '其对应的关键点在配准后的图像位置为(%i,%i)'%(z[0],z[1]))
    RMSE_distance += ((x[0]-z[0])**2+(x[1]-z[1])**2)
RMSE_distance = (RMSE_distance/count)**0.5
print("RMSE",RMSE_distance)



# 计算内点率
inlier_mask = mask.ravel().tolist()
num_inliers = sum(inlier_mask)
inlier_ratio = num_inliers / num_matches

print("Number of keypoints in image 1:", num_keypoints1)
print("Number of matches:", num_matches)
print("Matching accuracy:", matching_accuracy)
print("Number of inliers:", num_inliers)
print("Inlier ratio:", inlier_ratio)

cv2.imshow("Matches", imgOut)
cv2.waitKey(0)
cv2.destroyAllWindows()