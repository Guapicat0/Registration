import cv2
import numpy as np

def registration_test(keypoints1,
                      keypoints2,
                      H,
                      filtered_matches ):

    pt_base = np.float32([keypoints1[m.queryIdx].pt for m in filtered_matches])
    pt_Needregistering = np.float32([keypoints2[m.trainIdx].pt for m in filtered_matches])
    transforms_pts = cv2.perspectiveTransform(pt_Needregistering.reshape(-1, 1, 2), H)
    pt_registering = []
    for pts in transforms_pts:
        pt_x, pt_y = pts[0]
        pt_registering.append(pts[0])
    pt_registering = np.array(pt_registering)

    RMSE_distance = 0
    count = 0

    print("----- registration Test Results -----------------------------------")
    for x, y, z in zip(pt_base, pt_Needregistering, pt_registering):
        count += 1
        print('第%i个正确匹配点（内点）在参考图像的位置为(%i,%i)' % (count, x[0], x[1]),
              '在待配准图像的位置为(%i,%i)' % (y[0], y[1]),
              '其对应的关键点在配准后的图像位置为(%i,%i)' % (z[0], z[1]))
        RMSE_distance += ((x[0] - z[0]) ** 2 + (x[1] - z[1]) ** 2)

    RMSE = (RMSE_distance / count) ** 0.5
    print("参考图像与配准后图像正确匹配点的","RMSE:", RMSE)
    print("--------------------------------------------------------------------")

