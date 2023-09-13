import cv2
from feature.utils.create import DetectANDCompute
from feature.utils.BFmatches import Knn_match,BF_match

import numpy as np
# distance_threshold默认阈值为300,越小匹配点越正确

# 测试

def match_test(img1, img2, kp1, kp2, des1, des2, goodMatch,status,model, normType, distance_threshold):
    print("----- Feature Match Test Results -----------------------------------")

    # 参考图像与待配准/待匹配图像关键点数量，及使用蛮力匹配获得的匹配对数
    print("1.参考图像与待配准/待匹配图像关键点数量，以及使用蛮力匹配获得的匹配对数")
    num_keypoints1 = len(kp1)
    num_keypoints2 = len(kp2)
    print("   Number of keypoints in image 1:", num_keypoints1)
    print("   Number of keypoints in image 2:", num_keypoints2)
    print("   Brute Froce matches:",min(num_keypoints1,num_keypoints2))

    # 粗匹配对数/使用距离阈值进行初筛选
    print("2.粗匹配对数/使用距离进行初筛选获得的匹配对数")
    num_matches = len(goodMatch)
    print("   Correct Matching Points-using distance_threshold matches:", num_matches)

    # 特征点重复性，参考图像与待配准/待匹配图像的重复关键点个数
    print("3.特征点重复性，即参考图像与待配准图像/待匹配图像的重复关键点个数")
    keypoints1_coordinates = [keypoint.pt for keypoint in kp1]
    keypoints2_coordinates = [keypoint.pt for keypoint in kp2]
    num_duplicates = len(set(keypoints1_coordinates) & set(keypoints2_coordinates))
    print("   Duplicate keypoints matches:", num_duplicates)


    # 匹配正确率(阈值筛选）
    print("4.使用距离阈值筛选后获得的粗匹配对数占参考图像及与待配准图像/待匹配图像关键点的比重")
    matching_success_rate=num_matches*2/(num_keypoints1+num_keypoints2)
    print("   Distance_threshold matches*2/ keypoint1+keypoint2:",matching_success_rate)

    # 内点计算
    print("5.与内点(使用RANSAC)进行误差剔除方法获得相关的指标，包括"
          "(1)内点个数，(2)内点率（内点数占粗匹配（距离阈值筛选的的匹配对）的比重，"
          "(3)内点占参考图像及待配准图像/待匹配图像关键点的比重")
    if len(goodMatch) > 4:
        matches_mask = status.ravel().tolist()
        # inlier_ratio 模型评估，内点比例
        print("   inlier point(RANSAC) matches:",np.sum(matches_mask))
        inlier_ratio = np.sum(matches_mask) / len(matches_mask)
        print("   inlier_ratio/MSR :(inlier point matches/ distance_threshold matches):",inlier_ratio)

        num_keypoints1 = len(kp1)
        num_keypoints2 = len(kp2)

        print("   inlier point matches*2/ keypoint1+keypoint2:",np.sum(matches_mask)*2/(num_keypoints1+num_keypoints2))

    print("------------------------------------------------------------",'\n')



    print("----- Feature Match Rotated Test Results -----------------------------------")

    # 特征点鲁棒性
    rotation_matrix = cv2.getRotationMatrix2D((img1.shape[1] / 2, img1.shape[0] / 2), 30, 1.0)
    rotated_image = cv2.warpAffine(img1, rotation_matrix, (img1.shape[1], img1.shape[0]))
    _, keypoints_rotated, descriptors_rotated = DetectANDCompute(rotated_image, model=model)


    good_matches_rotated = Knn_match(descriptors_rotated, des2, normType,distance_threshold)



    num_matches_rotated = len(good_matches_rotated)
    print("Number of matches with rotated image:", num_matches_rotated)
    print("------------------------------------------------------------",'\n')






