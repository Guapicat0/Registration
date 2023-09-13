import cv2
from feature.utils.create import DetectANDCompute
import numpy as np
from feature.utils.registration import ImageAlignment
from feature.utils.BFmatches import BF_match,Knn_match
from feature.utils.match_test import match_test
from feature.utils.registration_test import registration_test

class SURF():
    def __init__(self,img1,img2,normType,distance_threshold,RANSAC,ransacReprojThreshold):
        super(SURF, self).__init__()
        self.img1=cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
        self.img2=cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
        self.normType=normType
        self.model="SURF"
        self.distance_threshold=distance_threshold
        self.RANSAC=RANSAC
        self.ransacReprojThreshold=ransacReprojThreshold



    def MatchANDOverlapping(self):
        # 匹配
        _, keypoint1, descriptor1 = DetectANDCompute(self.img1,model=self.model)
        _, keypoint2, descriptor2 = DetectANDCompute(self.img2,model=self.model)

        # Knn_match,BF_match都可以
        goodMatch = Knn_match(descriptor1, descriptor2,self.normType,self.distance_threshold)

        # 配准
        imgOut, H, status = ImageAlignment(self.img1, self.img2,goodMatch,keypoint1,keypoint2,self.ransacReprojThreshold)
        #overlapping = cv2.addWeighted(self.img1, 0.6, imgOut, 0.4, 0)
        overlapping=imgOut
        filtered_matches = [m for i, m in enumerate(goodMatch) if status[i]]

        # 测试


        match_test(img1=self.img1, img2=self.img2,
                                    kp1=keypoint1, kp2=keypoint2,
                                    des1=descriptor1, des2=descriptor2,
                                    goodMatch=goodMatch,
                                    status=status,
                                    model=self.model,
                                    normType=self.normType,
                                    distance_threshold=self.distance_threshold)

        registration_test(keypoints1=keypoint1,keypoints2=keypoint2,H=H,filtered_matches=filtered_matches)

        if self.RANSAC == True:
            match = cv2.drawMatches(self.img1, keypoint1, self.img2, keypoint2, filtered_matches, None, flags=2)
        else :
            match = cv2.drawMatches(self.img1, keypoint1, self.img2, keypoint2, goodMatch, None, flags=2)


        return match,overlapping






