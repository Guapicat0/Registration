import cv2
import matplotlib.pyplot as plt
from feature import get_model_from_name
from feature.utils.fps_test import fps_test
if __name__ == "__main__":

    model="SIFT"
    img1 = cv2.imread("F:/Registering/data_processing/inf/30R.jpg")
    img2 = cv2.imread("F:/Registering/data_processing/30_vis.jpg")
    img1 = cv2.resize(img1,(512,512))
    img2 = cv2.resize(img2, (512, 512))
    fps_test(model=model,img1=img1,img2=img2)

    model=get_model_from_name[model](img1,img2,cv2.NORM_L2,distance_threshold=0.9,RANSAC=True,ransacReprojThreshold=5)
    match,overlapping=model.MatchANDOverlapping()

    """
    create([, normType[, crossCheck]]) -> retval
    .   @brief Brute-force matcher create method.
    .   @param normType One of NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. L1 and L2 norms are
    .   preferable choices for SIFT and SURF descriptors, NORM_HAMMING should be used with ORB, BRISK and
    .   BRIEF, NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor
    .   description).
    .   @param crossCheck If it is false, this is will be default BFMatcher behaviour when it finds the k
    .   nearest neighbors for each query descriptor. If crossCheck==true, then the knnMatch() method with
    .   k=1 will only return pairs (i,j) such that for i-th query descriptor the j-th descriptor in the
    .   matcher's collection is the nearest and vice versa, i.e. the BFMatcher will only return consistent
    .   pairs. Such technique usually produces best results with minimal number of outliers when there are
    .   enough matches. This is alternative to the ratio test, used by D. Lowe in SIFT paper.
    """

    results = (img1, img2, match, overlapping)
    titles = ['RGB', 'Infrared', 'mate', 'Registration']

    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(results[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])


    plt.show()

    cv2.imshow("Matches", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()