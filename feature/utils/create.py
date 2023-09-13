import cv2

def DetectANDCompute(image,model):
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if model=="SIFT":
        model = cv2.xfeatures2d_SIFT.create()
    if model=="ORB":
        model = cv2.ORB_create()
    if model=="SURF":
        model = cv2.xfeatures2d_SURF.create()
    if model=="BRISK":
        model = cv2.BRISK_create()


    kp, des = model.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(image, kp, None)
    return kp_image, kp, des


