import time
import cv2

def fps_test(model,img1,img2):
    # 创建SIFT对象
    if model=="SIFT":
        model = cv2.xfeatures2d_SIFT.create()
    if model=="ORB":
        model = cv2.ORB_create()
    if model=="SURF":
        model = cv2.xfeatures2d_SURF.create()
    if model=="BRISK":
        model = cv2.BRISK_create()

    num_frames = 10  # 要计算的帧数
    print("----- FPS Test  -----")
    start_time = time.time()

    for _ in range(num_frames):
        # 在这里执行关键点检测和匹配的代码
        keypoints1, descriptors1 = model.detectAndCompute(img1, None)
        keypoints2, descriptors2 = model.detectAndCompute(img2, None)
        bf = cv2.BFMatcher()
        matches = bf.match(descriptors1, descriptors2)


    end_time = time.time()
    total_time = end_time - start_time
    fps = num_frames / total_time

    print("FPS:", fps)
    print("------------------------------------------------------------",'\n')

if __name__ == "__main__":

    model="BRISK"
    img1 = cv2.imread("F:/Registering/data/DroneRGBT/Train/RGB/3.jpg")
    img2 = cv2.imread("F:/Registering/data/DroneRGBT/Train/Infrared/3R.jpg")
    fps_test(model=model,img1=img1,img2=img2)

