import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Tuple, Union
from LoFTR.inference import loftrInfer

if __name__ == "__main__":
    # 调用实例
    testInfer = loftrInfer(model_path="F:/Registering/LoFTR/demo/outdoor_ds.ckpt")
    img1_pth = "F:/Registering/data_processing/30.jpg"
    img0_pth = "F:/Registering/data_processing/inf/30R.jpg"

    img0_bgr = cv2.imread(img0_pth)  # 读取图片，bgr格式
    img1_bgr = cv2.imread(img1_pth)


    #旋转变化
    rotation_matrix = cv2.getRotationMatrix2D((img1_bgr.shape[1] / 2, img1_bgr.shape[0] / 2), 30, 1.0)
    rotated_image = cv2.warpAffine(img1_bgr, rotation_matrix, (img1_bgr.shape[1], img1_bgr.shape[0]))

    result,draw_image = testInfer.run(img0_bgr, rotated_image, lenth=200)
    #print(draw_image.shape)

    results = (draw_image, result)
    titles = [ 'mate', 'Registration']

    #print(result.shape)

    for i in range(2):
        plt.subplot(2, 1, i + 1), plt.imshow(results[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

    ''' FPS测试（暂时不用）
    import time

    num_frames = 10  # 要计算的帧数
    print("----- FPS Test  -----")
    start_time = time.time()

    for _ in range(num_frames):
        # 在这里执行关键点检测和匹配的代码
        result, draw_image = testInfer.run(img0_bgr, img1_bgr, lenth=200)


    end_time = time.time()
    total_time = end_time - start_time
    fps = num_frames / total_time

    print("FPS:", fps)
    print("------------------------------------------------------------", '\n')
    '''
