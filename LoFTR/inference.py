import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from LoFTR.src.loftr import LoFTR, default_cfg
from typing import Tuple, Union


def show_image(image: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).show()



def drawMatches(img_left, img_right, kps_left, kps_right):
    H, status = cv2.findHomography(kps_right, kps_left, cv2.RANSAC,ransacReprojThreshold=4)
    # 获取图片宽度和高度
    matches_mask = status.ravel().tolist()
    # inlier_ratio 模型评估，内点比例
    print("inlier point matches:", np.sum(matches_mask))

    """利用以获得的单应性矩阵进行变透视换"""
    image = cv2.warpPerspective(img_right, H, (img_left.shape[1], img_left.shape[0]))  # (w,h

    return image



# 定义一个类，方便调用
class loftrInfer(object):
    ''' 初始化之后，仅需调用run函数 '''

    def __init__(self, model_path):
        '''
            初始化，输入参数:
                    model_path: 模型地址
        '''
        self.matcher = LoFTR(config=default_cfg)  # 初始化模型
        self.matcher.load_state_dict(torch.load(model_path)['state_dict'])  # 下载训练好的模型文件，可选indoor_ds 、outdoor_ds
        self.matcher = self.matcher.eval().cuda()  # cuda验证

    def _infer_run(self, img0_raw, img1_raw):
        '''
            推理单对图片，输入参数:
                    img0_raw 、img1_raw    numpy.ndarray类型，单通道图像
                    返回值:
                    np_result/False     False 或 (n,5)推理结果，numpy.ndarray类型； 格式为(p1x,p1y,p2x,p2y,conf)

        '''
        img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.  # 转torch格式，cuda ，归一到0-1
        img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
        batch = {'image0': img0, 'image1': img1}  # 模型输入为字典，加载输入

        # Inference with LoFTR and get prediction 开始推理
        with torch.no_grad():
            self.matcher(batch)  # 网络推理
            mkpts0 = batch['mkpts0_f'].cpu().numpy()  # (n,2) 0的结果 -特征点
            mkpts1 = batch['mkpts1_f'].cpu().numpy()  # (n,2) 1的结果 -特征点
            mconf = batch['mconf'].cpu().numpy()  # (n,)      置信度


        print("匹配点:",mconf.shape[0])

        # 筛选，需要四个以上的匹配点才能得到单应性矩阵
        if mconf.shape[0] < 4:
            return False

        mconf = mconf[:, np.newaxis]  # 末尾增加新维度
        np_result = np.hstack((mkpts0, mkpts1, mconf))  # 水平拼接
        #print(np_result.shape)
        list_result = list(np_result)

        def key_(a):
            return a[-1]

        list_result.sort(key=key_, reverse=True)  # 按得分从大到小排序
        np_result = np.array(list_result)


        return np_result

    def _points_filter(self, np_result):
        '''
            进行特征值筛选，输入参数:
                    np_result  推理结果(n,5)
                    lenth       -1   不进行筛选，取全部
                                >0  取前nums个
                    use_kmeans   bool类型: 0 - 不使用
                                        1 -  使用聚类，取最多一类
        '''
        '''' 本来想直接取前多少个进行矩阵运算，但发现聚类后好些 '''
        #lenth = min(lenth, np_result.shape[0])  # 选最大200个置信度较大的点对
        lenth = int(np_result.shape[0]*0.8)
        if lenth < 4: lenth = 4

        mkpts0 = np_result[:lenth, :2].copy()
        mkpts1 = np_result[:lenth, 2:4].copy()


        return mkpts0, mkpts1

    def _draw_matchs(self, img1, img2, p1s, p2s, mid_space=10):
        '''
            画匹配点并显示，分别输入图像、特征点  ; mid_space间隔
            输入参数:
                    img1,img2 彩色图像; p1s,p2s 分别的特征点位置
                    mid_space 左右图显示间隔
                    if_save 保存否
        '''
        h, w = img1.shape[:2]
        show = cv2.resize(img1, (2 * w + mid_space, h))
        show.fill(0)
        show[:, :w] = img1.copy()
        show[:, w + 10:] = img2.copy()
        p1s = p1s.astype(np.int)
        p2s = p2s.astype(np.int)

        for i in range(p1s.shape[0]):
            p1 = tuple(p1s[i])
            p2 = (p2s[i][0] + w + mid_space, p2s[i][1])
            cv2.line(show, p1, p2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)),
                     1)  # 画线

        return show


    def run(self, img0_bgr, img1_bgr, lenth=200):
        '''
            只需要调用该函数，完成推理+拼接
            输入参数 :
                img0_bgr , img1_bgr  彩色图像，默认左右
                lenth       -1   不进行筛选，取全部
                            >0  取前nums个
                use_kmeans   bool类型: 0 - 不使用
                                    1 -  使用聚类，取最多一类
                if_draw      bool类型  是否绘制特征点匹配图像
                if_save      bool类型  是否保存特征点匹配图像
                stitch_method    拼接方法选择， 0 为l2n方法 ； 其他为简单方法
            返回值:
                image  拼接图像
        '''
        img0_bgr = cv2.resize(img0_bgr, (640, 512))  # 统一尺寸为640x480
        img1_bgr = cv2.resize(img1_bgr, (640, 512))

        img0_raw = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2GRAY)  # 转灰度，网络输入的是单通道图
        img1_raw = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)

        np_result = self._infer_run(img0_raw, img1_raw)  # 推理
        if np_result is False:
            print("特征点数量不够！！！")
            return False

        lenth=np_result.shape[0]*0.8
        mkpts0, mkpts1 = self._points_filter(np_result)  # 特征点筛选
        print("正确匹配点",mkpts0.shape[0])

        draw_image= self._draw_matchs(img0_bgr, img1_bgr, mkpts0, mkpts1, mid_space=10)
        image = drawMatches(img0_bgr, img1_bgr, mkpts0, mkpts1)

        return image,draw_image



