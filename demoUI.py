import sys

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, \
    QLabel, QPushButton, QFileDialog, QComboBox, QGroupBox, QTextEdit, QStackedLayout
from PyQt5.QtWidgets import QToolBar, QToolButton, QStyle, QColorDialog, QFontDialog

from CycleGAN.predict import cyclegan
from LoFTR.inference import loftrInfer
from feature import get_model_from_name


class OutputLogger:
    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, message):
        self.text_edit.moveCursor(QTextCursor.End)
        self.text_edit.insertPlainText(message)

class DemoStackedLayout(QMainWindow):
    def __init__(self, parent=None):
        super(DemoStackedLayout, self).__init__(parent)

        # 设置窗口标题
        self.setWindowTitle('Mul-media image Matching and Registration')
        # 设置窗口大小
        self.resize(1300, 680)

        self.initUi()

    def initUi(self):
        toolBar = QToolBar(self)
        self.addToolBar(Qt.LeftToolBarArea, toolBar)

        btnFeature = self.createButton('基于特征')
        btnFeature.clicked.connect(lambda: self.onButtonClicked(0))

        btnDeepl = self.createButton('深度学习')
        btnDeepl.clicked.connect(lambda: self.onButtonClicked(1))


        btnGAN = self.createButton('风格迁移')
        btnGAN.clicked.connect(lambda: self.onButtonClicked(2))

        toolBar.addWidget(btnFeature)
        toolBar.addWidget(btnDeepl)
        toolBar.addWidget(btnGAN)


        mainWidget = QWidget(self)

        self.mainLayout = QStackedLayout(mainWidget)

        # 添加三个widget,演示三个页面之间的切换

        # 基于特征的方法
        self.mainLayout.addWidget(self.FeatureGroup())
        # 风格迁移
        self.mainLayout.addWidget(self.DeeplGroup())
        # 基于深度学习的方法
        self.mainLayout.addWidget(self.GANGroup())



        mainWidget.setLayout(self.mainLayout)
        # 设置中心窗口
        self.setCentralWidget(mainWidget)

    def createButton(self, text):
        icon = QApplication.style().standardIcon(QStyle.SP_DesktopIcon)
        btn = QToolButton(self)
        btn.setText(text)
        btn.setIcon(icon)
        btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

        return btn

    def onButtonClicked(self, index):
        if index < self.mainLayout.count():
            self.mainLayout.setCurrentIndex(index)


    '''
    基于特征的方法
    '''
    def FeatureGroup(self):

        main_group = QGroupBox("FeatureGroup Box")
        container = QHBoxLayout()

        """    子界面1        
        """
        # 模型及参数选择
        ModelANDParams = QGroupBox("Model and SUB-Params choose")
        V_layout1 = QVBoxLayout()

        # 创建模型超参数
        # 创建一个QComboBox控件
        self.algorithm_combobox = QComboBox()
        # 添加算法选项（sift和surf）
        self.algorithm_combobox.addItems(["SIFT", "SURF", "ORB", "BRISK"])
        self.selected_algorithm_label = QLabel("Feature methods")
        # 匹配超参数NoneType
        self.matchNoneType_combobox = QComboBox()
        self.matchNoneType_combobox.addItems(["cv2.NORM_L2", "cv2.NORM_L1", "cv2.NORM_HAMMING"])
        self.selected_matchNoneType_label = QLabel("Distance type")
        #距离比率阈值
        self.distance_threshold_combobox = QComboBox()
        self.distance_threshold_combobox.addItems(["0.85","0.9","0.8","0.75"])
        self.selected_distance_threshold_label = QLabel("Distance threshold")
        # 匹配超参数RANSAC
        self.matchRANSAC_combobox = QComboBox()
        self.matchRANSAC_combobox.addItems(["True","False"])
        self.selected_matchRANSAC_label = QLabel("RANSAC to filter matching pairs")
        # 匹配超参数RANSAC
        self.ransacReprojThreshold_combobox = QComboBox()
        self.ransacReprojThreshold_combobox.addItems(["4","5"])
        self.selected_ransacReprojThreshold_label = QLabel("RANSAC  Re-projection threshold")

        self.button3 = QPushButton("RUN")
        self.button3.clicked.connect(self.OutImage)

        # 创建一个垂直布局，并将QComboBox控件和标签添加到其中
        V_layout1.addWidget(self.selected_algorithm_label)
        V_layout1.addWidget(self.algorithm_combobox)
        V_layout1.addWidget(self.selected_matchNoneType_label)
        V_layout1.addWidget(self.matchNoneType_combobox)
        V_layout1.addWidget(self.selected_distance_threshold_label)
        V_layout1.addWidget(self.distance_threshold_combobox)
        V_layout1.addWidget(self.selected_matchRANSAC_label)
        V_layout1.addWidget(self.matchRANSAC_combobox)
        V_layout1.addWidget(self.selected_ransacReprojThreshold_label)
        V_layout1.addWidget(self.ransacReprojThreshold_combobox)

        V_layout1.addStretch(4)
        V_layout1.addWidget(self.button3)
        # 监听QComboBox控件的currentIndexChanged事件

        ModelANDParams.setLayout(V_layout1)
        """    子界面2        
        """
        input_image = QGroupBox("Input")
        V_layout2 = QVBoxLayout()
        # 第一张照片的按钮
        self.label1 = QLabel()
        self.label1.setAlignment(Qt.AlignCenter)
        self.button1 = QPushButton("Choose Fixing Image ")
        self.button1.clicked.connect(self.selectImage1)
        V_layout2.addWidget(self.label1)
        V_layout2.addWidget(self.button1)
        # 第二张照片的按钮
        self.label2 = QLabel()
        self.label2.setAlignment(Qt.AlignCenter)
        self.button2 = QPushButton("Choose Moving Image")
        self.button2.clicked.connect(self.selectImage2)
        V_layout2.addWidget(self.label2)
        V_layout2.addWidget(self.button2)

        input_image.setLayout(V_layout2)
        """    子界面3        
        """
        # 输出结果
        output_image=QGroupBox("Images of Matching and Registering")
        V_layout3 = QVBoxLayout()
        self.label3 = QLabel()
        self.label3.setAlignment(Qt.AlignCenter)
        self.label4 = QLabel()
        self.label4.setAlignment(Qt.AlignCenter)
        V_layout3.addWidget(self.label3)
        V_layout3.addWidget(self.label4)

        output_image.setLayout(V_layout3)
        """    子界面4        
        """
        text_output=QGroupBox("model evalution")
        V_layout4 = QVBoxLayout()


        self.text_edit = QTextEdit()


        V_layout4.addWidget(self.text_edit)
        text_output.setLayout(V_layout4)
        sys.stdout = OutputLogger(self.text_edit)
        """    总体布局        
        """
        container.addWidget(ModelANDParams)
        container.addWidget(input_image,stretch=3)
        container.addWidget(output_image,stretch=5)
        container.addWidget(text_output,stretch=1)
        main_group.setLayout(container)

        return main_group
    '''
    功能
    '''
    def selectImage1(self):
        # 打开文件对话框选择第一张图片
        filePath, _ = QFileDialog.getOpenFileName(self, "选择第一张图片", "", "Image Files (*.png *.jpg *.bmp)")
        if filePath:
            # 加载第一张图片并显示在界面中
            self.filePath1 = filePath
            img = QPixmap(filePath).scaled(500, 400, Qt.KeepAspectRatio)
            self.label1.setPixmap(img)

    def selectImage2(self):
        # 打开文件对话框选择第二张图片
        filePath, _ = QFileDialog.getOpenFileName(self, "选择第二张图片", "", "Image Files (*.png *.jpg *.bmp)")
        if filePath:
            # 加载第二张图片并显示在界面中
            self.filePath2 = filePath
            img = QPixmap(filePath).scaled(500, 400, Qt.KeepAspectRatio)
            self.label2.setPixmap(img)

    def OutImage(self):
        # 打开文件对话框选择第二张图片
        filePath1=self.filePath1
        filePath2=self.filePath2

        # 读取img1和img2
        img1 = cv2.imread(filePath1)
        img2 = cv2.imread(filePath2)

        model=self.algorithm_combobox.currentText()
        normType=self.matchNoneType_combobox.currentText()
        distance_threshold=self.distance_threshold_combobox.currentText()
        RANSAC = self.matchRANSAC_combobox.currentText()
        ransacReprojThreshold=self.ransacReprojThreshold_combobox.currentText()

        dict_normType={"cv2.NORM_L2":4,"cv2.NORM_L1":2,"cv2.NORM_HAMMING":6}
        dict_distance_threshold={"0.75":0.75,"0.8":0.80,"0.85":0.85,"0.9":0.90}
        dict_RANSAC={"False":False,"True":True}
        dict_ransacReprojThreshold={"4":4,"5":5}

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

        match=get_model_from_name[model](img1,img2,dict_normType[normType],
                                         dict_distance_threshold[distance_threshold],
                                         dict_RANSAC[RANSAC],
                                         dict_ransacReprojThreshold[ransacReprojThreshold])

        img3,img4 = match.MatchANDOverlapping()

        # 将读取到的灰度图片显示在（2，1）的位置上
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        img3 = QImage(img3.data, img3.shape[1], img3.shape[0], QImage.Format_RGB888)
        img3 = QPixmap.fromImage(img3).scaled(800, 600, Qt.KeepAspectRatio)
        self.label3.setPixmap(img3)

        img4 = QImage(img4.data, img4.shape[1], img4.shape[0],QImage.Format_Grayscale8)
        img4 = QPixmap.fromImage(img4).scaled(800, 600, Qt.KeepAspectRatio)
        self.label4.setPixmap(img4)

    '''
    基于深度学习方法
    '''
    def DeeplGroup(self):
        '''
        界面1
        '''
        main_group = QGroupBox("DeeplGroup Box")
        container = QHBoxLayout()
        ModelANDParams = QGroupBox("Model  choose")
        V_layout1 = QVBoxLayout()
        # 创建模型超参数
        # 创建一个QComboBox控件
        self.algorithm_combobox_deepl = QComboBox()
        # 添加算法选项（sift和surf）
        self.algorithm_combobox_deepl.addItems(["LoFTR"])
        self.selected_algorithm_label_deepl = QLabel()

        self.button3_deepl = QPushButton("RUN")
        self.button3_deepl.clicked.connect(self.OutImage_deepl)

        # 创建一个垂直布局，并将QComboBox控件和标签添加到其中
        V_layout1.addWidget(self.algorithm_combobox_deepl)
        V_layout1.addWidget(self.selected_algorithm_label_deepl)

        V_layout1.addStretch(4)
        V_layout1.addWidget(self.button3_deepl)
        # 监听QComboBox控件的currentIndexChanged事件

        ModelANDParams.setLayout(V_layout1)

        """    子界面2        """
        input_image = QGroupBox("Input")
        V_layout2 = QVBoxLayout()
        # 第一张照片的按钮
        self.label1_deepl = QLabel()
        self.label1_deepl.setAlignment(Qt.AlignCenter)
        self.button1_deepl = QPushButton("Choose Fixing Image ")
        self.button1_deepl.clicked.connect(self.selectImage1_deepl)
        V_layout2.addWidget(self.label1_deepl)
        V_layout2.addWidget(self.button1_deepl)
        # 第二张照片的按钮
        self.label2_deepl = QLabel()
        self.label2_deepl.setAlignment(Qt.AlignCenter)
        self.button2_deepl = QPushButton("Choose Moving Image")
        self.button2_deepl.clicked.connect(self.selectImage2_deepl)
        V_layout2.addWidget(self.label2_deepl)
        V_layout2.addWidget(self.button2_deepl)

        input_image.setLayout(V_layout2)

        """    子界面3        
        """
        # 输出结果
        output_image=QGroupBox("Images of Matching and Registering")
        V_layout3 = QVBoxLayout()
        self.label3_deepl = QLabel()
        self.label3_deepl.setAlignment(Qt.AlignCenter)
        self.label4_deepl = QLabel()
        self.label4_deepl.setAlignment(Qt.AlignCenter)
        V_layout3.addWidget(self.label3_deepl)
        V_layout3.addWidget(self.label4_deepl)

        output_image.setLayout(V_layout3)

        container.addWidget(ModelANDParams)
        container.addWidget(input_image,stretch=2)
        container.addWidget(output_image,stretch=5)

        main_group.setLayout(container)

        return main_group

    def selectImage1_deepl(self):
        # 打开文件对话框选择第一张图片
        filePath, _ = QFileDialog.getOpenFileName(self, "选择第一张图片", "", "Image Files (*.png *.jpg *.bmp)")
        if filePath:
            # 加载第一张图片并显示在界面中
            self.filePath1_deepl = filePath
            img = QPixmap(filePath).scaled(500, 400, Qt.KeepAspectRatio)
            self.label1_deepl.setPixmap(img)

    def selectImage2_deepl(self):
        # 打开文件对话框选择第二张图片
        filePath, _ = QFileDialog.getOpenFileName(self, "选择第二张图片", "", "Image Files (*.png *.jpg *.bmp)")
        if filePath:
            # 加载第二张图片并显示在界面中
            self.filePath2_deepl = filePath
            img = QPixmap(filePath).scaled(500, 400, Qt.KeepAspectRatio)
            self.label2_deepl.setPixmap(img)

    def OutImage_deepl(self):
        # 打开文件对话框选择第二张图片
        filePath1=self.filePath1_deepl
        filePath2=self.filePath2_deepl

        # 读取img1和img2
        img1 = cv2.imread(filePath1)
        img2 = cv2.imread(filePath2)

        model=self.algorithm_combobox_deepl.currentText()
        testInfer = loftrInfer(model_path="F:/Registering/LoFTR/demo/outdoor_ds.ckpt")
        img4_deepl, img3_deepl = testInfer.run(img1, img2, lenth=200)



        # 将读取到的灰度图片显示在（2，1）的位置上

        img3_deepl = cv2.cvtColor(img3_deepl, cv2.COLOR_BGR2RGB)
        h, w, ch = img3_deepl.shape
        bytesPerLine = ch * w
        img3_deepl = QImage(img3_deepl.data, img3_deepl.shape[1], img3_deepl.shape[0],bytesPerLine, QImage.Format_RGB888)
        img3_deepl = QPixmap.fromImage(img3_deepl).scaled(800, 600, Qt.KeepAspectRatio)
        self.label3_deepl.setPixmap(img3_deepl)

        h1, w1, ch1 = img4_deepl.shape
        bytesPerLine1 = ch1 * w1
        img4_deepl = QImage(img4_deepl.data, img4_deepl.shape[1], img4_deepl.shape[0],bytesPerLine1,QImage.Format_RGB888)
        img4_deepl = QPixmap.fromImage(img4_deepl).scaled(800, 600, Qt.KeepAspectRatio)
        self.label4_deepl.setPixmap(img4_deepl)

    def GANGroup(self):
        '''
        界面1
        '''
        main_group = QGroupBox("GanGroup Box")
        container = QHBoxLayout()
        ModelANDParams = QGroupBox("Model  choose")
        V_layout1 = QVBoxLayout()
        # 创建模型超参数
        # 创建一个QComboBox控件
        self.algorithm_combobox_gan = QComboBox()
        # 添加算法选项（sift和surf）
        self.algorithm_combobox_gan.addItems(["CycleGAN"])
        self.selected_algorithm_label_gan = QLabel()

        self.button3_gan = QPushButton("RUN")
        self.button3_gan.clicked.connect(self.OutImage_gan)

        # 创建一个垂直布局，并将QComboBox控件和标签添加到其中
        V_layout1.addWidget(self.algorithm_combobox_gan)
        V_layout1.addWidget(self.selected_algorithm_label_gan)

        V_layout1.addStretch(4)
        V_layout1.addWidget(self.button3_gan)
        # 监听QComboBox控件的currentIndexChanged事件

        ModelANDParams.setLayout(V_layout1)

        """    子界面2        """
        input_image = QGroupBox("Input")
        V_layout2 = QVBoxLayout()
        # 第一张照片的按钮
        self.label1_gan = QLabel()
        self.label1_gan.setAlignment(Qt.AlignCenter)
        self.button1_gan = QPushButton("Choose Vis Image ")
        self.button1_gan.clicked.connect(self.selectImage1_gan)
        V_layout2.addWidget(self.label1_gan)
        V_layout2.addWidget(self.button1_gan)


        input_image.setLayout(V_layout2)

        """    子界面3        
        """
        # 输出结果
        output_image = QGroupBox("fake vis Images")
        V_layout3 = QVBoxLayout()
        self.label3_gan = QLabel()
        self.label3_gan.setAlignment(Qt.AlignCenter)

        V_layout3.addWidget(self.label3_gan)


        output_image.setLayout(V_layout3)

        container.addWidget(ModelANDParams)
        container.addWidget(input_image, stretch=2)
        container.addWidget(output_image, stretch=5)

        main_group.setLayout(container)

        return main_group

    def selectImage1_gan(self):
        # 打开文件对话框选择第一张图片
        filePath, _ = QFileDialog.getOpenFileName(self, "选择第一张图片", "", "Image Files (*.png *.jpg *.bmp)")
        if filePath:
            # 加载第一张图片并显示在界面中
            self.filePath1_gan = filePath
            img = QPixmap(filePath).scaled(512, 512, Qt.KeepAspectRatio)
            self.label1_gan.setPixmap(img)

    def OutImage_gan(self):
        # 打开文件对话框选择第二张图片
        filePath1 = self.filePath1_gan

        # 读取img1和img2
        img1_dir = filePath1

        model = self.algorithm_combobox_gan.currentText()

        img3_gan = cyclegan(input_img=img1_dir)
        outputRs = img3_gan.permute(0, 2, 3, 1)
        mm = outputRs.cpu().detach().numpy() * 255
        mm = mm.astype("uint8").reshape(512, 512, 3)


        # 将读取到的灰度图片显示在（2，1）的位置上

        img3_gan = cv2.cvtColor(mm, cv2.COLOR_BGR2RGB)
        h, w, ch = img3_gan.shape
        bytesPerLine = ch * w
        img3_gan = QImage(img3_gan.data, img3_gan.shape[1], img3_gan.shape[0], QImage.Format_RGB888)
        img3_gan = QPixmap.fromImage(img3_gan).scaled(512, 512, Qt.KeepAspectRatio)
        self.label3_gan.setPixmap(img3_gan)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DemoStackedLayout()
    window.show()
    sys.exit(app.exec())