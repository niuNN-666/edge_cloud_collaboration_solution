import os
import traceback
import numpy as np
import time
import cv2
from input_reader import InputReader
# import sys
# from tracker import Tracker
# from EAR import eye_aspect_ratio
# from MAR import mouth_aspect_ratio
from S3_api import S3 # 存储obs
import xml.etree.ElementTree as ET # 生成xml文件


import torch
import os

from models.experimental import attempt_load
from utils1.general import check_img_size
from tempfile import NamedTemporaryFile
from utils1.torch_utils import TracedModel
from detect import detect
# from model_service.pytorch_model_service import PTServingBaseService
#import logging

from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkcore.http.http_config import HttpConfig
# 导入指定云服务的库 huaweicloudsdk{service}
from huaweicloudsdkocr.v1.region.ocr_region import OcrRegion
from huaweicloudsdkocr.v1 import *
"""
定义了全局变量user,后续创建该用户的相关文件和上传
显示置信度大于0.85的检测框，保存和生成置信度大于0.9的xml文件，并发给obs
"""

global user
user = 'niu'

# 大于此阈值进行显示画面，用于准确度的可视化
showFrame_acc = 0.85  # conf
# 大于此阈值保存对应的图片 我们取得是0.9+置信度的
save_acc = 0.9

class fatigue_driving_detection():
    def __init__(self, model_name, model_path):
        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = model_path

        # print(self.video_capture)
        self.capture = 0
        self.width = 1920
        self.height = 1080
        self.fps = 30
        self.first = True

        # print("into init")
        
        self.look_around_frame = 0
        self.eyes_closed_frame = 0
        self.mouth_open_frame = 0
        self.use_phone_frame = 0

        self.frame_3s = 9

        self.weights = "best.pt"
        #self.imgsz = 640
        self.imgsz = 256

        # 存储难例的列表
        self.hard_frames_info = []
        self.device = 'cpu'  # 大赛后台使用CPU判分

        model = attempt_load(model_path, map_location=self.device)
        self.stride = int(model.stride.max())
        self.imgsz = check_img_size(self.imgsz, s=self.stride)

        self.model = TracedModel(model, self.device, self.imgsz)
        
        self.behavior_info = []  # 用于记录行为信息的列表，因为需要按顺序输出依次发生的行为，存储在列表里

        self.total_duration = 0  # 记录总的检测时间

        self.need_reinit = 0
        self.failures = 0

    # 前处理函数
    def _preprocess(self, data):
        # preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                try:
                    try:
                        with open(self.capture, 'wb') as f:
                            file_content_bytes = file_content.read()
                            f.write(file_content_bytes)

                    except Exception:
                        return {"message": "There was an error loading the file"}

                    # self.capture = self.temp.name  # Pass temp.name to VideoCapture()
                except Exception:
                    return {"message": "There was an error processing the file"}
        return 'ok'
    
    # 推断函数    
    def _inference(self, data):
        print("into inference")

        # 读取视频的宽度、高度、帧率
        self.cap = cv2.VideoCapture(self.capture)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # print(self.width)
        # print(self.height)
        # print(self.fps)
        
        # print("tracker init OK-----------------------------------------------")

        self.input_reader = InputReader(self.capture, 0, self.width, self.height, self.fps)
        source_name = self.input_reader.name

        
        frame_count = 0
        frame_timer = 0
        #这里必须要清零，否则会带来上一个视频的数据
        self.behavior_info = []  # 用于记录行为信息的列表，因为需要按顺序输出依次发生的行为，存储在列表里
        self.total_duration = 0  # 记录总的检测时间

        three_frames = [0, 0, 0]   # 用于保存最近3帧的行为分类
        start = 0 # 最近行为的起始帧数
        stop = 0 # 最近行为的结束帧数
        behavior_now_counting = 0  # 目前正在计数的行为类型，默认初始化为0：normal

        now = time.time()

        # 循环读取视频文件里的每一帧
        while self.input_reader.is_open():
            if not self.input_reader.is_open() or self.need_reinit == 1:
                self.input_reader = InputReader(self.capture, 0, self.width, self.height, self.fps, use_dshowcapture=False, dcap=None)
                if self.input_reader.name != source_name:
                    print(f"Failed to reinitialize camera and got {self.input_reader.name} instead of {source_name}.")
                    # sys.exit(1)
                self.need_reinit = 2
                #time.sleep(0.02)
                continue
            if not self.input_reader.is_ready():
                #time.sleep(0.02)
                continue
            
            # 记录当前是第几帧
            frame_timer = frame_timer + 1
            
            # 读取1帧
            ret, frame = self.input_reader.read()

            self.need_reinit = 0

            #开始对这一帧进行判断处理
            try:
                if frame is not None:
                    # 剪裁主驾驶位
                    # frame = frame[:, 600:1920, :]
                    # frame = frame[:, half_width:full_width, :]
                    # print("into frame")
                    # 检测驾驶员是否接打电话 以及低头的人脸
                    bbox = detect(self.model, frame, self.stride, self.imgsz)
                    #从这里开始新的复赛判断程序，fusai
                    have_phone_in_frame = 0
                    have_mouth_in_frame = 0
                    have_eye_in_frame = 0
                    have_leftright_in_frame = 0
                    have_headdown_in_frame = 0
                    confidence_value = 0.0
                    # 每帧的行为默认为normal
                    have_normal_in_frame = 1
                    behavior_of_frame = 0

                    # 根据类别添加相应的标签
                    label_map = {
                        0: "look_around",
                        1: "phone",
                        2: "open_mouth",
                        3: "eye_closed",
                        4: "normal",
                        5: "headdown"
                    }
                    # cv2.namedWindow("Fatigue_driving_Detection", 1280,720)  # 创建窗口
                    cv2.namedWindow("Fatigue_driving_Detection", cv2.WINDOW_NORMAL)  # 使用WINDOW_NORMAL以允许调整窗口大小
                    # print("create window")
                    # 设置窗口的新大小（宽度x高度）
                    window_width = 640
                    window_height = 720
                    cv2.resizeWindow("Fatigue_driving_Detection", window_width, window_height)
                    cv2.imshow("Fatigue_driving_Detection", frame)

                    # 判断该帧所有的标注框
                    for box in bbox:
                        # print("into for_box")
                        class_id = box[0]
                        center_x, center_y, width, height = box[1][0], box[1][1], box[1][2], box[1][3]

                        # 置信度
                        confidence_tensor = torch.tensor(box[2])  # box2返回tensor()，是pytorch张量表现形式，需要转换浮点
                        confidence_value = round(confidence_tensor.item(),3) # 我选择保留置信度到小数点后三位
                        # confidence_value = box[2].clone().detach().float() # UserWarning：建议我使用sourceTensor.clone().detach()，而不是torch.tensor(box[2])
                            # 但是我不接受它给的建议，return到云端的时候tensor()形式可能遇到麻烦
                        # 将相对坐标转换为像素坐标，yolo和opencv用的坐标定义不同
                        # detect 返回的xy是物体的中心点坐标,wh,是物体的长宽,图像的左上角为原点，向下为h轴正半轴,向右为w正半轴
                        # x = int(center_x * self.width/2)
                        x = int(center_x * self.width)
                        y = int(center_y * self.height)
                        # w = int(width * self.width/2)
                        w = int(width * self.width)
                        h = int(height * self.height)

                        # 计算左上角和右下角坐标，用于确认矩形框
                        x1 = int(x - w / 2)
                        y1 = int(y - h / 2)
                        x2 = x1 + w
                        y2 = y1 + h


                        # 绘制识别框,注意opencv使用的是bgr，可以参照rgb颜色表，只是把b,r通道顺序变了
                        color0 = (255, 255, 255)  # 白色
                        color1 = (0, 255, 0)  # 绿色
                        color2 = (0,255,255)  # 黄色
                        color3 = (0, 0, 255)  # 红色
                        color4 = (255, 0, 255)  # 紫色
                        color5 = (225, 105, 65) # 品蓝
                        color6 = (80, 127, 255)  # 珊瑚粉

                        # 对不同行为的标签赋予不同颜色更加显眼
                        if box[0] == 4:
                            color = color1
                        elif box[0] == 1:
                            color = color3
                        elif box[0] == 2:
                            color = color2
                        elif box[0] == 3:
                            color = color0
                        elif box[0] == 0:
                            color = color4
                        elif box[0] == 5:
                            color = color6

                        # 绘制矩形，第2，3个参数是矩形框左上角坐标和右下角坐标，最后一给是线宽
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        # 矩形框周围内容区域参数设置：
                        label = label_map[class_id]
                        font_scale = 1.0  # 调整字体大小
                        font_color = color1
                        line_type = 2  # 线宽

                        # 这些是为了直播效果，后期可以注释掉避免影响训练 迭代模型。
                        # 显示 识别框标签
                        # cv2.putText(frame, label, (x2-10, y2+20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, line_type)
                        # # 显示 置信度文本
                        # confidence_text = f"Conf: {confidence_value:.2f}"
                        # cv2.putText(frame, confidence_text, (x2 - 10, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color,line_type)
                        # # 显示 帧数
                        # frame_text = f"Frames: {frame_count}"  # 这是你想要显示的额外内容
                        # cv2.putText(frame, frame_text, (40, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        #             font_color, line_type)
                        # print("has_behavior_frame_count is " + str(frame_count))
                        # 显示底部背景的视频帧画面
                        cv2.imshow("Fatigue_driving_Detection", frame)

                        # 取置信度大的保存和进行判断
                        if box[2] >= save_acc:
                            # ("into judge and save")
                            if box[0] == 4:
                                have_normal_in_frame = 1
                                # print("normal")
                            if box[0] == 1:
                                have_phone_in_frame = 1
                                have_normal_in_frame = 0
                                frame_count+=1
                            if box[0] == 0:
                                have_leftright_in_frame = 1
                                have_normal_in_frame = 0
                                frame_count += 1
                            if box[0] == 5:
                                have_headdown_in_frame = 1
                                have_normal_in_frame = 0
                                frame_count += 1
                            if box[0] == 2:
                                have_mouth_in_frame = 1
                                have_normal_in_frame = 0
                                frame_count += 1
                            if box[0] == 3:
                                have_eye_in_frame = 1
                                have_normal_in_frame = 0
                                frame_count += 1
                            cv2.imwrite(f"./frames/{user}_{frame_count}_{box[0]}.jpg", frame)

                            # 生成xml
                            # print("width", width)
                            # print(height)

                            # 创建根元素
                            annotation = ET.Element("annotation")

                            # 创建子元素并添加到根元素
                            folder = ET.SubElement(annotation, "folder")
                            folder.text = "NA"

                            filename = ET.SubElement(annotation, "filename")
                            # 纯文件名，不包含路径
                            filename.text = f"{user}_{frame_count}_{box[0]}.jpg"  # 根据元素的索引命名图片文件名

                            source = ET.SubElement(annotation, "source")
                            database = ET.SubElement(source, "database")
                            database.text = "Unknown"

                            size = ET.SubElement(annotation, "size")
                            width_element = ET.SubElement(size, "width")
                            width_element.text = str(self.width)
                            height_element = ET.SubElement(size, "height")
                            height_element.text = str(self.height)
                            depth = ET.SubElement(size, "depth")  # 通道数
                            depth.text = "3"

                            segmented = ET.SubElement(annotation, "segmented")  # 图像是否被分割
                            segmented.text = "0"

                            # 创建对象元素
                            object_element = ET.SubElement(annotation, "object")

                            # 创建 <name> 元素并设置类别标签
                            name = ET.SubElement(object_element, "name")
                            name.text = label

                            pose = ET.SubElement(object_element, "pose")  # 没有提供有关物体姿态的详细信息
                            pose.text = "Unspecified"

                            truncated = ET.SubElement(object_element, "truncated")
                            truncated.text = "0"

                            difficult = ET.SubElement(object_element, "difficult")
                            difficult.text = "0"

                            occluded = ET.SubElement(object_element, "occluded")
                            occluded.text = "0"

                            # 创建边界框元素
                            # print("pre bndbox")
                            bndbox = ET.SubElement(object_element, "bndbox")
                            # print("into bndbox")

                            # print(x1, y1, "左上角点坐标")

                            xmin = ET.SubElement(bndbox, "xmin")
                            # print("xxxxxxxxxxxxxxxxxxx")
                            xmin.text = str(x1)
                            # print("xmin", xmin.text)
                            ymin = ET.SubElement(bndbox, "ymin")
                            ymin.text = str(y1)
                            xmax = ET.SubElement(bndbox, "xmax")
                            xmax.text = str(x2)
                            ymax = ET.SubElement(bndbox, "ymax")
                            ymax.text = str(y2)

                            # 创建XML树并保存到文件
                            tree = ET.ElementTree(annotation)
                            xml_filename = f"./frames/{user}_{frame_count}_{box[0]}.xml"  # 根据图像路径命名XML文件  现在比较晚了，宿舍环境是黑暗状态，比较符合夜间行驶的复杂场景
                            print("Successfully save ", xml_filename)
                            tree.write(xml_filename, encoding="utf-8", xml_declaration=True)
                        print(f"Successfully save as {user}_{frame_count}_{box[0]}.jpg ")

                    key = cv2.waitKey(1) # 等待窗口刷新，否则卡住一帧不动
                    if key == ord('q'):
                        break


                    # 开始判断该帧的行为分类
                    #判断打哈欠
                    if (have_mouth_in_frame == 1):
                        behavior_of_frame = 2

                    #判断闭眼
                    if (have_eye_in_frame == 1):
                        behavior_of_frame = 1
                        if (have_mouth_in_frame == 1):
                            behavior_of_frame = 2

                    # 下面是判断左顾右盼和低头
                    # 既有左顾右盼也有低头
                    if ((have_leftright_in_frame == 1) and (have_headdown_in_frame == 1)):
                        # self.look_around_frame += 1
                        behavior_of_frame = 4

                    # 左顾右盼或者低头有1个
                    if ((have_leftright_in_frame == 1) or (have_headdown_in_frame == 1)):
                        #self.look_around_frame += 1
                        behavior_of_frame = 4

                    # 既有低头也有闭眼，则不计算闭眼
                    if ((have_headdown_in_frame == 1) and (have_eye_in_frame == 1)):
                        behavior_of_frame = 4

                    # 判断手机，放在最后，如果该帧有手机则忽略其他行为
                    if (have_phone_in_frame == 1):
                        behavior_of_frame = 3

                    # 3帧列表中先删1帧，再加1帧
                    del three_frames[0]
                    three_frames.append(behavior_of_frame)
                    # print(three_frames)
                    # print(frame_timer)
                    # 实时检测应该放入此模块
                    if (three_frames[1] == three_frames[2]) and three_frames[0] != 0:
                        print("预警状态开启")
                        stop = frame_timer  # 记录开始时间
                        # 计算持续时间并检查是否超过三秒，我这里×10，是因为一旦进入了检测处conf>0.95+的中间会涉及到一系列读写，会降低速度。
                        # 所以这个实时检测的不应该放入该程序
                        # 本程序更像是在一开始的时候进行升级。
                        duration_seconds = int(((stop - start) / self.fps)*5)
                        # print(duration_seconds,"持续时间")
                        if duration_seconds >= 3:
                            print(f"成功录入，请更换下一个疲劳行为")
                        print(start, stop)
                    else:
                        start = frame_timer
                        print("正在录入，请微调姿态")

                    # 实时检测应该放入此模块
                    # if (three_frames[1] == three_frames[2]) and three_frames[0] != 0:
                    #     print("预警状态开启")
                    #     stop = frame_timer  # 记录开始时间
                    #     print(start, stop, self.fps)
                    #     # 计算持续时间并检查是否超过三秒，我这里×10，是因为一旦进入了检测处conf>0.95+的中间会涉及到一系列读写，会降低速度。
                    #     # 所以这个实时检测的不应该放入该程序
                    #     # 本程序更像是在一开始的时候进行升级。
                    #     duration_seconds = int(((stop - start) / self.fps)*10)
                    #     # print(duration_seconds,"持续时间")
                    #     if duration_seconds >= 3:
                    #         print(f"危险！您已经持续疲劳/分神{duration_seconds}秒了")
                    #     print(start, stop)
                    # else:
                    #     start = frame_timer
                    #     print("正常状态")

                    self.failures = 0
                    # 找出置信度低于acc的帧，记录对应帧数，类型，和置信度，并红色输出作为难判帧
                    # 在独立窗口中显示难判帧信息
                    y_step = 20
                    if confidence_value >= showFrame_acc and have_normal_in_frame == 0:
                        frame = np.ones((600, 500, 3), dtype=np.uint8)  # 创建一个空白图像
                        self.hard_frames_info.append((frame_count, label_map[class_id], confidence_value))
                        for info in self.hard_frames_info:
                            hard_frame_text = f"Hard_Frame: {info[0]}  Class: {info[1]}  Conf: {info[2]:.2f}"
                            cv2.putText(frame, hard_frame_text, (0, y_step), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                            y_step += 20
                        cv2.imshow("HardFrames", frame)
                else:
                    # print("this frame is none, frame read over")
                    break
            except Exception as e:
                if e.__class__ == KeyboardInterrupt:
                    print("Quitting")
                    break
                traceback.print_exc()
                self.failures += 1
                # print("33333333333333333333333333333")
                if self.failures > 30:  # 失败超过30次就默认返回
                    # print("4444444444444444444444")
                    break
            del frame
        cv2.destroyAllWindows()
        # 对整个视频进行结果result写入
        final_time = time.time()
        duration = int(np.round((final_time - now) * 1000))
        # 传给obs
        s = S3()
        # s.list_bucket_content('driver01')  # 列出桶内对象
        # 定义文件夹路径
        folder_path = './frames'  # 请根据实际情况更改文件夹路径

        # 判断是否有桶，若无调用 create_bucket 方法创建桶
        # bucket_name = 'user001'
        # if s.create_bucket(bucket_name):
        #     print(f'创建新桶 {bucket_name}已成功创建。')
        # else:
        #     print(f'桶 {bucket_name} 已经存在。')

        # 列出文件夹中的所有文件
        files = os.listdir(folder_path)
        for file_name in files:
            local_file_path = f'./frames/{file_name}'  # 本地文件路径，需要根据实际情况进行修改
            remote_file_name = f'user001/{file_name}'  # 远程文件名，这里将文件上传到名为 'driver01' 的存储桶
            print(f"成功上传{file_name}")
            # 上传文件到云端
            s.upload_object('driver01', remote_file_name, local_file_path)
        print("end")
        # print(duration)
        print("end---------------------------------------------------------------------")

    def _postprocess(self, data):
        # os.remove(self.capture)
        return data

if __name__ == "__main__":
    fati = fatigue_driving_detection("serve", "./best0906.pt")
    data = {}
    fati._preprocess(data)
    fati._inference(data)
    fati._postprocess(data)
