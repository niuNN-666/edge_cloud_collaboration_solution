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
import torch
import os
from S3_api import S3 # 存储obs
import xml.etree.ElementTree as ET # 生成xml文件
import json
# coding=utf-8
import threading
import requests
from apig_sdk import signer
# from baseline import fatigue_driving_detection
# from baseline import test_speed
# from baselinevideo import fatigue_driving_detection


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


class fatigue_driving_detection():
    def __init__(self, model_name, model_path):
        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = model_path
        # 默认配置sdk
        config = HttpConfig.get_default_config()

        self.capture = './test/night_woman_002_3.mp4'
        # print(self.capture)

        # print("111111111111111111111111111")
        self.video_capture = cv2.VideoCapture(self.capture)
        # print(self.video_capture)
        #self.capture = './0822video/night_woman_002_3.mp4'
        # self.cap = 'test.mp4'
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
        self.imgsz = 160

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

        #self.tracker = Tracker(self.width, self.height, threshold=None, max_threads=4, max_faces=4,
        #                  discard_after=10, scan_every=3, silent=True, model_type=3,
        #                  model_dir=None, no_gaze=False, detection_threshold=0.6,
        #                  use_retinaface=0, max_feature_updates=900,
        #                  static_model=True, try_hard=False)

        # self.temp = NamedTemporaryFile(delete=True)  # 用来存储视频的临时文件

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
        capture_name =(str(self.capture)[:-6]).split("/")[-1]
        return capture_name
    
    # 推断函数    
    def _inference(self, data):       
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        # print(data)
        # result = {"result": {"category": 0, "duration": 6000}}
        # result = {"result": {"duration": 6000, "category": 1, "drowsy": [{"periods":[],"category":0}]}}

        print("into inference")
        
        # 读取视频的宽度、高度、帧率
        self.cap = cv2.VideoCapture(self.capture)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        look_around = 0
        using_phone = 0
        yawn_frame = 0
        eye_closed_frame = 0
        normal_frame = 0

        self.input_reader = InputReader(self.capture, 0, self.width, self.height, self.fps)
        source_name = self.input_reader.name

        #self.look_around_frame = 0
        #self.eyes_closed_frame = 0
        #self.mouth_open_frame = 0
        #self.use_phone_frame = 0
        
        # get_frames_per_second = 3  #每秒取多少帧处理
        get_frames_per_second = 10
        self.frame_3s = 3 * get_frames_per_second
        
        frame_count = 0        
                
        half_width = int(self.width/2)
        full_width = int(self.width)
        
        # HavePhone = 0
        # HaveNoPhone = 0
        
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

            cap = cv2.VideoCapture(self.capture)
            # 记录当前是第几帧
            frame_count = frame_count + 1
            # print("frame_count is "+ str(frame_count))
            # if 5s
            # 读取1帧
            ret, frame = self.input_reader.read()

             # 每 how_many_frames_get_one 帧，取1帧
            how_many_frames_get_one = round(self.fps / get_frames_per_second)  # 计算多少帧取1帧
            if(frame_count % how_many_frames_get_one != 1):           
                continue

            self.need_reinit = 0

            #开始对这一帧进行判断处理
            try:
                if frame is not None:

                    print("into frame")
                    # 剪裁主驾驶位
                    # frame = frame[:, 600:1920, :]
                    frame = frame[:, half_width:full_width, :]

                    bbox = detect(self.model, frame, self.stride, self.imgsz)

                    #从这里开始新的复赛判断程序，fusai
                    have_phone_in_frame = 0
                    have_mouth_in_frame = 0
                    have_eye_in_frame = 0
                    have_leftright_in_frame = 0
                    have_headdown_in_frame = 0

                    confidence_value = 0.00  # 初始化 confidence_value

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

                    # 判断该帧所有的标注框
                    for box in bbox:

                        class_id = box[0]
                        # center_x, center_y, width, height = box[1][0], box[1][1], box[1][2], box[1][3]
                        # 置信度
                        confidence_tensor = torch.tensor(box[2])  # box2返回tensor()，是pytorch张量表现形式，需要转换浮点
                        confidence_value = round(confidence_tensor.item(),3) # 我选择保留置信度到小数点后三位


                        if box[0] == 4:
                            have_normal_in_frame = 1
                            continue
                        if box[0] == 1:
                            have_phone_in_frame = 1
                            have_normal_in_frame = 0
                            continue
                        if box[0] == 0:
                            have_leftright_in_frame = 1
                            have_normal_in_frame = 0
                            continue
                        if box[0] == 5:
                            have_headdown_in_frame = 1
                            have_normal_in_frame = 0
                            continue
                        if box[0] == 2:
                            have_mouth_in_frame = 1
                            have_normal_in_frame = 0
                            continue
                        if box[0] == 3:
                            have_eye_in_frame = 1
                            have_normal_in_frame = 0
                            continue

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

                    # 尝试找出置信度高于85%的帧，筛选除了正常的图像，生成xml

                    if confidence_value >= 0.85 and have_normal_in_frame == 0:
                        # frame = np.ones((600, 500, 3), dtype=np.uint8)  # 创建一个空白图像
                        print(box)
                        self.hard_frames_info.append((frame_count, label_map[class_id], confidence_value, box[1], self.width,self.height))
                        # for info in self.hard_frames_info:
                        #     hard_frame_text = f"Hard_Frame: {info[0]}  Class: {info[1]}  Conf: {info[2]:.2f}"
                        #     cv2.putText(frame, hard_frame_text, (0, y_step), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        #     y_step += 20

                        # cv2.imshow("HardFrames", frame)

                    # 该帧的行为类型判断结束，开始计算该行为持续时间
                    # print("behavior of frame " + str(frame_count) +" is " + str(behavior_of_frame))

                    # 3帧列表中先删1帧，再加1帧
                    del three_frames[0]
                    three_frames.append(behavior_of_frame)
                    # print(three_frames)

                    # 判断3帧的内容，是否有了行为改变。第2,3帧一样且和第1帧不同
                    if((three_frames[1] == three_frames[2]) and (three_frames[1] != three_frames[0])):
                        # print("behavior changed!")
                        stop = frame_count
                        # 判断如果持续时间大于3秒，且不是normal行为，写入result
                        if( ((stop - start)/self.fps >= 3 ) and (behavior_now_counting != 0) ):
                            self.behavior_info.append({
                                'category': behavior_now_counting,
                                'periods': [int(start/self.fps*1000), int(stop/self.fps*1000)],
                            })

                        # 修改计数的行为类型，以及起始帧和结束帧
                        behavior_now_counting = three_frames[2]
                        start = frame_count
                        stop = frame_count
                    else:
                        stop = frame_count

                    self.failures = 0

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
                if self.failures > 30:   # 失败超过30次就默认返回
                    # print("4444444444444444444444")
                    break
            del frame

        #对整个视频进行结果result写入
        final_time = time.time()
        duration = int(np.round((final_time - now) * 1000))

        # self.total_duration = duration
        result = {"result": {"duration": 6000, "drowsy": [{"periods":[],"category":0}]}}
        
        drowsy_list = []
        for behavior in self.behavior_info:
            periods_list = behavior['periods']
            category = behavior['category']
            drowsy_list.append({
                'periods': periods_list,
                'category': category,
            })
        
        result = {
            'result': {
                'duration': duration,
                'drowsy': drowsy_list,
            }           
        }
               
        print(result)
        print("end log---------------------------------------------------------------------")        
        return result

    # 返回 图片和xml 到OBS
    def get_hard_frames_info(self):
        output_folder = 'frames'  # 保存图像的文件夹路径
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("self.hard_frames_info",self.hard_frames_info)
        frame_number_list = []  # 调用下一个视频后来清空列表
        frame_number_list = [info[0] for info in self.hard_frames_info]
        frame_category_list = [info[1] for info in self.hard_frames_info]
        # print("frem_category_wlist",frame_category_list)  # frem_category_wlist ['eye_closed', 'open_mouth',
        # relational_conf = self.hard_frames_info[2]  # 创建置信度变
        capture_name = fati._preprocess(data)  # ./test/night_woman_002_3.mp4_frame_567.jpg
        output_image_paths = []  # 存储生成的图像文件路径
        for i, category in zip(frame_number_list, frame_category_list):  # 这个zip列表很重要。生成了双属性的可迭代对象，如果用两层for会造成大量重复
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)  # 捕获当前帧位置
            ret, frame = self.video_capture.read()  # cap.read()读取该帧
            if ret:
                output_image_path = os.path.join(output_folder, f'{capture_name}_{category}_{i}.jpg')
                # print("output_image_path",output_image_path)
                cv2.imwrite(output_image_path, frame)
                output_image_paths.append(output_image_path)
                # print(output_image_paths,"list==============")
                # print(f"Frame {i} saved as {output_image_path}")
            else:
                print(f"Unable to read frame {i}")

        # 生成xml
        # 循环遍历 self.hard_frames_info 中的每个元素
        # for i, item in enumerate(self.hard_frames_info):
        for paths, item in zip(output_image_paths, self.hard_frames_info):
            # 解包元素中的信息
            index, class_name, confidence, box, width, height = item
            print("width",width)
            print(height)

            # 创建根元素
            annotation = ET.Element("annotation")

            # 创建子元素并添加到根元素
            folder = ET.SubElement(annotation, "folder")
            folder.text = "NA"

            filename = ET.SubElement(annotation, "filename")
            # 纯文件名，不包含路径
            paths_file = paths[7:]
            print(paths_file)
            filename.text = f"{paths_file}"  # 根据元素的索引命名图片文件名

            source = ET.SubElement(annotation, "source")
            database = ET.SubElement(source, "database")
            database.text = "Unknown"

            size = ET.SubElement(annotation, "size")
            width_element = ET.SubElement(size, "width")
            width_element.text = str(width)
            height_element = ET.SubElement(size, "height")
            height_element.text = str(height)
            depth = ET.SubElement(size, "depth")  # 通道数
            depth.text = "3"

            segmented = ET.SubElement(annotation, "segmented")  # 图像是否被分割
            segmented.text = "0"

            # 创建对象元素
            object_element = ET.SubElement(annotation, "object")

            # 创建 <name> 元素并设置类别标签
            name = ET.SubElement(object_element, "name")
            name.text = class_name

            pose = ET.SubElement(object_element, "pose")  # 没有提供有关物体姿态的详细信息
            pose.text = "Unspecified"

            truncated = ET.SubElement(object_element, "truncated")
            truncated.text = "0"

            difficult = ET.SubElement(object_element, "difficult")
            difficult.text = "0"

            occluded = ET.SubElement(object_element, "occluded")
            occluded.text = "0"

            # 创建边界框元素
            print("pre bndbox")
            bndbox = ET.SubElement(object_element, "bndbox")
            print("into bndbox")
            # detect 返回的xy是物体的中心点坐标,wh,是物体的长宽,图像的左上角为原点，向下为h轴正半轴,向右为w正半轴
            center_x, center_y, width2box, height2box = box[0], box[1], box[2], box[3]
            # 所以检测框的左上坐标点为：(x- w/2 , y-h/2)
            # 等比例缩放，所以乘以图像的原尺寸便是坐标点,detect检测是半张但传入的宽是原始宽，所以不可以*2
            x = int(center_x * width/2 + width/2)
            y = int(center_y * height)
            w = int(width2box * width/2)
            h = int(height2box * height)

            # # 方案2 全景
            # x = int(center_x * width)
            # y = int(center_y * height)
            # w = int(width2box * width)
            # h = int(height2box * height)

            x1 = int(x - w/2 )
            y1 = int(y - h/2 )
            x2 = x1 + w
            y2 = y1 + h
            print(x1,y1,"左上角点坐标")

            xmin = ET.SubElement(bndbox, "xmin")
            print("xxxxxxxxxxxxxxxxxxx")
            xmin.text = str(x1)
            print("xmin",xmin.text)
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(y1)
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(x2)
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(y2)

            paths_x = paths_file[:-4]
            # folder_path = './frames'
            print(paths_x)
            # 创建XML树并保存到文件
            tree = ET.ElementTree(annotation)
            xml_filename = f"./frames/{paths_x}.xml"  # 根据图像路径命名XML文件
            tree.write(xml_filename, encoding="utf-8", xml_declaration=True)

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

        return output_image_paths

    def _postprocess(self, data):                        
        # os.remove(self.capture)
        return data


# 类外构建
# def run_fatigue_driving_detection():
if __name__ == "__main__":
    fati = fatigue_driving_detection("serve", "./best.pt")
    data = {}
    fati._preprocess(data)
    fati._inference(data)
    # 不可以把存储放在推断里，这样会极大的降低速度
    fati.get_hard_frames_info()

    fati._postprocess(data)
