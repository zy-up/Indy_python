# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Hyeun Jeong Min 2020

import atexit
import bisect
import multiprocessing as mp
import cv2
import torch
import pandas as pd
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
import numpy as np
import os
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

## communication message
MSG_ROBOT_READY = 100
MSG_ROBOT_NOT_READY = 101
MSG_TRIGGER = 102
MSG_IMG_DATA = 103
MSG_NO_DATA = 104
MSG_PROGRAM_END = 105
MSG_TRIGGER_SHAPE=106

# 인식 세팅값
VAR_LAYER_CNT = 101
VAR_PER_BATCH = 12

# 작업 경로 
WORK_DIR = './'
VAR_OUTPUT_DIR = WORK_DIR + 'output' # 모델이 저장된 경로
VAR_IMAGE_DIR = WORK_DIR + 'train' # 테스트 이미지 경로
VAR_RES_DIR = WORK_DIR + 'result' # 실행 결과를 저장할 경로


f_path = VAR_IMAGE_DIR
m_path = VAR_OUTPUT_DIR
r_path = VAR_RES_DIR


class_name = ['Planted_spot']
VAR_NUM_CLASSES = len(class_name)

metadata = MetadataCatalog.get("mdata_").set(thing_classes=class_name)
def start():
    global VS
    VS = Visualization(m_path)  #  
class Visualization(object):
    def __init__(self, path, instance_mode=ColorMode.IMAGE, parallel=False):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

        cfg.SOLVER.IMS_PER_BATCH = VAR_PER_BATCH
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
        
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = VAR_NUM_CLASSES

        cfg.MODEL.WEIGHTS = os.path.join(path, "model_final.pth")
        
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set a custom testing threshold
        cfg.freeze()
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)
            self.temp_var = []


    def run_on_image(self, img, metadata_): # 컬러 이미지에 예측 실행
        vis_output = None
        predictions = self.predictor(img)
        image = img[:, :, ::-1]
        v = Visualizer(image, metadata_, scale=1.2)
        if "instances" in predictions:
            self.instances = predictions["instances"].to("cpu")
            self.v_output = v.draw_instance_predictions(predictions=predictions["instances"].to("cpu"))
            self.num_instances = len(self.instances)

        if self.num_instances > 0:
            print('detected: ', self.num_instances)
            self.pp = image.shape
            
            return self.v_output

        else:
            print('detected: nothing')
            return None
    def ins_to_pd(self): # 인식 결과를 판다스 데이터 프레임으로 변환
        tem_pos = []
        self.vv=[0]*20
        boxes= self.instances.pred_boxes.tensor.numpy()
        classes = self.instances.pred_classes
        scores = self.instances.scores
        for i in range(self.num_instances):
            score = round(scores.numpy()[i]* 100)
            class_n = classes.numpy()[i]
            box=boxes[i]
            tem_pos.append({'Class': class_n, # 클래스
                            'X': (box[2]+box[0])/2, # x 좌표
                            'Y': (box[3]+box[1])/2, # y 좌표
                            # 'Theta': theta,
                            # 'Direct': direct,
                            'Box': box, # 경계 상자
                            'Score' : score} # 인식 정확도
                            )
        tem_pos = sorted(tem_pos, key=lambda tem_pos: tem_pos['Class']) # default score => 클래스 순으로 정렬
        xy_pos = pd.DataFrame(tem_pos)
        
        return xy_pos
def send_data(count,img,img_depth):
    if img == 'end': # 종료 명령 수신시 프로그램 종료
      print(img)
      return 0
    # img = cv2.imread('/content/drive/MyDrive/ML/auto_label_dev/test/snapshot_20220531_202.jpg') # 테스트 이미지
    print("spot searching")
    save_path=r_path
    v_output = VS.run_on_image(img,metadata) # 예측 실행
    
    
    if v_output != None: # 인식 결과 있는 경우
        instances= VS.ins_to_pd() # 인식 결과를 데이터프레임으로 변환
        for i in range (VS.num_instances):
            box=instances['Box'][i]
            X=int((box[2]+box[0])/2)
            Y=int((box[3]+box[1])/2)
            cv2.circle(img, (X,Y), 10, (0,0,255), thickness=3 )
            
        
        # 오리지날 이미지 저장
        fname = 'orgimg_' + str(count) + '.jpg'
        out_1 = os.path.join(save_path, fname)
        cv2.imwrite(out_1, img)
        
        # depth 이미지 저장
        fname = 'orgdepth_' + str(count) + '.jpg'
        out_2 = os.path.join(save_path, fname)
        cv2.imwrite(out_2, img_depth)

        # 인식 결과 이미지 저장
        fname = 'img_' + str(count) + '.jpg'
        out_filename = os.path.join(save_path, fname)
        v_output.save(out_filename)

        return instances # 예측 결과 송신
    else: # 인식 결과가 없는 경우
        s_data = 'nothing'
        
        return s_data
    

class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5

