from detectron2.utils.logger import setup_logger
setup_logger()
import pyrealsense2 as rs
# import some common libraries
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment',  None)
pd.options.display.max_rows = 1000
import time
# import some common detectron2 utilities
from detectron2.data.detection_utils import read_image
from visualize import send_data, start


MSG_ROBOT_READY = 100
MSG_ROBOT_NOT_READY =101
MSG_TRIGGER = 102
MSG_IMG_DATA = 103
MSG_NO_DATA = 104
MSG_PROGRAM_END = 105
MSG_EMERGENCY=106


### Robot info
robot_ip = "192.168.0.226"
robot_name = "NRMK-Indy7"



color_intrin = None
depth_intrin = None
depth_to_color_extrin = None


class Realsense():
    def __init__(self): # Realsense 실행
        self.pipeline = rs.pipeline()
        config = rs.config()
        dev = rs.device()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
    def snapshot(self): # 촬영
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            self.aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not self.aligned_depth_frame or not color_frame: continue
            
            self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            depth_intrin = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            self.depth_to_color_extrin = self.aligned_depth_frame.profile.get_extrinsics_to(color_frame.profile)
            depth_frame = self.aligned_depth_frame.get_data()

            color_img = np.asanyarray(color_frame.get_data()) # 컬러 이미지
            depth_img = np.asanyarray(depth_frame) # depth 이미지
            
            return color_img, depth_img
        
    def pixel_to_point(self, x, y): # x, y pixel 값을 3D x, y 좌표값으로 변환 및 거리 추출
        coordinate=[[0,0,0]]*3
        while True:
            for i in range (0,3):
                color_img, depth_img=RS.snapshot()
                dist = self.aligned_depth_frame.get_distance(x, y)
                # print(f"\r Distance at the selected pixel x:{x},y:{y} is: {dist} mtr")
                depth_point = rs.rs2_deproject_pixel_to_point(self.color_intrin,[x,y],dist)
                coordinate[i] = rs.rs2_transform_point_to_point(self.depth_to_color_extrin,depth_point)
            std=np.std(coordinate,axis=0,dtype=np.float32)
            mean=np.mean(coordinate,axis=0,dtype=np.float32).tolist()
            if mean[2]<0.01:
                continue
            elif std.all()<0.003:
                break
        # [x3d,y3d,z3d] = [x3d*1000,y3d*1000,z3d*1000] # m -> mm 단위로 변환
        # print(f"\r{x3d}, {y3d}, {z3d}")
        return mean
        
    def close(self): # camera 및 화면 종료
        self.pipeline.stop()
        # cv2.destroyAllWindows()



def test_main(p_conn):
    start()
    global RS
    RS = Realsense()
    print('realsense is running')
    with open('log.txt', "a") as f:
        f.write('PROGRAM_START' + ' ' + str(time.time()) + '\n')
    count = 0
    shape_count=0
    while True:
        
        # cv2.imshow('camera', c_img) # 이미지 화면 띄우기
        data = p_conn.recv() # indy_task로부터 트리거 수신
        
        print('data received', data)
        if data == MSG_PROGRAM_END :   #or (cv2.waitKey(1)&0xFF == ord('q')): indy_task에서 종료 명령이 수신된 경우
            s_data1 = 'end'
            s_data2 = 'program'
                 # indy_task 종료
            send_data(count, s_data1, s_data2) # colab client 종료
            with open('log.txt', "a") as f:
                f.write('MSG_PROGRAM_END' + ' ' + str(time.time()) + '\n')
            break
        
        elif data == MSG_TRIGGER: # indy_task에서 촬영 트리거가 수신된 경우
            c_img, d_img = RS.snapshot() # 촬영 및 이미지 저장
            instances = send_data(count, c_img, d_img)# 인식 결과
            
            if type(instances) == str: # 인식 결과가 없는 경우
                print('Not detected planted_spot')
                p_conn.send(MSG_NO_DATA)
                p_conn.send(0)
                 # indy_task에 인식 실패 송신
                with open('log.txt', "a") as f:
                    f.write('MSG_NO_DATA' + ' ' + str(time.time()) + '\n')
            else: # 인식 결과가 있는 경우
                print('detect')
                coord_3D = list()
                for i in range(len(instances)):  # relative 3D coordinate from pyrealsense of each instances
                    pixel = [int(instances['X'][i]), int(instances['Y'][i])]
                    coord_3D.append(RS.pixel_to_point(pixel[0], pixel[1]))
                instances['3D'] = coord_3D
                    
                # print('cycle: ', R_count, '\n', instances)
                p_conn.send(MSG_IMG_DATA)
                p_conn.send(instances) # indy_task에 인식 결과 송신
                with open('log.txt', "a") as f:
                    f.write('MSG_IMG_DATA' + ' '+str(instances)+ ''+ str(time.time()) + '\n')
                count += 1
    RS.close()
    p_conn.close()                 
    
from multiprocessing import Pipe, Process
from task1 import call_task1
from task2 import call_task2

def task1(c_conn,m_out):
    call_task1(c_conn,m_out)
def task2(m_in):
    call_task2(m_in)
   
if __name__ == '__main__':
    p_conn, c_conn = Pipe()
    m_in,m_out=Pipe()
    p1 = Process(target=test_main, args=((p_conn),))
    p2 = Process(target=task1, args=((c_conn),(m_out)))
    p3 = Process(target=task2, args=((m_in),))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()

