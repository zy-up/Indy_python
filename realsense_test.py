import numpy as np
import pyrealsense2 as rs
from visualize import send_data, start
import math
import cv2
from indy_utils import indydcp_client as client 
robot_ip1 = "192.168.0.226"
robot_name1 = "NRMK-Indy7"
indy1 = client.IndyDCPClient(robot_ip1, robot_name1)
indy1.connect()
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
            if std.all()<0.003:
                break
        # [x3d,y3d,z3d] = [x3d*1000,y3d*1000,z3d*1000] # m -> mm 단위로 변환
        # print(f"\r{x3d}, {y3d}, {z3d}")
        mean=np.mean(coordinate,axis=0,dtype=np.float32).tolist()
        return mean
        
    def close(self): # camera 및 화면 종료
        self.pipeline.stop()
        # cv2.destroyAllWindows()
RS=Realsense() 
count=0     
start()
print(2222)
TD_xyz=[[0]]*4
relative=[[0]]*4
while True:
    c_img, d_img = RS.snapshot() # 촬영 및 이미지 저장
    cv2.imshow('camera', c_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        robot_x1y1z1=np.array(indy1.get_task_pos()[:3])
        break
print('origin set')
for i in range (4):
    
    
    while True:
        c_img, d_img = RS.snapshot() # 촬영 및 이미지 저장
        cv2.imshow('camera', c_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            instances = send_data(count, c_img, d_img)# 인식 결과
            if type(instances) == str: # 인식 결과가 없는 경우
                print('Not detected card')
            else: # 인식 결과가 있는 경우
                print('detect')
                coord_3D = list()
                for j in range(len(instances)):  # relative 3D coordinate from pyrealsense of each instances
                    pixel = [int(instances['X'][j]), int(instances['Y'][j])]
                    coord_3D.append(RS.pixel_to_point(pixel[0], pixel[1]))
                instances['3D'] = coord_3D 
                dist=[400]*10
                for j in range(0,len(instances.index)):
                        dist[j]=math.sqrt(pow((instances['X'][j]-320),2)+pow((instances['Y'][j]-240),2))
                        min_index=dist.index(min(dist))
                TD_xyz[i]=instances.loc[min_index,'3D'] #object choosen
                robot_xyz=np.array(indy1.get_task_pos()[:3])
                relative[i]=(robot_x1y1z1-robot_xyz).tolist()
                print('end',i)
                break
            break 
print(TD_xyz)         
print(relative)       
X=np.mat([[relative[0][0]],[relative[1][0]],[relative[2][0]],[relative[3][0]]])
Y=np.mat([[relative[0][1]],[relative[1][1]],[relative[2][1]],[relative[3][1]]])
Z=np.mat([[relative[0][2]],[relative[1][2]],[relative[2][2]],[relative[3][2]]])

prime=np.mat([[TD_xyz[0][0],TD_xyz[0][1],TD_xyz[0][2],1],
              [TD_xyz[1][0],TD_xyz[1][1],TD_xyz[1][2],1],
              [TD_xyz[2][0],TD_xyz[2][1],TD_xyz[2][2],1],
              [TD_xyz[3][0],TD_xyz[3][1],TD_xyz[3][2],1]])
print(prime.I)
dx_matrix=prime.I*X
dy_matrix=prime.I*Y
dz_matrix=prime.I*Z

print('X_matrix',dx_matrix)
print('Y_matrix',dy_matrix)
print('Z_matrix',dz_matrix)
while True:
    c_img, d_img = RS.snapshot()
    cv2.imshow('camera', c_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        instances = send_data(count, c_img, d_img)# 인식 결과
        if type(instances) == str: # 인식 결과가 없는 경우
            print('Not detected card')
        else: # 인식 결과가 있는 경우
            print('detect')
            coord_3D = list()
            for i in range(len(instances)):  # relative 3D coordinate from pyrealsense of each instances
                pixel = [int(instances['X'][i]), int(instances['Y'][i])]
                coord_3D.append(RS.pixel_to_point(pixel[0], pixel[1]))
            instances['3D'] = coord_3D
            dist=[400]*10
            for i in range(0,len(instances.index)):
                    dist[i]=math.sqrt(pow((instances['X'][i]-320),2)+pow((instances['Y'][i]-240),2))
                    min_index=dist.index(min(dist))
            TD_xyz=instances.loc[min_index,'3D'] #object choosen  
            break
TD_xyz.append(1)
TD_xyz=np.asmatrix(TD_xyz)
real_dist_x=TD_xyz*dx_matrix
real_dist_y=TD_xyz*dy_matrix
real_dist_z=TD_xyz*dz_matrix
print(real_dist_x)
print(real_dist_y)
print(real_dist_z)
RS.close()
indy1.disconnect