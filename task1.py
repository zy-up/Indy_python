import time
from indy_utils import indydcp_client as client 
import numpy as np
import math
# communication
MSG_ROBOT_READY = 100
MSG_ROBOT_NOT_READY = 101
MSG_TRIGGER = 102
MSG_IMG_DATA = 103
MSG_NO_DATA = 104
MSG_PROGRAM_END = 105
MSG_EMERGENCY=106
# constant
SIDE_LENGTH=0.2025
DIGGING_NUM=3 
DIGGING_DEPTH=0.048
## Robot info
robot_ip1 = "192.168.0.226"
robot_name1 = "NRMK-Indy7"

X_matrix=np.mat([[-1.01125262],
        [ 0.11810224],
        [ 0.04560716],
        [ 0.02939205]])
Y_matrix=np.mat([[ 0.0148139 ],
        [ 1.00947993],
        [-0.04043532],
        [-0.07486916]])
Z_matrix=np.mat([[ 0.02680089],
        [-0.00318657],
        [-0.9946918 ],
        [ 0.1678744 ]])

def close(indy):
    indy.set_do(8,1)
    indy.set_do(9,0)
def open(indy):
    indy.set_do(8,0)
    indy.set_do(9,1)

def digging(indy):
    close(indy)
    for i in range(DIGGING_NUM):
        indy.task_move_by((0,0,-DIGGING_DEPTH/DIGGING_NUM*(i+1),0,0,0)) 
        indy.wait_for_move_finish()
        open(indy)
        time.sleep(0.5)
        indy.task_move_by((0,0,DIGGING_DEPTH/DIGGING_NUM*(i+1),0,0,0)) 
        indy.wait_for_move_finish()
        close(indy)
    
def planting(indy):
    indy.task_move_by((0,0,-0.01,0,0,0))
    indy.wait_for_move_finish()
    open(indy)
    time.sleep(0.5)
    indy.task_move_by((0,0,0.01,0,0,0))
    indy.wait_for_move_finish()        
    
def waiting_for_plant(indy):
    indy.task_move_to((0.0928670107987948, -0.266142179150747, 0.38541635373746386, 1.126176387251928, -179.6333631003841, 69.9844615820403))
    indy.wait_for_move_finish()
    
def find_spot(indy,data):
    #카메라 중앙에서 가장 가까운 객체 선택
    dist=[400]*10
    for i in range(0,len(data.index)):
            dist[i]=math.sqrt(pow((data['X'][i]-320),2)+pow((data['Y'][i]-240),2))
    min_index=dist.index(min(dist))
    TD_xyz=data.loc[min_index,'3D']
    snap_point=indy.get_task_pos()
    
    move_to=calibration(TD_xyz)
    indy.task_move_by(move_to)
    indy.wait_for_move_finish()
    digging (indy)
    indy.task_move_by((0,0,0.1,0,0,0))
    indy.wait_for_move_finish()
    return snap_point,TD_xyz

def plant(indy,TD_xyz):
    move_to=calibration(TD_xyz)
    indy.task_move_by(move_to)
    indy.wait_for_move_finish()
    planting (indy)
    indy.task_move_by((0,0,0.1,0,0,0))
    indy.wait_for_move_finish()

def calibration(TD_xyz):
    TD_xyz.append(1)
    TD_xyz=np.asmatrix(TD_xyz)
    real_dist_x=TD_xyz*X_matrix
    real_dist_y=TD_xyz*Y_matrix
    real_dist_z=TD_xyz*Z_matrix
    move_to=(real_dist_x,real_dist_y,real_dist_z,0,0,0)
    return move_to
    
def call_task1(c_conn,m_out):
    time.sleep(1)
    m_out.send(MSG_ROBOT_READY)  #task1 과 task2 통신 확인
    res=m_out.recv()
    if res!= MSG_ROBOT_READY:
        print('Error..')
    else:
        # indy 객체 생성
        indy1 = client.IndyDCPClient(robot_ip1, robot_name1)
        # indy connect
        indy1.connect()
        print('Robot1 task started..')
        # indy 속도 설정
        indy1.set_task_vel_level(3)
        # indy current status
        status = indy1.get_robot_status()

        if not (status['ready'] == True and status['home'] == True): # Indy가 홈위치에서 준비 상태인지 확인
            
            # 홈위치에서 준비 상태가 아닌 경우 홈위치로
            indy1.go_home()
            indy1.wait_for_move_finish()   
        # 그리퍼 초기화(close)
        close(indy1)   
        print('robot is ready')
        while True:
            print('Robot1 center_detection is operated..')
            #center point 
            c_conn.send(MSG_TRIGGER) # 촬영 트리거 송신
            res= c_conn.recv() # 인식 결과 수신
            data=c_conn.recv()
            print(data)
            if res==MSG_NO_DATA:
                continue
            # 카메라 중앙에서 가장 가까운 개체 선택
            dist=[400]*10
            for i in range(0,len(data.index)):
                dist[i]=math.sqrt(pow((data['X'][i]-320),2)+pow((data['Y'][i]-240),2))
            min_index=dist.index(min(dist))
            center_point_x=data['3D'][min_index][0]
            a=np.array(indy1.get_task_pos())
            point=[[0]]*5
            print('Robot1 center_detection is finished..')
            for i in range(5):
                point[i]=(a+np.array([SIDE_LENGTH*(i-2)-center_point_x,0,0,0,0,0])).tolist()
            print('each point of planting spot is defined')
            
            snap_point=[0]*5
            for i in range(1):
                print('move to spot:',i+1)
                indy1.task_move_to(point[i]) #촬영위치로 이동
                indy1.wait_for_move_finish()
                c_conn.send(MSG_TRIGGER) # 촬영 트리거 송신
                 # 인식 결과 수신
                res= c_conn.recv()
                data=c_conn.recv()
                print(data)

                if res == MSG_PROGRAM_END: # 종료
                    break
            
                elif res == MSG_NO_DATA: # 인식 결과 없음
                    continue
                                    
                else: # 인식 결과 있음
                    m_out.send(MSG_ROBOT_READY) #task 2에 동작 트리거 송신

                    snap_point[i],coordinate=find_spot(indy1,data)
                    waiting_for_plant(indy1)
                    print('Robot 1 task is finished')
                    m_out.send(MSG_ROBOT_READY)
                    res=m_out.recv()
                    if res==MSG_ROBOT_READY:
                        # 모종 심기
                        print('Robot 1 task is started')
                        indy1.task_move_to(snap_point[i])
                        indy1.wait_for_move_finish()
                        plant(indy1,coordinate)
                        indy1.task_move_to(snap_point[i])
                        indy1.wait_for_move_finish()
                        close(indy1)
            indy1.go_home()
            indy1.wait_for_move_finish()
            
            break
            
            
            # if(data == PROCESS_QUIT):
            #     print('process: call_task1 ends....')
            #     break
            # if(data[0] == str(MSG_IMG_DATA)):
            #     print('Robot has received the message: ', data)
            ## data의 2번째부터 이전 데이터 (센터위치, width, height, degree, UP/DOWN/LEFT/RIGHT)


            ## 여기에 원하는 코드 삽입 가능
            #time.sleep(5)
    print('finish')
    m_out.send(MSG_PROGRAM_END)
    c_conn.send(MSG_PROGRAM_END)     
    # indy1.go_home()
    # indy1.wait_for_move_finish()
    indy1.disconnect()
    m_out.close() 
    c_conn.close()
     

