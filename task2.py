import time
from indy_utils import indydcp_client as client 
import numpy as np
import math
from multiprocessing import Pipe, Process

MSG_ROBOT_READY = 100
MSG_ROBOT_NOT_READY = 101
MSG_TRIGGER = 102
MSG_IMG_DATA = 103
MSG_NO_DATA = 104
MSG_PROGRAM_END = 105
MSG_TRIGGER_SHAPE=106
SIDE_LENGTH=(0.495-0.293)/4
SIDE_LENGTH1=(0.201-0.193)/4
## Robot info
robot_ip1 = "192.168.0.226"
robot_name1 = "NRMK-Indy7"
robot_ip2 = "192.168.0.113"
robot_name2 = "NRMK-Indy7"
indy2 = client.IndyDCPClient(robot_ip2, robot_name2)# indy 객체 생성

def up(indy):
    indy.set_do(8,1)
    indy.set_do(9,0)
    
def down(indy):
    indy.set_do(8,0)
    indy.set_do(9,1)
    
def grap(indy):
    indy.set_do(1,1)
    indy.set_do(2,0)
    
def reverse_grap(indy):
    indy.set_do(1,0)
    indy.set_do(2,1) 
    
def release(indy):
    indy.set_do(1,1)
    indy.set_do(2,1)
    
def call_task2(m_in):
    time.sleep(1)
    res=m_in.recv()
    if res!=MSG_ROBOT_READY:
        print('Error')
    else:
        m_in.send(MSG_ROBOT_READY)
        print('Robot2 task started..')
        # indy connect
        indy2.connect()
        # indy current status
        status = indy2.get_robot_status()
        
        if not (status['ready'] == True and status['home'] == True): # Indy가 홈위치에서 준비 상태인지 확인        
            # 홈위치에서 준비 상태가 아닌 경우 홈위치로
            indy2.go_home()
            indy2.wait_for_move_finish()
            
        # 그리퍼 초기화(close)
        release(indy2)
        # down(indy2)
        print('robot is ready')
        while True:
            for i in range(1):
                res=m_in.recv()
                if res==MSG_ROBOT_READY:
                    print('Robot2 task is started')
                    # up(indy2)
                    print('pick plant:',i+1)
                    indy2.task_move_to((0.1930212364814092+SIDE_LENGTH1*i, 0.2930182365241853+SIDE_LENGTH*i, 0.625, -179.994269626575, 0.005494467944409564, 67.00091788805518))
                    indy2.wait_for_move_finish()
                    indy2.task_move_to((0.19300614797530952+SIDE_LENGTH1*i, 0.29297563623456424+SIDE_LENGTH*i, 0.44991828240639403, -179.9983292872579, 0.012714609479980718, 67.00188109594978))
                    indy2.wait_for_move_finish()
                    reverse_grap(indy2)
                    indy2.task_move_to((0.19299834593268658+SIDE_LENGTH1*i, 0.2929735271487578+SIDE_LENGTH*i, 0.42797404495670605, 179.99571239749199, 0.005805834519455521, 67.00236524129231))
                    indy2.wait_for_move_finish()
                    grap(indy2)
                    time.sleep(0.5)
                    indy2.task_move_to((0.19300614797530952+SIDE_LENGTH1*i, 0.29297563623456424+SIDE_LENGTH*i, 0.44991828240639403, -179.9983292872579, 0.012714609479980718, 67.00188109594978))
                    indy2.wait_for_move_finish()
                    indy2.task_move_to((0.1930212364814092+SIDE_LENGTH1*i, 0.2930182365241853+SIDE_LENGTH*i, 0.625, -179.994269626575, 0.005494467944409564, 67.00091788805518))
                    indy2.wait_for_move_finish()
                    indy2.task_move_to((0.1930212364814092, 0.2930182365241853, 0.625, -179.994269626575, 0.005494467944409564, 67.00091788805518))
                    indy2.wait_for_move_finish()
                    # down(indy2)
                    print('Robot2 task is finished')
                    res=m_in.recv()
                    if res==MSG_ROBOT_READY:
                        print('Robot2 task is started')
                        indy2.joint_move_to((-244.45438101273567, 11.004714335292565, 52.52881262913569, 0.31349106590348597, 117.66319350441678, -38.86391819113808))
                        indy2.wait_for_move_finish()
                        release(indy2)
                        time.sleep(0.5)
                        indy2.joint_move_to((-155.4845081676239, -0.527389148050138, 59.96287952770282, -0.004840510906559725, 120.5720686204442, -42.484076471614806))
                        indy2.wait_for_move_finish()
                        m_in.send(MSG_ROBOT_READY)
                        print('Robot2 task is finished')
                elif res==MSG_PROGRAM_END :  
                     indy2.disconnect()
                     m_in.close()   
                    
    
    
