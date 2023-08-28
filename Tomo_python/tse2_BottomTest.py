from  tomoNV_Cpp import *
import os

g_input_mesh_filename = '..\\Tomo_MeshData\\(5)Bunny_69k.stl'
theta_YP = 30 #should be integer
mesh0 = o3d.geometry.TriangleMesh()
mesh0 = o3d.io.read_triangle_mesh(g_input_mesh_filename)

import pandas as pd
excel1 = pd.read_excel('D:\\OneDrive\\Documents\\__PaperWorks\\__InProgress\\_섬공지_tomo4편(2023)\\TSE2_바닥구조(2022연구년)\\_연구생\\TSE2_30도테스트_최종취합_0623.xlsx',sheet_name='30',engine='openpyxl', header=None,dtype={'Column C':float})[2:171]
(NoBed,Skirt,Brim,Raft) = np.transpose( excel1.to_numpy() [0:172, 3:7]).astype(np.float32)
#print(NoBed)

nYPR_Intervals= int(360 / theta_YP) +1
yaw_range   = np.linspace(toRadian(0), toRadian(360), num=nYPR_Intervals, endpoint=True, dtype=np.float32)
pitch_range = np.linspace(toRadian(0), toRadian(360), num=nYPR_Intervals, endpoint=True, dtype=np.float32)
roll_range  = np.zeros(1) #for generality. roll direction  is not needed.

(xx,yy,zz)   = np.meshgrid( yaw_range, pitch_range, roll_range)
YPR = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32) #https://rfriend.tistory.com/352

Mtotal = Raft

(optimal_YPRs,worst_YPRs) = findOptimals(YPR, Mtotal, g_nOptimalsToDisplay) # 결과값으로부터 최적 배향 찾기
Plot3D(mesh0, yaw_range, pitch_range, Mtotal, optimal_YPRs, worst_YPRs) # 찾은 최적 배향 그래프 출력.
