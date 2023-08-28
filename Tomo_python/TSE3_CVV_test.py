# TomoNV example, C++ Dll version
from  tomoNV_Cpp import *
import os

#-----------------------------------------------
#(1) specify search conditon, filename and initial orientation.
#========================================================================================================================
# DataSet= [ ('..\\Tomo_MeshData\\(1)sphere90_39k.obj',   0, 0, 0)]
# DataSet= [ ('..\\Tomo_MeshData\\(2)cube50_nocut.obj', 0, 0, 0)]
# DataSet= [ ('..\\Tomo_MeshData\\(3)cone50_63k.obj', 0,  0, 0)] 
# DataSet= [ ('..\\Tomo_MeshData\\(4)Bunny_69k_2x.obj', 0,  0, 0)] 
# DataSet= [ ('..\\Tomo_MeshData\\(5)Bunny_69k.stl',  0,  0, 0)] 
# DataSet= [ ('..\\Tomo_MeshData\\(6)Bunny_69k_0.5x.stl',  0,  0, 0)] 
# DataSet= [ ('..\\Tomo_MeshData\\(7)Bunny_5k.stl', 0, 0, 0)] 
# DataSet= [ ('..\\Tomo_MeshData\\(8)Bunny_1k.obj',   0,  0, 0)] 
# DataSet= [ ('..\\Tomo_MeshData\\(9)manikin.obj',   0,  0, 0)] 

DataSet1= '..\\Tomo_MeshData\\(5)Bunny_69k.stl'
DataSet2= '..\\Tomo_MeshData\\(5)Bunny_69k_yaw90도.stl'
DataSet3= '..\\Tomo_MeshData\\(5)Bunny_69k_pitch90도.stl'

theta_YP = 0 #in Degree, should be integer


#========================================================================================================================
#-----------------------------------------------
(Yaw, Pitch, Roll) = (0,0,0)

if(theta_YP==0):
  # type 1). seeing a specific orientation.  
  nYPR_Intervals=1
  yaw_range   = np.ones(nYPR_Intervals) * toRadian(Yaw)
  pitch_range = np.ones(nYPR_Intervals) * toRadian(Pitch)
  roll_range  = np.ones(nYPR_Intervals) * toRadian(Roll)
elif(theta_YP> 0):
  # type 2). search optimal orientation
  nYPR_Intervals= int(360 / theta_YP) +1
  yaw_range   = np.linspace(toRadian(0), toRadian(360), num=nYPR_Intervals, endpoint=True, dtype=np.float32)
  pitch_range = np.linspace(toRadian(0), toRadian(360), num=nYPR_Intervals, endpoint=True, dtype=np.float32)
  roll_range  = np.zeros(1) #for generality. roll direction  is not needed.

# (2) specify 3D printer's g-code conditions.
tomoNV_Cpp1 = tomoNV_Cpp(DataSet1, nYPR_Intervals, yaw_range, pitch_range, roll_range , bVerbose=True)# load mesh file and Cpp Dll
tomoNV_Cpp1.bUseExplicitSS = True #Set as True to see the SS pixels visually.
tomoNV_Cpp1.bUseClosedVolumeVoxel = False
tomoNV_Cpp1.BedType = ( enumBedType.ebtRaft, 0, 2, 0.3 + 0.27 + 2 * 0.2)# 0, 래프트 크기(mm), 베이스 두께 + 인터페이스 두께  + 서피스 레이어 수 * 서피스 레이어 두께
tomoNV_Cpp1.Run(cpp_function_name = 'TomoNV_INT3') #call CPU version


#Plot3DPixels(tomoNV_Cpp1) #show pixels of the 1st optimal
#-----------------------------------------------
(Yaw, Pitch, Roll) = (90,0,0)

if(theta_YP==0):
  # type 1). seeing a specific orientation.  
  nYPR_Intervals=1
  yaw_range   = np.ones(nYPR_Intervals) * toRadian(Yaw)
  pitch_range = np.ones(nYPR_Intervals) * toRadian(Pitch)
  roll_range  = np.ones(nYPR_Intervals) * toRadian(Roll)
elif(theta_YP> 0):
  # type 2). search optimal orientation
  nYPR_Intervals= int(360 / theta_YP) +1
  yaw_range   = np.linspace(toRadian(0), toRadian(360), num=nYPR_Intervals, endpoint=True, dtype=np.float32)
  pitch_range = np.linspace(toRadian(0), toRadian(360), num=nYPR_Intervals, endpoint=True, dtype=np.float32)
  roll_range  = np.zeros(1) #for generality. roll direction  is not needed.

# (2) specify 3D printer's g-code conditions.
tomoNV_Cpp2 = tomoNV_Cpp(DataSet1, nYPR_Intervals, yaw_range, pitch_range, roll_range , bVerbose=True)# load mesh file and Cpp Dll
tomoNV_Cpp2.bUseExplicitSS = True #Set as True to see the SS pixels visually.
tomoNV_Cpp2.bUseClosedVolumeVoxel = False
tomoNV_Cpp2.BedType = ( enumBedType.ebtRaft, 0, 2, 0.3 + 0.27 + 2 * 0.2)# 0, 래프트 크기(mm), 베이스 두께 + 인터페이스 두께  + 서피스 레이어 수 * 서피스 레이어 두께
tomoNV_Cpp2.Run(cpp_function_name = 'TomoNV_INT3') #call CPU version

#Plot3DPixels(tomoNV_Cpp2) #show pixels of the 1s
(Yaw, Pitch, Roll) = (0,0,0)

if(theta_YP==0):
  # type 1). seeing a specific orientation.  
  nYPR_Intervals=1
  yaw_range   = np.ones(nYPR_Intervals) * toRadian(Yaw)
  pitch_range = np.ones(nYPR_Intervals) * toRadian(Pitch)
  roll_range  = np.ones(nYPR_Intervals) * toRadian(Roll)
elif(theta_YP> 0):
  # type 2). search optimal orientation
  nYPR_Intervals= int(360 / theta_YP) +1
  yaw_range   = np.linspace(toRadian(0), toRadian(360), num=nYPR_Intervals, endpoint=True, dtype=np.float32)
  pitch_range = np.linspace(toRadian(0), toRadian(360), num=nYPR_Intervals, endpoint=True, dtype=np.float32)
  roll_range  = np.zeros(1) #for generality. roll direction  is not needed.

# (2) specify 3D printer's g-code conditions.
tomoNV_Cpp3 = tomoNV_Cpp(DataSet3, nYPR_Intervals, yaw_range, pitch_range, roll_range , bVerbose=True)# load mesh file and Cpp Dll
tomoNV_Cpp3.bUseExplicitSS = True #Set as True to see the SS pixels visually.
tomoNV_Cpp3.bUseClosedVolumeVoxel = False
tomoNV_Cpp3.BedType = ( enumBedType.ebtRaft, 0, 2, 0.3 + 0.27 + 2 * 0.2)# 0, 래프트 크기(mm), 베이스 두께 + 인터페이스 두께  + 서피스 레이어 수 * 서피스 레이어 두께
tomoNV_Cpp3.Run(cpp_function_name = 'TomoNV_INT3') #call CPU version

#Plot3DPixels(tomoNV_Cpp3) #show pixels of the 1s


CVV1 = np.append( tomoNV_Cpp1.al_pxls.reshape( -1, 1), tomoNV_Cpp1.be_pxls.reshape( -1, 1)).reshape(-1,6)
CVV2 = np.append( tomoNV_Cpp2.al_pxls.reshape( -1, 1), tomoNV_Cpp2.be_pxls.reshape( -1, 1)).reshape(-1,6)
CVV3 = np.append( tomoNV_Cpp3.al_pxls.reshape( -1, 1), tomoNV_Cpp3.be_pxls.reshape( -1, 1)).reshape(-1,6)

cvv_list = [(CVV1, 0, 0), (CVV2, -90, 0) , (CVV3, 0, -90)]

from scipy.spatial.transform import Rotation as R

for (cvv, yaw, pitch) in cvv_list:
  AABB = np.array([1000,1000,1000,-1000,-1000,-1000])
  AABB = updateAABB3D( AABB, cvv)
  for voxel in cvv:
    vtx = voxel[0:3]
    #translation
    vtx[0] -= (AABB[0]+AABB[3])*0.5
    vtx[1] -= (AABB[1]+AABB[4])*0.5
    vtx[2] -= (AABB[2]+AABB[5])*0.5
    #rotation
    qn    = Rotation.from_euler('xyz', [[yaw, pitch, 0]], degrees=True)
    vtx   = qn.apply(vtx) 
    voxel[0] = vtx[0,0]
    voxel[1] = vtx[0,1]
    voxel[2] = vtx[0,2]



cvv_PixelTitles     = (     '(0°,0°)',          '(90°,0°)',    '(0°,90°)')
cvv_PixelVarNames   = (CVV1, CVV2, CVV3)
cvv_PixelColors      = (  'red',   'green', ' blue')

AABB3D = np.array([1000,1000,1000,-1000,-1000,-1000])
Plot3DPixels_data = []
for p_title, p_name,  p_color in zip (  cvv_PixelTitles, cvv_PixelVarNames, cvv_PixelColors   ):
  pxls =  p_name
  if pxls.shape[0] > 1e5:
    pxls = pxls[0::2]
  AABB3D = updateAABB3D( AABB3D, pxls)
  Plot3DPixels_data.append( 
      go.Scatter3d( 
        x=pxls[:,0],  y=pxls[:,1], z=pxls[:,2], name=p_title, mode='markers',
        customdata = [pxls[:,3],pxls[:,4],pxls[:,5]], 
          marker=dict( size=1, color=p_color,  colorscale='Jet',line=dict(width=0.0),opacity=0.5)))
fig = go.Figure( data=Plot3DPixels_data)
(x0,y0,z0,x1,y1,z1) = AABB3D
if y0!= y1 and x0 != x1:
  fig.update_scenes(aspectratio = dict( x=1.,y=(y1-y0) / (x1-x0), z=(z1-z0) / (x1-x0)))
fig.update_scenes(camera_projection_type='orthographic')
fig.show()  


