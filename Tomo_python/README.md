TomoNV : prediction of optimal 3D printing orientation using pixelization approach. 
-------------
![tomonv_logo](https://user-images.githubusercontent.com/68842543/167759123-cff545db-47d0-4253-b339-e7df9818ef97.jpg)

#### 1) Install Python 3 and packages. 
     pip install python==3.7.8 
     pip install numpy==1.21.4
     pip install setuptools==60.2.0
     pip install matplotlib==3.5.0
     pip install pandas==1.3.5
     pip install plotly==5.6.0
     pip install scipy==1.7.3
     pip install open3d==0.13.0 #use 0.10.0 in Colab
     pip install ray==1.11.0 #not tested in Colab yet
     pip install numba==0.55.1

#### 2) Install VSCode  ( https://code.visualstudio.com/download)
Recommend using VScode if you want to use RAY package's parallel processing feature in the python version algorithm (Case1,Case2,and Case3).

#### 3) Download these files in a same folder in your PC.

    tomoNV.py           #Python version of TomoNV algorithm
    tomoNV_io.py        #python user interfaces, e.g. Plotly 3D graphs
    TomoNV_Win64.dll    #C++ version of TomoNV algorithm

    case1_SphereCapCalibration.py  #Python and C++ version
    case2_singleOrientation.py     #Python and C++ version
    case3_searchOptimal.py         #Python and C++ version
    case4_STomoNV_TMPxl.py	#calculation is done in "TomoNV_Win64.dll", rendering in Python. Stores pixel info with STomoPixel class, 7 integers.
    case5_STomoNV_INT3.py	 #Memory-optimized version of case 4. Sotres only 3 integers for pixels. Fastest version.

+ Setting "bUseCDLL" as "True" uses C++ DLL's calculation in the Case1,2,and 3.
+ Setting "bUseRAY" as "True" and "bUseCDLL" as "False" uses multi-threaded python calculation, but somewhat unstable in MS Windows. <https://docs.ray.io/en/latest/ray-overview/installation.html> )
+ Better not set "bUseRAY" and "bUseCDLL" as True simulatenously. Can be unstable. 

#### 4) Open "case5_STomoNV_INT3.py" in VSCode and set your mesh file name properly. For exmaple;
    filename = '..\\TomoNVData\\masha_standing1501.obj
(Mesh file is loaded by the Open3D package. Supported file formats are .ply, .obj, .stl, and .off. <http://www.open3d.org/docs/0.9.0/tutorial/Basic/file_io.html>).

#### 5) Set the search options.
5-1) Calculating filament mass in a specific orientation. 

    nYPR_Intervals = 1
    (Yaw, Pitch, Roll) = (45, 0, 0)    #IN DEGREES
    yaw_range   = np.ones(nYPR_Intervals) * toRadian(Yaw)
    pitch_range = np.ones(nYPR_Intervals) * toRadian(Pitch)
    roll_range  = np.ones(nYPR_Intervals) * toRadian(Roll)

+ In this case, comment out the lines of the following "5-2)".
 
5-2) Finding an optimal orientation. 

    nYPR_Intervals = 5     
    yaw_range   = np.linspace(toRadian(0), toRadian(360), num=nYPR_Intervals, endpoint=True, dtype=np.float64)
    pitch_range = np.linspace(toRadian(0), toRadian(360), num=nYPR_Intervals, endpoint=True, dtype=np.float64)
    roll_range  = np.zeros(1) 

+ Increasing the "nYPR_Intervals" leads to long calculation time.
+ For example, nYPR_Intervals=36 means that 360 / 36 = 10 degrees interval. In thie case, the calculation of "5-1)" is repeated 10 x 10 = 100 times for the (yaw, pitch)'s.
 
#### 6) Set the 3D printer parameters. The default values are set to the Sindoh DP-202 Printer.

    wall_thickness = 1.0 # [mm]
    PLA_density    = 0.00121 # density of PLA filament, [g/mm^3]
    Fclad = 1.0 # fill ratio of cladding, always 1.0
    Fcore = 0.15 # fill ratio of core, (0~1.0)
    Fss   = 0.2 # fill ratio of support structure, (0~1.0)
    Css   = 1. # correction constant for Mss. obsolete. replaced by getWeightZSum[]
    bDefaultParams = True
        (theta_c, Fcore, Fss) =  \
          (toRadian(60.) , 0.15, 0.20) if ( bDefaultParams )  \
        else  (toRadian( 0.) , 1.,   1.)
        
+ "Css" , "bDefaultParams", and "bUseSlotParing" are for research purposes

#### 7) Click F5(Run) in VScode.
To see graphs, uncomment lines such as.

    Plot2DTomo(this)
    Plot3DPixels(this) # Very slow when the input mesh file size is big.
    Plot3DPixels_matplotlib(this) #Just in case, when Plotly package is not avaiable.
    PrintSlotInfo( this, X=73,Y=68)   #seeing pixel information in the speicified (X,Y) slot.

p.s. 
+ The printer's maximal volume is assumed to be 256 x 256 x 256 mm^3. The input mesh file's dimension also should be smaller than this.
+ VS2019 Source code for "TomoNV_Win64.dll" is in <https://github.com/cfms-lab/TomoNVC_Win64>. 
+ C++ Dll uses the CPU's (maximal number-1) threads. 
