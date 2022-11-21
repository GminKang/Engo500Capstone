import pykitti
import numpy as np
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

basedir = r'C:\Users\zifan\OneDrive\Desktop\School\ENGO 500\Test 1\data'
date ="2011_09_26"
drive = '0001'

# The 'frames' argument is optional - default: None, which loads the whole dataset.
# Calibration, timestamps, and IMU data are read automatically. 
# Camera and velodyne data are available via properties that create generators
# when accessed, or through getter methods that provide random access.

data = pykitti.raw(basedir, date,drive,)

# dataset.calib:         Calibration data are accessible as a named tuple
# dataset.timestamps:    Timestamps are parsed into a list of datetime objects
# dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
# dataset.camN:          Returns a generator that loads individual images from camera N
# dataset.get_camN(idx): Returns the image from camera N at idx  
# dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
# dataset.get_gray(idx): Returns the monochrome stereo pair at idx  
# dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
# dataset.get_rgb(idx):  Returns the RGB stereo pair at idx  
# dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
# dataset.get_velo(idx): Returns the velodyne scan at idx  
# Grab some data
second_pose = data.oxts[1].T_w_imu
first_gray = next(iter(data.gray))
first_cam1 = next(iter(data.cam1))

##Index = Idx = Time Stamp 

first_rgb = data.get_rgb(0)
first_cam2 = data.get_cam2(0)
third_velo = data.get_velo(107)

# Display some of the data
np.set_printoptions(precision=4, suppress=True)
print('\nDrive: ' + str(data.drive))
print('\nFrame range: ' + str(data.frames))

print('\nIMU-to-Velodyne transformation:\n' + str(data.calib.T_velo_imu))
print('\nGray stereo pair baseline [m]: ' + str(data.calib.b_gray))
print('\nRGB stereo pair baseline [m]: ' + str(data.calib.b_rgb))

print('\nFirst timestamp: ' + str(data.timestamps[0]))
print('\nSecond IMU pose:\n' + str(second_pose))

f, ax = plt.subplots(2, 2, figsize=(15, 5))
ax[0, 0].imshow(first_gray[0], cmap='gray')
ax[0, 0].set_title('Left Gray Image (cam0)')

ax[0, 1].imshow(first_cam1, cmap='gray')
ax[0, 1].set_title('Right Gray Image (cam1)')

ax[1, 0].imshow(first_cam2)
ax[1, 0].set_title('Left RGB Image (cam2)')

ax[1, 1].imshow(first_rgb[1])
ax[1, 1].set_title('Right RGB Image (cam3)')


f2 = plt.figure()
ax2 = f2.add_subplot(111, projection='3d')
# Plot every 100th point so things don't get too bogged down
velo_range = range(0, third_velo.shape[0], 100)
ax2.scatter(third_velo[velo_range, 0],
            third_velo[velo_range, 1],
            third_velo[velo_range, 2],
            c=third_velo[velo_range, 3],
            cmap='gray')
ax2.set_title('Third Velodyne scan (subsampled)')

plt.show()