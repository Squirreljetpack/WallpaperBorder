import colour as cl
import numpy as np
#converts from rgb to xyz
def rgb_to_xyz(p):
    RGB_to_XYZ_matrix = np.array(
        [[0.41240000, 0.35760000, 0.18050000],
        [0.21260000, 0.71520000, 0.07220000],
        [0.01930000, 0.11920000, 0.95050000]])
    illuminant_RGB = np.array([0.31270, 0.32900])
    illuminant_XYZ = np.array([0.34570, 0.35850])
    return cl.RGB_to_XYZ(p / 255, illuminant_RGB, illuminant_XYZ, 
							RGB_to_XYZ_matrix, 'Bradford')

#converts from rgb to lab
def rgb_to_lab(p):
    new = rgb_to_xyz(p)
    return cl.XYZ_to_Lab(new)

#converts from xyz to rgb
def xyz_to_rgb(p):
    XYZ_to_RGB_matrix = np.array(
        [[3.24062548, -1.53720797, -0.49862860],
        [-0.96893071, 1.87575606, 0.04151752],
        [0.05571012, -0.20402105, 1.05699594]])
    illuminant_RGB = np.array([0.31270, 0.32900])
    illuminant_XYZ = np.array([0.34570, 0.35850])
    newp = cl.XYZ_to_RGB(p, illuminant_XYZ, illuminant_RGB, 
							XYZ_to_RGB_matrix, 'Bradford')
    return newp * 255

#converts from lab to rgb
def lab_to_rgb(p):
    xyz = cl.Lab_to_XYZ(p)
    return xyz_to_rgb(xyz)
