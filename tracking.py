import cv2
import numpy as np
import DNF as DNF
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import control as ct
from scipy import ndimage



# TODO : Pour installer opencv:
# sudo apt-get install opencv* python3-opencv

# Si vous avez des problèmes de performances
#
# self.kernel = np.zeros([width * 2, height * 2], dtype=float)
# for i in range(width * 2):
#     for j in range(height * 2):
#         d = np.sqrt(((i / (width * 2) - 0.5) ** 2 + ((j / (height * 2) - 0.5) ** 2))) / np.sqrt(0.5)
#         self.kernel[i, j] = self.difference_of_gaussian(d)
#
#
# Le tableau de poids latéreaux est calculé de la façon suivante :
# self.lateral = signal.fftconvolve(self.potentials, self.kernel, mode='same')

size = (96, 128)
dnf1 = DNF.DNF(size[1], size[0])
xm=size[1]/2
ym=size[0]/2
def selectByColor(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([80, 50, 50])
    upper_blue = np.array([110, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.normalize(mask, None, 0, 1,cv2.NORM_MINMAX,cv2.CV_32F)
    mask = cv2.resize(mask,(size[0],size[1]))
    #cv2.imshow('frame', frame)
    #cv2.imshow('res', mask)
    return mask

def move(x_speed, y_speed):
    print(x_speed, y_speed)
    #ct.move(x_speed, y_speed)

def findCenter():
    return ndimage.center_of_mass(dnf1.potentials)

def motorControl(center):
    # TODO utilisez la fonction ct.move pour déplacer la caméra
    if math.isnan(center[0]) and math.isnan(center[1]):
        depX=0
        depY=0
    else:
        depX = (xm -center[1])/9
        depY = (ym -center[0])/12

    if depX < -10:
        depX = -10
    if depX > 10:
        depX = 10
    if depY > 10:
        depY = 10
    if depY < -10:
        depY = -10
    move(depX, depY)

def track(frame):
    input = selectByColor(frame)
    dnf1.input = input
    cv2.imshow("Camera", frame)
    dnf1.update_map()
    cv2.imshow("Input", dnf1.input)
    cv2.imshow("Potentials", dnf1.potentials)
    center = findCenter()
    motorControl(center)

if __name__ == '__main__':
    cv2.namedWindow("Camera")
    vc = cv2.VideoCapture(0) # 2 pour la caméra sur moteur, 0 pour tester sur la votre.

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()

        # initialisez votre DNF ici


    else:
        rval = False

    while rval:
        rval, frame = vc.read()
        track(frame)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cv2.destroyAllWindows()
