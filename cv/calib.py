import numpy as np
import cv2
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7,3),np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []

images = glob.glob('sample/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

    if ret == True:
        objpoints.append(objp)
        
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img,(7,6),corners2,ret)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(ret)
print(mtx)
print(dist)
print(rvecs)
print(tvecs)

"""
img = cv2.imread('left12.jpg')
h, w = img.shape[:2]

newcameramtx, roi =cv2.getOptimalNewCameraMatrix(mtx, dist,(w,h),1,(w,h))

dst = cv2.undistort(img,mtx,dist,None,newcameramtx)

x,y,w,h = roi
dst = dst[y:y+h,x:x+w]
cv2.imwrite('calibresult.jpg',dst)
"""