from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math

path = os.path.dirname(os.path.abspath(__file__))

img1 = cv2.imread(path+'/left.jpg',0)  #queryimage # left image
img2 = cv2.imread(path+'/right.jpg', 0)  # trainimage # right image
#"""
matches = [[[1194,759],[1400,610]],
           [[1211,411],[2009,390]],
           [[504,826],[1120,511]],
           [[834,440],[1734,346]],
           [[575,1608],[402,855]],
           [[1562,100],[2938,170]],
           [[60,205],[1525,82]],
           [[2050,245],[3153,437]]]
matches = np.array(matches)

addpoint = [[[1573,353],[2436,428]],
            [[1359,528],[1903,511]],
            [[1744,454],[2366,558]],
            [[1835,884],[1489,868]],
            [[620,200],[1888,125]],
            [[1853,582],[2044,681]],
            [[1912,1528],[676,1200]],
            [[2685,1200],[1482,1456]]]
addpoint = np.array(addpoint)
#"""
"""
matches = [[[860,50],[833,50]],
           [[2276,34],[2250,22]],
           [[912,977],[880,915]],
           [[2265,896],[2228,911]],
           [[1615,418],[1560,409]],
           [[758,1099],[725,1047]],
           [[2264,1039],[2205,1057]],
           [[1638,1212],[1519,1199]]]
matches = np.array(matches)

addpoint = [[[182, 63], [301, 70]],
            [[2386, 663], [2491, 678]],
            [[1248, 319], [1202, 308]],
            [[315, 1774], [141, 1639]],
            [[1390, 1495], [1178, 1456]],
            [[3202, 1625], [3053, 1731]],
            [[3113, 178], [3182, 165]],
            [[2456, 1365], [2298, 1401]]]
addpoint = np.array(addpoint)
"""
matches = np.array([
            [[575,1608],[402,855]],
            [[1562,100],[2938,170]],
            [[60,205],[1525,82]],
            [[2050,245],[3153,437]],
            [[2685,1200],[1482,1456]],
            [[1194,759],[1400,610]],
            [[1211,411],[2009,390]],
            [[504,826],[1120,511]],
            [[834,440],[1734,346]],
            [[1573,353],[2436,428]],
            [[1359,528],[1903,511]],
            [[1744,454],[2366,558]],
            [[1835,884],[1489,868]],
            [[620,200],[1888,125]],
            [[1853,582],[2044,681]],
            [[1912,1528],[676,1200]]
                    ])


pts1 = matches[:,0,:8]
pts2 = matches[:,1,:8]

#F行列の計算

F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_8POINT)
print(F)
print(np.linalg.matrix_rank(F))
print(np.linalg.det(F))

#E行列の計算
E, mask = cv2.findEssentialMat(pts1, pts2, focal=2.53866666667e+03, pp=(1632., 918.))
points, R, t, mask = cv2.recoverPose(E, pts1, pts2)

K = np.array([
    [2.53866666667e+03, 0.00000000000e+00, 1.63200000000e+03],
    [0.00000000000e+00, 2.53866666667e+03, 9.18000000000e+02],
    [0.00000000000e+00, 0.00000000000e+00, 1.00000000000e+00]
    ])

K_inv = np.linalg.inv(K)
"""
F = np.dot(K_inv.T, np.dot(E, K_inv))
F = F/F[2,2]
"""
#pts1 = np.r_[pts1,addpoint[:,0,:]]
#pts2 = np.r_[pts2,addpoint[:,1,:]]
pts1 = matches[:,0,:]
pts2 = matches[:,1,:]

"""
addpoint = [[[2090,395],[2926,625]],
            [[2170,848],[1750,973]]]
addpoint = np.array(addpoint)           
"""
"""
addpoint = [[[2010, 1457], [1798, 1462]],
            [[603, 303], [672, 292]],
            [[2892,938],[2950,985]]]
addpoint = np.array(addpoint)
"""

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - img2上の点に対応するエピポーラ線を描画する画像
        lines - 対応するエピポーラ線 '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    count = 0
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        count += 1
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,5)
        img1 = cv2.circle(img1,tuple(pt1),20,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),20,color,-1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img1,str(count),tuple(pt1), font, 3,color,5,cv2.LINE_AA)
    return img1,img2

# 右画像(二番目の画像)中の点に対応するエピポーラ線の計算
# 計算したエピポーラ線を左画像に描画
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# 左画像(一番目の画像)中の点に対応するエピポーラ線の計算
# 計算したエピポーラ線を右画像に描画
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

F = np.array(F)
print(F)

# エピポールの計算
def calc_epipole(Fmat):
    w, v = np.linalg.eig(np.dot(Fmat, Fmat.T))
    e1 = v[2] / v[2,2]
    return e1

e1 = calc_epipole(F)

# fの計算
def calc_camera_f(Fmat, cx, cy, e1):
    F = Fmat
    ex, ey, ez = e1
    a = cx - ex
    b = cy - ey
    alpha = F[0, 0]*cx + F[0, 1]*cy + F[0, 2]
    beta = F[1, 0]*cx + F[1, 1]*cy + F[1, 2]
    tmp0 = a**2 + 2*a*b - b**2
    tmp1 = 2*a*b*alpha**2 - tmp0*beta**2
    tmp2 = tmp0*(F[1,0]**2 + F[1,1]**2) - 2*a*b*(F[0,0]**2 + F[0,1]**2)
    f_2 = tmp1 / tmp2
    f2 = -a*b * (((F[0, 0]**2 + F[0, 1]**2)*f_2 + alpha**2) / ((F[0,0]*F[1,0] + F[0,1]*F[1,1])*f_2 + alpha*beta)) - b**2


    f = math.sqrt(abs(f2))
    f_ = math.sqrt(abs(f_2))
    return f, f_

print(calc_camera_f(F, 1632, 918, e1))

# 結果の表示
plt.figure(figsize=(18,6), dpi=300)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.savefig("epipolar1.jpg", bbox_inches="tight")
#"""