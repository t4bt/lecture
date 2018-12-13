from cv2 import cv2
import numpy as np
import math

# Esc キー
ESC_KEY = 0x1b
# s キー
S_KEY = 0x73
# r キー
R_KEY = 0x72
# 特徴点の最大数
MAX_FEATURE_NUM = 300
# 反復アルゴリズムの終了条件
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
# インターバル （1000 / フレームレート）
INTERVAL = 30
# ビデオデータ
VIDEO_DATA = 'motion.avi'

class Motion:
    # コンストラクタ
    def __init__(self):
        # 表示ウィンドウ
        cv2.namedWindow("motion")
        # マウスイベントのコールバック登録
        cv2.setMouseCallback("motion", self.onMouse)
        # 映像
        self.video = cv2.VideoCapture(VIDEO_DATA)
        # インターバル
        self.interval = INTERVAL
        # 現在のフレーム（カラー）
        self.frame = None
        # 現在のフレーム（グレー）
        self.gray_next = None
        # 前回のフレーム（グレー）
        self.gray_prev = None
        # 特徴点
        self.features = None
        # 特徴点のステータス
        self.status = None
        # 特徴点集合
        self.data2d = []
        # 重心
        self.data2d_c = []
        # 計測行列D
        self.matD = np.empty((0,MAX_FEATURE_NUM))
        # 特異値分解
        self.matU = None
        self.matW = None
        self.matV = None

    # メインループ
    def run(self):

        # 最初のフレームの処理
        end_flag, self.frame = self.video.read()
        self.gray_prev = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        while end_flag:
            # グレースケールに変換
            self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            self.autoAddFeatures()

            # 特徴点が登録されている場合にOpticalFlowを計算する
            if self.features is not None:
                # オプティカルフローの計算
                features_prev = self.features
                self.features, self.status, err = cv2.calcOpticalFlowPyrLK( \
                                                    self.gray_prev, \
                                                    self.gray_next, \
                                                    features_prev, \
                                                    None, \
                                                    winSize = (10, 10), \
                                                    maxLevel = 3, \
                                                    criteria = CRITERIA, \
                                                    flags = 0)

                # 有効な特徴点のみ残す
                #self.refreshFeatures()

                # フレームに有効な特徴点を描画
                if self.features is not None:
                    for feature in self.features:
                        cv2.circle(self.frame, (feature[0][0], feature[0][1]), 4, (0, 0, 255), -1, 8, 0)

            # 特徴点保存
            data = self.features.reshape((-1,2))
            center = np.average(data, axis=0)
            if len(data)==MAX_FEATURE_NUM:
                self.data2d.append(data)
                self.data2d_c.append(center)
                self.matD = np.r_[self.matD, (data-center).T]

            # 表示
            cv2.imshow("motion", self.frame)

            # 次のループ処理の準備
            self.gray_prev = self.gray_next
            end_flag, self.frame = self.video.read()
            if end_flag:
                self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            
            # インターバル
            key = cv2.waitKey(self.interval)
            # "Esc"キー押下で終了
            if key == ESC_KEY:
                break
            # "s"キー押下で一時停止
            elif key == S_KEY:
                self.interval = 0
            elif key == R_KEY:
                self.interval = INTERVAL

        # 終了処理
        cv2.destroyAllWindows()
        self.video.release()


    # マウスクリックで特徴点を指定する
    #     クリックされた近傍に既存の特徴点がある場合は既存の特徴点を削除する
    #     クリックされた近傍に既存の特徴点がない場合は新規に特徴点を追加する
    def onMouse(self, event, x, y, flags, param):
        # 左クリック以外
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 最初の特徴点追加
        if self.features is None:
            self.addFeature(x, y)
            return

        # 探索半径（pixel）
        radius = 5
        # 既存の特徴点が近傍にあるか探索
        index = self.getFeatureIndex(x, y, radius)

        # クリックされた近傍に既存の特徴点があるので既存の特徴点を削除する
        if index >= 0:
            self.features = np.delete(self.features, index, 0)
            self.status = np.delete(self.status, index, 0)

        # クリックされた近傍に既存の特徴点がないので新規に特徴点を追加する
        else:
            self.addFeature(x, y)

        return


    # 指定した半径内にある既存の特徴点のインデックスを１つ取得する
    #     指定した半径内に特徴点がない場合 index = -1 を応答
    def getFeatureIndex(self, x, y, radius):
        index = -1

        # 特徴点が１つも登録されていない
        if self.features is None:
            return index

        max_r2 = radius ** 2
        index = 0
        for point in self.features:
            dx = x - point[0][0]
            dy = y - point[0][1]
            r2 = dx ** 2 + dy ** 2
            if r2 <= max_r2:
                # この特徴点は指定された半径内
                return index
            else:
                # この特徴点は指定された半径外
                index += 1

        # 全ての特徴点が指定された半径の外側にある
        return -1


    # 特徴点を新規に追加する
    def addFeature(self, x, y):

        # 特徴点が未登録
        if self.features is None:
            # ndarrayの作成し特徴点の座標を登録
            self.features = np.array([[[x, y]]], np.float32)
            self.status = np.array([1])
            # 特徴点を高精度化
            cv2.cornerSubPix(self.gray_next, self.features, (10, 10), (-1, -1), CRITERIA)

        # 特徴点の最大登録個数をオーバー
        #elif len(self.features) >= MAX_FEATURE_NUM:
            #print("max feature num over: " + str(MAX_FEATURE_NUM))

        # 特徴点を追加登録
        elif len(self.features) < MAX_FEATURE_NUM:
            # 既存のndarrayの最後に特徴点の座標を追加
            self.features = np.append(self.features, [[[x, y]]], axis = 0).astype(np.float32)
            self.status = np.append(self.status, 1)
            # 特徴点を高精度化
            cv2.cornerSubPix(self.gray_next, self.features, (10, 10), (-1, -1), CRITERIA)


    # 有効な特徴点のみ残す
    def refreshFeatures(self):
        # 特徴点が未登録
        if self.features is None:
            return

        # 全statusをチェックする
        i = 0
        while i < len(self.features):

            # 特徴点として認識できず
            
            if self.status[i] == 0:
                # 既存のndarrayから削除
                self.features = np.delete(self.features, i, 0)
                self.status = np.delete(self.status, i, 0)
                i -= 1

            i += 1

    
    def autoAddFeatures(self):
        # find Harris corners
        gray = np.float32(self.gray_next)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        dst = cv2.dilate(dst,None)
        ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        dst = np.uint8(dst)

        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # define the criteria to stop and refine the corners
        corners = cv2.cornerSubPix(gray,np.float32(centroids),(10,10),(-1,-1), CRITERIA)
        for corner in corners:
            x, y = corner
            self.addFeature(x,y)

    def calc_SVD(self):
        # Dを特異値分解
        self.matU, self.matW, self.matV = np.linalg.svd(np.array(self.matD))
        numframe = int(self.matU.shape[0] / 2)
        # Rank3をWにあてはめ(特異値の上位3つ以外を削除)
        W_3 = self.matW[:3]
        self.W_3 = np.array([
                                [math.sqrt(W_3[0]), 0, 0],
                                [0, math.sqrt(W_3[1]), 0],
                                [0, 0, math.sqrt(W_3[2])]
                            ])
        # 3次元座標を計算（射影復元）
        self.V_3 = self.matV[:3]
        self.S_d = np.dot(self.W_3, self.V_3)
        # モーション行列を計算（射影解）
        self.U_3 = self.matU[:,:3]
        self.M_d = np.dot(self.U_3, self.W_3)
        # 拘束行列の作成
        self.Met_C = np.array([[0., 0., 0., 0., 0., 0.] for i in range(numframe*3)])
        self.B = np.array([[0.] for i in range(numframe*3)])
        for i in range(numframe):
            # イコール0
            for j in range(2):
                self.Met_C[i*3+j, 0] = self.M_d[i*2+j, 0] ** 2
                self.Met_C[i*3+j, 1] = 2 * self.M_d[i*2+j, 0] * self.M_d[i*2+j, 1]
                self.Met_C[i*3+j, 2] = 2 * self.M_d[i*2+j, 0] * self.M_d[i*2+j, 2]
                self.Met_C[i*3+j, 3] = self.M_d[i*2+j, 1] ** 2
                self.Met_C[i*3+j, 4] = 2 * self.M_d[i*2+j, 1] * self.M_d[i*2+j, 2]
                self.Met_C[i*3+j, 5] = self.M_d[i*2+j, 2] ** 2

            # イコール1
            self.Met_C[i*3+2, 0] = self.M_d[i*2, 0] * self.M_d[i*2+1, 0]
            self.Met_C[i*3+2, 1] = self.M_d[i*2, 0] * self.M_d[i*2+1, 1] + self.M_d[i*2+1, 0] * self.M_d[i*2, 1]
            self.Met_C[i*3+2, 2] = self.M_d[i*2, 0] * self.M_d[i*2+1, 2] + self.M_d[i*2+1, 0] * self.M_d[i*2, 2]
            self.Met_C[i*3+2, 3] = self.M_d[i*2, 1] * self.M_d[i*2+1, 1]
            self.Met_C[i*3+2, 4] = self.M_d[i*2, 1] * self.M_d[i*2+1, 2] + self.M_d[i*2+1, 1] * self.M_d[i*2, 2]
            self.Met_C[i*3+2, 5] = self.M_d[i*2, 2] * self.M_d[i*2+1, 2]

            # Bにセット
            self.B[i*3, 0] = 1
            self.B[i*3+1, 0] = 1
            self.B[i*3+2, 0] = 0

        # 拘束式からX(=CC^Tの6自由度)を計算
        ok, self.X = cv2.solve(self.Met_C, self.B, flags=cv2.DECOMP_NORMAL)

        # XからCを計算
        self.mat = np.array([
                                [self.X[0,0], self.X[1,0], self.X[2,0]],
                                [self.X[1,0], self.X[3,0], self.X[4,0]],
                                [self.X[2,0], self.X[4,0], self.X[5,0]]
                            ])
        self.evals, self.evecs = np.linalg.eig(self.mat)
        self.evecs = self.evecs.T
        self.evalmat = np.array([
                                    [math.sqrt(self.evals[0]), 0, 0],
                                    [0, math.sqrt(self.evals[1]), 0],
                                    [0, 0, math.sqrt(self.evals[2])]
                                ])
        self.M = np.dot(self.M_d, self.evecs)
        self.M = np.dot(self.M, self.evalmat)
        R_e0 = self.M[0].T
        R_e1 = self.M[1].T
        R_e2 = np.cross(R_e0, R_e1)
        self.R = np.array([
                            [R_e0[0], R_e1[0], R_e2[0]],
                            [R_e0[1], R_e1[1], R_e2[1]],
                            [R_e0[2], R_e1[2], R_e2[2]]
                        ])
        self.C = np.dot(self.evecs, self.evalmat)
        self.C = np.dot(self.C, self.R)

        # 3次元座標を計算
        self.C_inv = np.linalg.inv(self.C)
        self.S = np.dot(self.C_inv, self.S_d)


if __name__ == '__main__':
    M = Motion()
    M.run()
    M.calc_SVD()
    X, Y, Z = M.S
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # 3Dでプロット
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(X, Y, Z, "o", color="#ff0000", ms=4, mew=0.5)
    # 軸ラベル
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # 表示
    plt.show()