import itertools
import numpy as np
from datetime import datetime

#しらみつぶし
def part_sum0(a, W):
    flg = False
    for i in range(1, len(a)+1):
        for x in itertools.combinations(a, i):
            if sum(x) == W:
                flg = True
    return flg

#動的計画法
def part_sum1(a, W):
    #初期化
    N=len(a)
    dp=[[0 for i in range(W+1)] for j in range(N+1)]
    #DP
    for i in range(1,N+1):
        for j in range(W+1):
            if j < a[i-1]:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = max(dp[i-1][j], a[i-1]+dp[i-1][j-a[i-1]])
    return dp[N][W]

time_list = []
W_list = np.random.choice(range(1,1001), 100, replace=False)

for num_w in range(10,21):
    list_w = np.random.choice(range(1,101), num_w, replace=False)
    time0 = datetime.now()
    time0 = time0 - time0
    for i in W_list:
        t0 = datetime.now()
        result = part_sum0(list_w, i)
        t1 = datetime.now()
        time0 += t1-t0
    print("time:{}".format(time0.total_seconds()))

    time1 = datetime.now()
    time1 = time1 - time1
    for i in W_list:
        t0 = datetime.now()
        result = part_sum1(list_w, i)
        t1 = datetime.now()
        time1 += t1-t0
    print("time:{}".format(time1.total_seconds()))

    time_list.append([num_w, time0.total_seconds(), time1.total_seconds()])

import csv
with open('data.csv', 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(time_list)
