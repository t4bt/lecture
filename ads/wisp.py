import pandas as pd
import numpy as np
from datetime import datetime

def BT(LIST, N):
    low = 0
    high = len(LIST)
    t = int((low+high)/2)
    while(low < high):
        if LIST[t] < N and N <= LIST[t+1]:
            break
        elif N <= LIST[t]:
            high = t
        elif LIST[t+1] < N:
            low = t + 1
        t = int((low+high)/2)
    return t

#Weighted Interval Selection Problem
def wisp(tasks):
    num_tasks = len(tasks)
    tasks = tasks.sort_values(by=["end"], ascending=True).reset_index()[["start","end","w"]]
    print(tasks)
    dp = [[0,0] for i in range(num_tasks)]
    t0 = datetime.now()
    dp[0] = [tasks["w"][0], 0]
    for d in range(1,num_tasks):
        S = BT(tasks["end"], tasks["start"][d])
        if 0 < S:   #task[d]の前に終了するtaskがある
            v = tasks["w"][d] + dp[S-1][0]  #それまでに終了するtask集合で最高の重み和を足す
        else:       #task[d]の前に終了するtaskが無い
            v = tasks["w"][d] + 0           #それまでに終了するtaskが無いので、自身の重みのみ

        dp[d][0] = max(dp[d-1][0], v)       #task[d]を採用するか否か、重み和の大きい方を採用

        if dp[d-1][0] < v:  #task[d]採用
            dp[d][1] = S    #S <= d
        else:               #task[d]不採用
            dp[d][1] = d + 1

    t1 = datetime.now()
    print(np.array(dp))

    d = num_tasks
    result = []
    while(0 < d):
        if dp[d-1][1] < d:
            result.append(list(tasks.iloc[d-1]) + [dp[d-1][0]])
            d = dp[d-1][1]
        else:
            d -= 1
    result.reverse()
    result = pd.DataFrame(result, columns=["start", "end", "w", "total"])
    print(result)
    print((t1-t0).total_seconds())

tasks = []
for i in range(10000):
    a = np.random.choice([i for i in range(1000)], 2, replace=False)
    a.sort()
    b = np.random.choice([i+1 for i in range(100)], 1)
    task = list(a) + list(b)
    tasks.append(task)
tasks = pd.DataFrame(tasks, columns=["start", "end", "w"])
wisp(tasks)
