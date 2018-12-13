import datetime

def cut_rod1(p, n):
    if n==0:
        return 0
    q = 0
    for i in range(1,n+1):
        q = max(q, p[i-1] + cut_rod1(p, n-i))
    return q

def cut_rod2(p, n):
    r = []
    for i in range(n):
        tmp = []
        tmp.append(p[i])
        for j in range(i):
            tmp.append(p[j] + r[i-1-j])
        r.append(max(tmp))
    return max(r)

p = [7,11,14,17,18,20,21,22,24,24]
t0 = datetime.datetime.now().microsecond
print(cut_rod1(p, 10))
t1 = datetime.datetime.now().microsecond
print("{}μs".format(t1-t0))
t0 = datetime.datetime.now().microsecond
print(cut_rod2(p, 10))
t1 = datetime.datetime.now().microsecond
print("{}μs".format(t1-t0))
