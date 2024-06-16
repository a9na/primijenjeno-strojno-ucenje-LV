import os

temp = 0
with open("/home/profesor/Downloads/dataset/Test.csv") as f:
    for line in f.readlines():
        if temp == 0:
            temp += 1
            continue
        info = line.split(",")
        
        try:
            os.mkdir("/home/profesor/Downloads/dataset/TestSorted/" + info[-2])
        except Exception:
            pass
        
        os.rename("/home/profesor/Downloads/dataset/" + info[-1][:-1], "/home/profesor/Downloads/dataset/TestSorted/" + info[-2] + "/" + info[-1][:-1].split("/")[-1])
        
