import keras
import matplotlib
import sys
import time

data = None
def slow_process():
    print("Loading...")
    for i in range(5):
        time.sleep(1)
        print(i+1)
    return "useful data"
data = slow_process()

print(data)