# import numpy as np
# img = [[1,33,33213,5632,532,51,213,6,7],[2,133,3233213,59632,6532,251,2513,1,24]]
# img = np.array(img)
# print(img[:,0:4])
# img = [[1,2,3],[3,4,5]]
# print(img[-1])

# import time
# import random
# from multiprocessing import Queue, Process
#
#
# def consumer(name, q):
#     while True:
#         if q.empty():
#             print("饿啦，赶紧生产吧！")
#             time.sleep(0.2)
#             continue
#         res = q.get()
#         time.sleep(0.5)
#         print('消费者》》%s 准备开吃%s。' % (name, res))
#
#
# def producer(name, q):
#     for i in range(50):
#         time.sleep(2)
#         res = '大虾%s' % i
#         q.put(res)
#         print('生产者》》》%s 生产了%s' % (name, res))
#
#
# if __name__ == '__main__':
#     q = Queue()  # 一个队列
#
#     p1 = Process(target=producer, args=('monicx', q))
#     c1 = Process(target=consumer, args=('lili', q))
#
#     p1.start()
#     c1.start()
#     p1.join()
#     q.put(None)
import cv2
import time

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
times = 0
while True:
    times += 1
    if times == 20:
        break

    start = time.time()
    ret,img = video.read()
    end = time.time() - start

    print("spend %f time to captrue a pic" %(end))
