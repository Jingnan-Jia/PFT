

from mlflow import log_metric, log_metrics
import mlflow
import threading
import time
import random
# global_lock = threading.Lock()
from multiprocessing import Process, Pool
import os
from  mlflow.tracking import MlflowClient
client = MlflowClient()


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


def sub_thread4(id):
    info(f'sub_process: {id}')
    with mlflow.start_run(nested=True):
        for i in range(500):
            tmp4 = random.random()
            try:
                log_metrics({f'Accuracy{id}': tmp4, f'bccuracy{id}': tmp4, f'cccuracy{id}': tmp4, f'dccuracy{id}': tmp4, f'eccuracy{id}': tmp4}, step=i)
            except:
                print('------------------------------------------------------------------------------------')
                pass
            print(f'accuracy{id}: {tmp4}')
            time.sleep(tmp4)

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("test")

    with mlflow.start_run(nested=True):
        p4 = Process(target=sub_thread4, args=(4, ))
        p3 = Process(target=sub_thread4, args=(3, ))
        p2 = Process(target=sub_thread4, args=(2, ))
        p1 = Process(target=sub_thread4, args=(1, ))

        p4.start()
        p3.start()
        p2.start()
        p1.start()
        # for i in range(500):
        #     tmp0 = random.random()
        #     log_metric('Accuracy0', tmp0, step=i)
        #     print(f'accuracy0: {tmp0}')
        #     time.sleep(tmp0)

        # with Pool(5) as p:
        #     print(p.map(sub_thread4, [1, 2, 3]))


        p4.join()
        p3.join()
        p2.join()
        p1.join()

    print("Finished!")