import numpy as np
from glob import glob
import pickle
import time
from multiprocessing import Pool, Queue, Process, Value, Array, Manager, Lock
from glob import glob


def add_neighbor(source, target):
    t1 = time.time()
    print(f"opening {source}")
    with open(source, 'rb') as f:
        data_dt = pickle.load(f)
        xyzr = data_dt['data']
    tmp = np.zeros((len(xyzr), 12)) # first 4 is original xyzr, second 4 is one neighbor, third 4 is another neighbor, final one is a flag if more neighbors exist
    for i, point in enumerate(xyzr):
        dis = np.abs(xyzr[:, :-1] - point[:-1])
        dis_manhattan = np.sum(dis, axis=1)
        
        mx = np.max(dis_manhattan)
        dis_manhattan[i] = mx  # itself
        min_idx1 = np.argmin(dis_manhattan)
        dis_manhattan[min_idx1] = mx
        min_idx2 = np.argmin(dis_manhattan)

        relative_neighbor = xyzr[min_idx1] - point
        relative_neighbor2 = xyzr[min_idx2] - point
        point_wt_neighbor = np.hstack((point, relative_neighbor, relative_neighbor2))
        tmp[i] = point_wt_neighbor
    data_dt['data_wt_neighbors'] = tmp
    pickle.dump(data_dt,open(target, 'wb'))
    print(f"write new data to {target}, time cost: {time.time() - t1}")

    


def main():
    vessel_skeleton_ls = glob('/home/jjia/data/dataset/lung_function/ori_resolution/SSc_patient_*_skeleton_coordinates140000.pt')
    vessel_neighbor_ls = [i.replace('.pt', '_wt_neighbors.pt') for i in vessel_skeleton_ls]

    pool = Pool(processes=12)
    for source, target in zip(vessel_skeleton_ls[240:], vessel_neighbor_ls[240:]):
        pool.apply_async(add_neighbor, args=(source, target, ))
    pool.close()
    pool.join()
    print('finish')


if __name__ == '__main__':
    main()
