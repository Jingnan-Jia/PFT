from multiprocessing import Pool, Queue, Process, Value, Array, Manager, Lock
from glob import glob
from skimage.morphology import skeletonize_3d
from medutils.medutils import load_itk, save_itk
import itk
import SimpleITK as sitk
from tqdm import tqdm

def skeleton(file_q, process_lock):
    while True:
        file = None
        with process_lock:
            if not file_q.empty():
                file = file_q.get()
        if file:
            pass
        else:
            break

def skeleton_vessel(file_fpath, out_fpath):
    itkimage = itk.imread(file_fpath)
    # itkimage = sitk.ReadImage(file_fpath)
    print(file_fpath)
    skeleton = itk.MedialThicknessImageFilter3D.New(itkimage)
    itk.imwrite(skeleton, out_fpath)

    
    # img, ori, sp = load_itk(file_fpath, require_ori_sp=True)
    # skeletonize_3d(img)

def main():
    vessel_HR_ls = sorted(glob("/home/jjia/data/dataset/lung_function/ori_resolution/SSc_patient_0422335.mha"))
    # vessel_HR_ls = ["/home/jjia/data/dataset/lung_function/ori_resolution/SSc_patient_0422335.mha",
    # "/home/jjia/data/dataset/lung_function/ori_resolution/SSc_patient_0456204.mha",
    # "/home/jjia/data/dataset/lung_function/ori_resolution/SSc_patient_6216723.mha",
    # "/home/jjia/data/dataset/lung_function/ori_resolution/SSc_patient_6318939.mha"]
    vessel_skeleton_ls = [i.replace('.mha', '_skeleton.mha') for i in vessel_HR_ls]
    print(len(vessel_HR_ls))
    pool = Pool(processes=4)

    for source, target in zip(vessel_HR_ls, vessel_skeleton_ls):
        # skeleton_vessel(source, target)
        pool.apply_async(skeleton_vessel, args=(source, target, ))
    pool.close()
    pool.join()
    print('finish')
    # lock = Lock()
    # # file_q = Queue(maxsize=len(vessel_HR_ls))
    # # for i in vessel_HR_ls:
    # #     file_q.put(i)

    # p1 = Process(target=skeleton, args=(file_q, lock))
    # p3 = Process(target=skeleton, args=(file_q, lock))
    # multi_process(file_q)


if __name__ == '__main__':
    main()
