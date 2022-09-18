import numpy as np
from medutils.medutils import load_itk, save_itk
import matplotlib.pyplot as plt
import SimpleITK

def main():
    ct_fpath: str = "/data1/jjia/ssc_scoring/ssc_scoring/dataset/SSc_DeepLearning/Pat_136/CTimage.mha"
    lung_mask_fpath: str = "/data1/jjia/ssc_scoring/ssc_scoring/dataset/SSc_DeepLearning/Pat_136/CTimage_lung.mha"
    vessel_mask_fpath: str = "/data1/jjia/dataset/vessel/train/gdth_ct/SSc/SSc_patient_47.mhd"

    ct, ori, sp = load_itk(ct_fpath, require_ori_sp=True)
    vessel_mask = load_itk(vessel_mask_fpath, require_ori_sp=False)
    dialation = SimpleITK.BinaryDilateImageFilter()
    lung_mask = load_itk(lung_mask_fpath, require_ori_sp=False)  # binary

    SHIFT_VALUE = 5000
    ct += SHIFT_VALUE  # shift all voxel values to higher values for the convenience of lung mask
    lung = ct * lung_mask
    lung -= SHIFT_VALUE
    ct -= SHIFT_VALUE

    lung_flat = lung.flatten()
    nb_voxels = lung_flat.shape[0]

    lung_flat = lung_flat[(lung_flat!=-SHIFT_VALUE)]
    # dras the histrogram

    pers75 = np.percentile(lung_flat, 75)  # The percentile is based on the voxels in lung
    pers25 = np.percentile(lung_flat, 25)

    lung_flat[lung_flat<pers25] = pers25
    lung_flat[lung_flat>pers75] = pers75
    lung_flat = lung_flat[(lung_flat!=pers25)&(lung_flat!=pers75)]
    while lung_flat.shape[0] < nb_voxels:
        lung_flat = np.concatenate((lung_flat, lung_flat))
    lung_flat = lung_flat[:nb_voxels]
    lung_tissue = lung_flat.reshape(lung.shape)

    ct_out = ct * (1 - vessel_mask) + lung_tissue * vessel_mask

    out_fpath = "results/tmp/ct_replaced.mha"
    save_itk(out_fpath, ct_out, ori, sp)

    plt.figure()
    plt.imshow(ct[300])
    plt.savefig("results/tmp/ct_ori.png")

    plt.figure()
    plt.imshow(ct_out[300])
    plt.savefig("results/tmp/ct_replaced.png")


if __name__ == "__main__":
    main()