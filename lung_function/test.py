import numpy as np
import pickle
from glob import glob
from skimage.morphology import skeletonize_3d
from medutils.medutils import load_itk, save_itk
import SimpleITK as sitk
import itk
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.morphology import disk,diamond,rectangle,square,erosion,dilation,opening,closing,skeletonize
import time


def interpolate(coor_all: np.ndarray, img: np.ndarray, target_points: int) -> np.ndarray:
    need_point_nb = target_points - len(coor_all)
    # print('needed points', need_point_nb)
    factor = target_points //len(coor_all) + 1
    added_point_ls = []
    for coor in coor_all:
        # print(len(added_point_ls))
        if len(added_point_ls) < need_point_nb:
            tmp_ls = interpolate_once(coor, img, factor)
            added_point_ls.extend(tmp_ls)
        else:
            break
        
    added_points = np.array(added_point_ls)
    assert len(added_points.shape) == len(coor_all.shape)
    assert added_points.shape[1] == coor_all.shape[1]
    out = np.concatenate((coor_all, added_points))
    return out


def interpolate_once(coor: np.ndarray, skl: np.ndarray, factor: int) -> list:
    """
    We only do linear interpolation between two "positive" voxels.
    """
    assert factor>=1

    added_ls = []
    z, y, x = int(coor[0]), int(coor[1]), int(coor[2])

    pos_coor = np.argwhere(skl[z:z+2, y:y+2, x:x+2]>0)   # if there is any neighboring positive values, do the interpolation
    # convert the relative cooridate to global one
    pos_coor += np.array([z,y,x])
     
    if len(pos_coor)>1:
        for i in pos_coor[1:]:
            # print('i', i)
            # print('zyx',z,y,x)
            # print(skl.shape)
            coor_change = (pos_coor[0] - i) / factor  # if only need 1 interpolation
            r_change = (skl[z,y,x] - skl[tuple(i)]) / factor
            for j in range(1, factor):  # two times means insert one value for each voxel
                if coor[0]%1!=0 or coor[1]%1!=0 or coor[2]%1!=0:
                    tmp = np.append(i + coor_change * j * 0.5, skl[tuple(i)] + r_change * j * 0.5)
                else:
                    tmp = np.append(i + coor_change * j, skl[tuple(i)] + r_change * j)
                added_ls.append(tmp) 
    return added_ls




skl_cord_ls = sorted(glob('/home/jjia/data/dataset/lung_function/ori_resolution/SSc_patient_???????_skeleton_coordinates.pt'))
skl_ls = sorted(glob('/home/jjia/data/dataset/lung_function/ori_resolution/SSc_patient_???????_skeleton.mha'))
llss = []
for skl_img, skl_cord_np in tqdm(zip(skl_ls, skl_cord_ls)):
    tmp = skl_cord_np
    with open(tmp, 'rb') as handle:
        a = pickle.load(handle)

    x = a['data']
    # print('x len', len(x))
    np.take(x,np.random.permutation(x.shape[0]),axis=0,out=x)

    skl, ori, sp = load_itk(skl_img, require_ori_sp=True)

    tt = time.time()
    if len(x)<140000:
        new_coor_all = interpolate(coor_all=x, img=skl, target_points=140000) 
        # print('total points',len(new_coor_all))
        if len(new_coor_all)<140000:  # second interpolation !!
            print(f'{skl_cord_np} original point number: {len(x)}, first intplt: {len(new_coor_all)}')
            new_coor_all = interpolate(coor_all=new_coor_all, img=skl, target_points=140000) 
            print(f"second intplt: {len(new_coor_all)}")
        if len(new_coor_all)<140000:  # third interpolation !!
            new_coor_all = interpolate(coor_all=new_coor_all, img=skl, target_points=140000) 
            print(f"third intplt: {len(new_coor_all)}")
        if len(new_coor_all)<140000:
            raise Exception(f"Why?!")

        tt2 = time.time()
        a['data'] = new_coor_all
    else:
        print(f"{skl_cord_np} has over 140000 points, it is {len(x)}")
    llss.append(new_coor_all)
    pickle.dump(a,open(f"{skl_cord_np.replace('.pt', '140000.pt')}", 'wb'))
        # print('time cost', tt2-tt)
print('finished')
