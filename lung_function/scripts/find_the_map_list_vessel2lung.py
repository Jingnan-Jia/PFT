import numpy as np
from medutils.medutils import load_itk, save_itk
from glob import glob
import json
from tqdm import tqdm

def main():
    ct_dir = "/data1/jjia/ssc_scoring/ssc_scoring/dataset/SSc_DeepLearning"
    ct_list = glob(ct_dir + "/*/CTimage.mha")

    ssc_77_dir = "/data1/jjia/dataset/vessel/train/ori_ct/SSc"
    ct_77_list = glob(ssc_77_dir + "/SSc_*.mhd")

    ct_imgs = []
    for file in tqdm(ct_list):
        ct = load_itk(file)[100]
        print(ct.shape)
        ct_imgs.append(ct)
        del ct

    ct_imgs77 = []
    for file in tqdm(ct_77_list):
        ct = load_itk(file)[100]
        print(ct.shape)
        ct_imgs77.append(ct)
        del ct

    map = {}
    for img_fpath, img in zip(ct_list, tqdm(ct_imgs)):
        for img77_fpath, img77 in zip(ct_77_list, tqdm(ct_imgs77)):
            if img.shape[0]==img77.shape[0]:
                if (img==img77).all():
                    map[img_fpath] = img77_fpath

    print(map)
    with open('map_vessel2lung_data.json', 'w') as fp:
        json.dump(map, fp)




if __name__ == "__main__":
    main()