import sys
sys.path.append("../..")
from lung_function.modules.datasets import all_loaders
from lung_function.modules.cam import GradCAM
from tqdm import tqdm

def main():
    AttentionMethod = "GradCAM"  # or others
    Ex_id = 1405
    if AttentionMethod=="GradCAM":
        attention = GradCAM(Ex_id)
    else:
        raise Exception(f"Please set the correct AttentionMethod")


    ts_id = [68, 83, 36, 187, 238, 12, 158, 189, 230, 11, 35, 37, 137, 144, 17, 42, 66, 70, 28, 64, 210, 3, 49, 32,
             236, 206, 194, 196, 7, 9, 16, 19, 20, 21, 40, 46, 47, 57, 58, 59, 60, 62, 116, 117, 118, 128, 134, 216]

    for pat_id in tqdm(ts_id[:3]):
        attention.run(pat_id)
        print('Finish pat_id: ', pat_id)
    print("Finish all")


if __name__ == "__main__":
    main()