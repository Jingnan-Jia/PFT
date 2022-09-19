import sys
sys.path.append("../..")

from lung_function.modules.cam import GradCAM
import tqdm

def main():
    AttentionMethod = "GradCAM"  # or others
    if AttentionMethod=="GradCAM":
        attention = GradCAM(1405)
    else:
        raise Exception(f"Please set the correct AttentionMethod")

    ts_id = [68, 83, 36, 187, 238, 12, 158, 189, 230, 11, 35, 37, 137, 144, 17, 42, 66, 70, 28, 64, 210, 3, 49, 32,
             236, 206, 194, 196, 7, 9, 16, 19, 20, 21, 40, 46, 47, 57, 58, 59, 60, 62, 116, 117, 118, 128, 134, 216]

    for id in tqdm(ts_id[4:10 ]):
        for level in [1,2,3,4,5]:
            attention.run(id, level)
        print('finish id: ', id)
    print("finish all")


if __name__ == "__main__":
    main()