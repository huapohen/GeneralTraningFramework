import os
import sys
import json
import glob
import random
from tqdm import tqdm
from ipdb import set_trace as ip


def v1():
    classes = [
        "no_rain",
        "light_rain",
        "moderate_rain",
        "heavy_rain",
        "unkown",
    ]

    classes_ind = {
        "no": 0,
        "light": 1,
        "light_moderate": 2,
        "moderate": 3,
        "moderate_heavy": 4,
        "heavy": 5,
    }

    set_version = "v1"
    bp = f"/data/lwb/data/rain/{set_version}/"

    lab_info_list = []
    for file_name in os.listdir(bp + "label"):
        with open(os.path.join(bp + "label", file_name)) as f:
            data = json.load(f)
        num = len(data)
        for i in tqdm(range(num)):
            img_name = data[i]["image"]
            prefix = "_".join(img_name.split("_")[:2])
            img_name = (os.sep).join([prefix, img_name])
            label = data[i]["annotations"]
            if len(label) == 1:
                lab = label[0].split("_")[0]
                if lab == "unkown":
                    continue
                ind = classes_ind[lab]
            else:
                label = [k.split("_") for k in label]
                if "light" in label:
                    lab = "light_moderate"
                    ind = 2
                else:
                    lab = "moderate_heavy"
                    ind = 4
            line = " ".join([img_name, lab, str(ind)])
            lab_info_list.append(line)

    random.seed(925)
    random.shuffle(lab_info_list)

    for mode in ["train", "test"]:
        sv_path = f"{bp}/{mode}_file_names.txt"
        if os.path.exists(sv_path):
            os.remove(sv_path)
        ratio = 0.9 if mode == "train" else 0.1
        num = int(len(lab_info_list) * ratio)
        line_list = lab_info_list[:num]
        if mode == "test":
            line_list = sorted(line_list)
        with open(sv_path, "a+") as f:
            for line in line_list:
                f.write(line + "\n")


def v2():
    """
    {
        "image": "rain_0006_000303601.jpg",
        "annotations": [
            "中",
            "大"
        ]
    },
    100%|███████████████████████████████████| 163/163   [00:00<00:00, 251998.36it/s]
    100%|███████████████████████████████████| 2262/2262 [00:00<00:00, 397416.15it/s]
    100%|███████████████████████████████████| 268/268   [00:00<00:00, 374566.30it/s]
    100%|███████████████████████████████████| 2161/2161 [00:00<00:00, 542293.34it/s]
    100%|███████████████████████████████████| 4849/4849 [00:00<00:00, 509882.17it/s]
    100%|███████████████████████████████████| 5015/5015 [00:00<00:00, 474122.27it/s]
    100%|███████████████████████████████████| 5007/5007 [00:00<00:00, 501058.86it/s]
    100%|███████████████████████████████████| 5008/5008 [00:00<00:00, 525915.73it/s]
    """
    classes_ind = {
        "无": 0,
        # 无 小: 1
        "小": 2,
        # 小 中: 3
        "中": 4,
        # 中 大: 5
        "大": 6,
        "待": -1,
        "删": -1,
    }

    set_version = "v2"
    bp = f"/data/lwb/data/rain/{set_version}/"
    todo_list = []
    del_list = []

    lab_info_list = []
    for file_name in os.listdir(bp + "label"):
        with open(os.path.join(bp + "label", file_name)) as f:
            data = json.load(f)
        num = len(data)
        for i in tqdm(range(num)):
            img_name = data[i]["image"]
            if "rain_" in file_name:
                prefix = "_".join(img_name.split("_")[:2])
                img_relative_path = (os.sep).join([prefix, img_name])
            else:
                img_dir = file_name.split(".")[0]
                img_relative_path = (os.sep).join(["frames_split", img_dir, img_name])
            label = data[i]["annotations"]
            if len(label) == 1:
                if label[0] not in classes_ind.keys():
                    raise ValueError
                if label[0] in ["待", "删"]:
                    if label[0] == "待":
                        todo_list.append(img_relative_path)
                    else:
                        del_list.append(img_relative_path)
                    continue
                ind = classes_ind[label[0]]
            else:
                if "无" in label and "小" in label:
                    ind = 1
                elif "小" in label and "中" in label:
                    ind = 3
                elif "中" in label and "大" in label:
                    ind = 5
            line = img_relative_path + " " + str(ind)
            lab_info_list.append(line)

    random.seed(925)
    random.shuffle(lab_info_list)

    for mode in ["train", "test"]:
        sv_path = f"{bp}/{mode}_file_names.txt"
        if os.path.exists(sv_path):
            os.remove(sv_path)
        ratio = 0.95 if mode == "train" else 0.05
        num = int(len(lab_info_list) * ratio)
        line_list = lab_info_list[:num]
        if mode == "test":
            line_list = sorted(line_list)
        with open(sv_path, "a+") as f:
            for line in line_list:
                f.write(line + "\n")

    # # 23069 + 1213 + 356 + 92

    with open(f"{bp}/todo_list.txt", "w") as f:
        for key in todo_list:
            f.write(key + "\n")

    with open(f"{bp}/del_list.txt", "w") as f:
        for key in del_list:
            f.write(key + "\n")


if __name__ == "__main__":
    v2()
