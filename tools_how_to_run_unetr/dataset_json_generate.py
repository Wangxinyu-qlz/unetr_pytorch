import glob
import os
import re
import json
from collections import OrderedDict

# 根目录，包含 train/ val/ test/ 三个文件夹
path_originalData = r"C:\Users\Administrator\Desktop\unetr_pytorch\dataset\Task044_CTspine"

def list_sort_nicely(l):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    l.sort(key=alphanum_key)
    return l

# 读取train文件夹里的images和labels
train_image = list_sort_nicely(glob.glob(os.path.join(path_originalData, "train", "imagesTr", "*")))
train_label = list_sort_nicely(glob.glob(os.path.join(path_originalData, "train", "labelsTr", "*")))

# 读取val文件夹里的images和labels
val_image = list_sort_nicely(glob.glob(os.path.join(path_originalData, "val", "imagesTr", "*")))
val_label = list_sort_nicely(glob.glob(os.path.join(path_originalData, "val", "labelsTr", "*")))

# 读取test文件夹里的images和labels
test_image = list_sort_nicely(glob.glob(os.path.join(path_originalData, "test", "imagesTr", "*")))
test_label = list_sort_nicely(glob.glob(os.path.join(path_originalData, "test", "labelsTr", "*")))

# 只保留文件名，不含路径
train_image = [os.path.basename(x) for x in train_image]
train_label = [os.path.basename(x) for x in train_label]
val_image = [os.path.basename(x) for x in val_image]
val_label = [os.path.basename(x) for x in val_label]
test_image = [os.path.basename(x) for x in test_image]
test_label = [os.path.basename(x) for x in test_label]

print(f"train images: {len(train_image)}, train labels: {len(train_label)}")
print(f"val images: {len(val_image)}, val labels: {len(val_label)}")
print(f"test images: {len(test_image)}, test labels: {len(test_label)}")
print(f"example train image: {train_image[0]}")

# 构造json字典
json_dict = OrderedDict()
json_dict['name'] = "CTspine"
json_dict['description'] = "Segmentation"
json_dict['tensorImageSize'] = "3D"
json_dict['reference'] = "see challenge website"
json_dict['licence'] = "see challenge website"
json_dict['release'] = "0.0"
json_dict['modality'] = {"0": "CT"}
json_dict['labels'] = {
    "0": "background",
    "1": "C1",
    "2": "C2",
    "3": "C3",
    "4": "C4",
    "5": "C5",
    "6": "C6",
    "7": "C7",
    "8": "T1",
    "9": "T2",
    "10": "T3",
    "11": "T4",
    "12": "T5",
    "13": "T6",
    "14": "T7",
    "15": "T8",
    "16": "T9",
    "17": "T10",
    "18": "T11",
    "19": "T12",
    "20": "L1",
    "21": "L2",
    "22": "L3",
    "23": "L4",
    "24": "L5",
    "25": "L6"
}

json_dict['numTraining'] = len(train_image)
json_dict['numValidation'] = len(val_image)
json_dict['numTest'] = len(test_image)

json_dict['training'] = []
for i in range(len(train_image)):
    json_dict['training'].append({
        "image": f"./train/imagesTr/{train_image[i]}",
        "label": f"./train/labelsTr/{train_label[i]}"
    })

json_dict['validation'] = []
for i in range(len(val_image)):
    json_dict['validation'].append({
        "image": f"./val/imagesTr/{val_image[i]}",
        "label": f"./val/labelsTr/{val_label[i]}"
    })

json_dict['test'] = []
for i in range(len(test_image)):
    # 注意test标签一般不公开，如果有标签可以填
    # 这里先写image路径即可
    json_dict['test'].append(f"./test/imagesTs/{test_image[i]}")

# 保存json文件
with open(os.path.join(path_originalData, "dataset.json"), 'w') as f:
    json.dump(json_dict, f, indent=4, sort_keys=False)

print("dataset.json generated successfully.")
