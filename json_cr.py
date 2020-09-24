from datetime import datetime
import json
import os
from collections import defaultdict
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import gc
import cv2
from torchvision.transforms import Compose, Normalize
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, KFold


def get_data(df):
    feats = defaultdict(lambda: defaultdict(list))

    for idx in range(len(df)):
        s = df.iloc[idx]
        map_key = s["key_frame"]
        frames = s["frames"]
        frames.sort(key=lambda x: x['frame_name'])
        key = [frame['frame_name'] for frame in frames].index(map_key)
        for name, feat in s['feats'].items():
            feats[name]['mean'].append(np.mean(feat))
            feats[name]['std'].append(np.std(feat))
            feats[name]['key'].append(feat[key])
            feats[name]['gap'].append(np.max(feat) - np.min(feat))

    train_df = pd.concat([
        df['id'],
        df['status'].rename('label').apply(int),
        *[pd.Series(v, name=f'{name}_{k}') for name, feat in feats.items() for k, v in feat.items()],
    ], axis=1)

    return train_df

if __name__ == '__main__':
    png_path = r"/home/jiamingjie/python_program/depth_amap_traffic_final_train_data"
    json_path = r"/home/jiamingjie/python_program/bts/amap_traffic_final_train_0906.json"
    # png_path = r"E:\depth_amap_traffice_final_train_data\depth_amap_traffic_final_train_data"
    # json_path = r"E:\amap_traffic_final_train_0906.json"
    # norm = dict(mean=[67.97004699707031],
    #      std=[28.86768913269043])

    # 写入 JSON 数据
    with open(json_path,'rb') as jsonfile:
        data = json.load(jsonfile)
    data_anno = data.get("annotations")
    png_list = os.listdir(png_path)
    png_list.sort()
    # k = 0
    # data_anno, png_list = data_anno[k:k+1],png_list[k:k+1]
    # data_anno,png_list = data_anno[:],png_list[:]
    for an,pn in zip(data_anno,png_list):
        print(pn)
        # an.setdefault('depth', [])

        idx_list = os.listdir(os.path.join(png_path,pn))
        idx_list.sort()

        for idx,it in enumerate(idx_list):
            frames = an.get('frames')
            itm = frames[idx]
            itm["feats"]={}
            imm = cv2.imread(os.path.join(os.path.join(png_path, pn), it),cv2.IMREAD_GRAYSCALE)
            # cv2.imshow("Image", imm)
            # cv2.waitKey(0)
            # im = np.array(Image.open(os.path.join(os.path.join(png_path, pn), it)))
            imm = cv2.resize(imm,(122,77))
            # im = (im - 9) / 8.8
            itm["feats"]["depth"]=imm
        del imm
        gc.collect()
            # frames =


            # an.get("depth").append(d_dic)

    filename = r'depth_amap_traffic_final_train_0906.pkl'#.format(k)
    with open(filename, 'wb+') as file_obj:
        pickle.dump(data_anno, file_obj)




    # train_df = get_data(train_json[:])
