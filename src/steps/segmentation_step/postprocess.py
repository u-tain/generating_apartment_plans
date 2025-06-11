import os
import cv2
import numpy as np
from sklearn.cluster import KMeans


def get_box(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = np.vstack(contours)

    rect = cv2.minAreaRect(all_contours)  
    box = cv2.boxPoints(rect)               
    box = np.int0(box)
    return box


def postprocess_door(path):
    depth_map = np.load(path)
    depth_flat = depth_map.reshape(-1, 1)

    # Выполним кластеризацию на 2 кластера
    kmeans = KMeans(n_clusters=2, random_state=0).fit(depth_flat)
    labels = kmeans.labels_.reshape(depth_map.shape)  # Классы в виде двумерного массива
    centers = kmeans.cluster_centers_  # Координаты центров кластеров

    # Определяем порог (среднее между центрами кластеров)
    threshold_kmeans = np.min(centers)
    mask = depth_map > (threshold_kmeans)
    return 1-mask

def detect_door(mask_door,mask_floor,mask_wall,room_path,img):
    if mask_door.sum()!=0:
        # запускаем прероцесс
        # если дверь с краю, то вероятно это часть двери, пропускаем
        if mask_door.sum()>mask_door[:,50:-50].sum():pass
        else:
            # кластеризация
            mask_door = postprocess_door(os.path.join(room_path,'depth',img[:-4]+'_depth.npy'))

            # обновляем маски пола и стен
            mask_floor *= (1-mask_door)
            mask_wall *= (1-mask_door)
            if not os.path.exists(os.path.join(room_path,'segms',img[:-4]+'_floor.png')) and not os.path.exists(os.path.join(room_path,'segms',img[:-4]+'_wall.png')):
                cv2.imwrite(os.path.join(room_path,'segms',img[:-4]+'_door.png'),mask_door*255)
    return mask_door,mask_floor,mask_wall
