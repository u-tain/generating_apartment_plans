import cv2
import numpy as np

def detect_position(image_w_door, planes,path,tag, idx=None):
    mask = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
    mask = mask[:,:,-1]
    y_point, x_point,_ = planes[image_w_door]['cross_point'][0] # Координаты точки

    # Получение координат маски
    y_indices, x_indices = np.where(mask == 255)

    # Определение границ маски
    min_x = np.min(x_indices)
    max_x = np.max(x_indices)

    distance_to_left_edge = x_point - min_x 
    distance_to_right_edge = max_x - x_point 

    if len(planes['order'])!=len(planes['final_points']):
        order = [str(i) for i in range(len(planes['final_points']))]
    else:
        order = planes['order']
    
    if idx is None:
        idx = order.index(image_w_door)
    print(len(order))
    # Определение позиции точки относительно маски
    if y_point < min_x or distance_to_left_edge < distance_to_right_edge:
        position = f"{tag} на правой стене"
        idx +=1
        if idx == len(order):
            idx = 0
        second_wall_point = order[idx]
        pos = f'{image_w_door}_{second_wall_point}'
        planes[image_w_door][f'{tag}_pos'] = f'{image_w_door}_{second_wall_point}'
    elif x_point > max_x or distance_to_left_edge > distance_to_right_edge:
        position = f"{tag} на левой стене"
        second_wall_point = order[idx-1 ]
        pos = f'{second_wall_point}_{image_w_door}'
        planes[image_w_door][f'{tag}_pos'] = f'{second_wall_point}_{image_w_door}'
    else:
        position = "Не понятно, точка по середине"
        pos = ''
    print(idx,position)
    return planes, pos