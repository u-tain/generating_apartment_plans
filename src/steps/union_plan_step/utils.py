import itertools
import cv2
import numpy as np
import os

def make_preprocess(room_path1, room_path2,imgs1,imgs2):
    print('начинаем препроцессинг')
    flat_path = os.path.dirname(os.path.dirname(room_path1))
    # получаем путь до изображений
    # составляем файл с парами
    
    imgs1 = [os.path.join(room_path1,item) for item in imgs1 if '.' in item]
    imgs2 = [os.path.join(room_path2,item) for item in imgs2 if '.' in item]
    imgs = imgs1 + imgs2
    pairs = list(itertools.product(imgs1, imgs2))# получаем возможные комбинации
    os.makedirs(os.path.join(flat_path,'dump_match_pairs'),exist_ok=True)
    with open(os.path.join(flat_path,'dump_match_pairs', 'pairs.txt'), 'w') as file:
        for i in range(len(pairs)):
            if pairs[i][0]!=pairs[i][1]:
                file.write(f"{pairs[i][0]} {pairs[i][1]}\n")
                file.write(f"{pairs[i][1]} {pairs[i][0]}\n")

    # изменяем размер всех изображений в удобный формат
    # пока рассматриваем только горизонтальные
    for item in imgs:
        # print(os.path.join(self.flat_path,item))
        img = cv2.imread(item)
        img = cv2.resize(img, (640, 480))
        cv2.imwrite(item,img)
    print('препроцессинг закончен')


def get_mask_center(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)
    return center

def distance_point_to_segment(p, v, w):
    """Вычисляет расстояние от точки p до отрезка vw."""
    vw = np.array(w) - np.array(v)
    vp = np.array(p) - np.array(v)
    
    vw_length_squared = np.dot(vw, vw)
    if vw_length_squared == 0:  # если v == w
        return np.linalg.norm(vp)
    
    t = np.dot(vp, vw) / vw_length_squared
    t = np.clip(t, 0, 1)
    
    nearest_point = v + t * vw
    return np.linalg.norm(np.array(p) - nearest_point)

def distance_segments(seg1, seg2):
    """Вычисляет максимальное расстояние между двумя отрезками seg1 и seg2."""
    v1, w1 = seg1
    v2, w2 = seg2
    distances = [
        distance_point_to_segment(v1, v2, w2),  # Расстояние от v1 до отрезка [v2, w2]
        distance_point_to_segment(w1, v2, w2),  # Расстояние от w1 до отрезка [v2, w2]
        distance_point_to_segment(v2, v1, w1),  # Расстояние от v2 до отрезка [v1, w1]
        distance_point_to_segment(w2, v1, w1)   # Расстояние от w2 до отрезка [v1, w1]
    ]
    return max(distances)

# door1: {'left': array([166.       , 283.5026178]), 'right': array([306.       , 283.5026178])}
def draw_circle(center, text, mask,step=0):
    cv2.circle(mask, (int(center[0])+step,int(center[1])+step),5,255,-1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (255, 0, 0)  
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate the text position to center it within the circle
    text_x = int(center[0])+step - text_width // 2+ 15
    text_y = int(center[1])+step + text_height // 2 

    # Add text to the image
    cv2.putText(mask, text, (text_x, text_y), font, font_scale, font_color, thickness)
    return mask

def draw_result(doors_pos_centers):
    all_points = []
    for img in doors_pos_centers.keys():
        if img != 'ordered_points':
            for door in doors_pos_centers[img].keys():
                if 'connect_room_points' in doors_pos_centers[img][door]:
                    all_points += list(doors_pos_centers[img][door]['connect_room_points'])
        else:
            all_points += list(doors_pos_centers['ordered_points'])
    min_x = np.min(np.array(all_points)[:, 0])
    min_y = np.min(np.array(all_points)[:, 1])
    max_x= np.max(np.array(all_points)[:, 0])
    max_y = np.max(np.array(all_points)[:, 1])

    # Если минимальные значения отрицательные, сдвигаем все точки
    if min_x < 0 or min_y < 0:
            shift_x = abs(min_x) if min_x < 0 else 0
            shift_y = abs(min_y) if min_y < 0 else 0
    else:
            shift_x = 0
            shift_y = 0
    stride = int(max(shift_x,shift_y))+50

    h = w = int(max(max_x,max_y)) +stride + 100
    print((h,w))

    mask = np.zeros((h,w),dtype=np.uint8)


    for i in range(len(all_points)):
            draw_circle((int(all_points[i][0]),int(all_points[i][1])),' ',mask,stride)



    points =list(doors_pos_centers['ordered_points'])
    for i in range(len(points)):
            if i!= len(points)-1:
                cv2.line(mask,(int(points[i][0]+stride),int(points[i][1])+stride),(int(points[i+1][0]+stride),int(points[i+1][1])+stride),color=255,thickness=10)
            else:
                cv2.line(mask,(int(points[0][0]+stride),int(points[0][1])+stride),(int(points[i][0]+stride),int(points[i][1])+stride),color=255,thickness=10)
    for img in list(doors_pos_centers.keys()):
        if img != 'ordered_points':
            for door in doors_pos_centers[img].keys():
                if 'connect_room_points' in doors_pos_centers[img][door].keys():
                    points = doors_pos_centers[img][door]['connect_room_points']
                    for i in range(len(points)):
                        if i!= len(points)-1:
                            cv2.line(mask,(int(points[i][0]+stride),int(points[i][1])+stride),(int(points[i+1][0]+stride),int(points[i+1][1])+stride),color=255,thickness=10)
                        else:
                            cv2.line(mask,(int(points[0][0]+stride),int(points[0][1])+stride),(int(points[i][0]+stride),int(points[i][1])+stride),color=255,thickness=10)
                    
                point = doors_pos_centers[img][door]['door_points']
                cv2.line(mask,(int(point[0][0]+stride),int(point[0][1])+stride),(int(point[1][0]+stride),int(point[1][1])+stride),color=0,thickness=10)
                if 'window_pos' in doors_pos_centers[img][door].keys():
                    window_points_1,window_points_2=doors_pos_centers[img][door]['window_pos']

                    window_points_1 = [int(point+stride) for point in window_points_1]
                    window_points_2 = [int(point+stride) for point in window_points_2]
                    print(window_points_1)
                    print(window_points_2)
                    cv2.line(mask, window_points_1, window_points_2, color=0, thickness=10)
                    window_points_1[1] -=5
                    window_points_2[1] -=5

                    cv2.line(mask, window_points_1, window_points_2, color=255, thickness=2)

                    window_points_1[1] +=10
                    window_points_2[1] +=10

                    cv2.line(mask, window_points_1, window_points_2, color=255, thickness=2)
    return mask