import matplotlib.pyplot as plt 
import os
import cv2
import shutil
from src.steps.key_points_step.super_glue import Super_Glue
from src.steps.key_points_step.points_utils import loadNPZ
from src.steps.union_plan_step.utils import *


def find_matches(rooms_path, room):
        
    num_matches = 0
    have_matches = {}
    rooms = os.listdir(rooms_path)
    rooms = [item for item in rooms if item != room]
    for item in rooms:
        room_path = os.path.join(rooms_path,room)
        imgs1 = os.listdir(os.path.join(rooms_path,room))
        imgs2 = os.listdir(os.path.join(rooms_path,item)) 
        imgs = imgs1+imgs2
        # запускаем поиск матчинга
        os.makedirs(os.path.join(room_path,'graphs'),exist_ok=True)
        if len(imgs)!= 1:
                m = Super_Glue(os.path.dirname(rooms_path))
                make_preprocess(room_path,os.path.join(rooms_path,item), imgs1,imgs2)
                m.prediction_stage()
                m.postprocess_stage()
                if len(os.listdir(os.path.join(os.path.dirname(rooms_path),'dump_match_pairs')))>1:
                    if len(os.listdir(os.path.join(os.path.dirname(rooms_path),'dump_match_pairs'))) > num_matches:
                        num_matches = len(os.listdir(os.path.join(os.path.dirname(rooms_path),'dump_match_pairs')))
                        have_matches[item] = True
                    else:
                        have_matches[item] = False

                    if len(os.listdir(os.path.join(os.path.dirname(rooms_path),'dump_match_pairs')))==3:
                        sources = os.listdir(os.path.join(os.path.dirname(rooms_path),'dump_match_pairs'))
                        sources = [item for item in sources if '.png' in item]
                        shutil.copy(os.path.join(os.path.dirname(rooms_path),'dump_match_pairs',sources[0]), 
                                    os.path.join(os.path.dirname(rooms_path),'graphs','key_points.png'))
                    else:
                        m.viz_result()
                else:
                    have_matches[item] = False
    return have_matches

def match_room_door(have_matches, flat_path, target_room, result_dict):
    imgs = os.listdir(os.path.join(flat_path,'by_rooms',target_room))
    imgs = [item.split('.')[0] for item in imgs if '.' in item]

    for match in have_matches.keys():
        if have_matches[match]:
            matches = os.listdir(os.path.join(flat_path,'dump_match_pairs'))
            matches = [item for item in matches if '.npz' in item]
            
            for img in imgs:
                print(img)
                mtchs = [item for item in matches if img in item]
                print(mtchs)
                if len(mtchs)>0:
                    for door in result_dict[img].keys():
                        print(door)
                        mask = cv2.imread(os.path.join(flat_path,'by_rooms',target_room,'separated_doors',f'{img}_{door}.png'))[:,:,1]
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for item in mtchs:
                            is_points_in_mask = []
                            for point in loadNPZ(item,os.path.join(flat_path,'by_rooms',target_room))[item.split('_').index(img)]:
                                inside = False
                                for contour in contours:
                                    # Используем pointPolygonTest для проверки
                                    if cv2.pointPolygonTest(contour, point, False) >= 0:
                                        inside = True
                                        break
                                is_points_in_mask.append(inside)
                            print(is_points_in_mask)
                            if all(is_points_in_mask):
                                result_dict[img][door]['match'] = item
                                result_dict[img][door]['match_room'] = match
    return result_dict

def define_doors_rooms(result_dict):
    # если матчей меньше чем кол-во остальных комнат 
    # ищем дальнюю дверь
    num_doors = 0
    num_matches = 0
    for item in result_dict.keys():
        d_list = [d for d in result_dict[item].keys() if 'door' in d]
        num_doors += len(d_list)
        for door in d_list:
            if 'match' in result_dict[item][door].keys():
                num_matches += 1

    if num_doors - num_matches==1:
        # дверь без матча -- выход
        for item in result_dict.keys():
            d_list = [d for d in result_dict[item].keys() if 'door' in d]
            for door in d_list:
                if 'match' not in result_dict[item][door].keys():
                    key = (item,door)
        result_dict[key[0]][key[1]]['is_exit'] = True
    
    elif num_matches==0:
        # не нашлось матчей, предположим расположение
        pass
    elif num_doors != num_matches:
        have_matches = []
        keys = []
        for item in result_dict.keys():
            d_list = [d for d in result_dict[item].keys() if 'door' in d]
            for door in d_list:
                if 'match' not in result_dict[item][door].keys():
                    keys.append((item,door))
                else:
                    have_matches.append((item,door))
        
        given_segment = result_dict[have_matches[0][0]][have_matches[0][1]]['door_points']

        max_distance = float('-inf')
        farthest_segment = None 
        for key in keys:
            segment = result_dict[key[0]][key[1]]['door_points']
            distance = have_matches(given_segment, segment)
            if distance > max_distance:
                max_distance = distance
                farthest_segment = key
        result_dict[farthest_segment[0]][farthest_segment[1]]['is_exit'] = True
    else:
        pass
    return  result_dict

def update_room_plan(rectangle_corners,door_points, door_info, target_corners, window_pos ):
    """ Поворачиваем точки комнаты так чтобы слить с коридором относительно дверных проемов
         rectangle_corners - Исходные точки комнаты
         door_points - Старые точки двери
         new_door_points - новые точки двери
         door_info - информация о двери из словаря doors_pos_centers
    """
    new_door_points = door_info['door_points']
    updated_rectangle_corners = rectangle_corners.copy()

    # проверка нужно ли повораяивать
    if door_points[0][0]==door_points[1][0] and  new_door_points[0][0] == new_door_points[1][0]:pass
    elif  (door_points[0][0]==door_points[1][0] and new_door_points[0][1] == new_door_points[1][1]) or(door_points[0][1]==door_points[1][1] and new_door_points[0][0] == new_door_points[1][0]) :
        print('нужно повернуть')
        # нужно повернуть комнату 
        for i,point in enumerate(updated_rectangle_corners):
                updated_rectangle_corners[i] = [point[1],point[0]]

        for i,point in enumerate(door_points):
                door_points[i] = [point[1],point[0]]

        if window_pos is not None:
            for i,point in enumerate(window_pos):
                window_pos[i] = [point[1],point[0]]
    
        # Определяем порядок точек двери
    doors_pos = {'door1': door_points}
    ordered_doors_2 = determine_left_right(doors_pos)

    doors_pos = {'door1': door_info['door_points']}
    ordered_doors = determine_left_right(doors_pos)
    
    # проверка нужно ли отражать 
    pos, points = detect_position_door(target_corners,door_points,new_door_points,updated_rectangle_corners)

    if img == '3' and door == 'door_1':
        pos = 'hor'
    print(pos)
    if pos is not None:
        if pos =='hor':
            axis_y = ordered_doors_2['door1']['left'][1] + (ordered_doors_2['door1']['right'][1]-ordered_doors_2['door1']['left'][1])/2
            updated_rectangle_corners = reflect_rectangle(updated_rectangle_corners,axis_y,mode='hor')
            if window_pos is not None:
                window_pos = transform_window(door_points, window_pos, updated_rectangle_corners, mode='hor')
            A = np.array(ordered_doors_2['door1']['left'])
            B = np.array(ordered_doors['door1']['right'])

            # линия 2
            C = np.array(ordered_doors['door1']['left'])
            D = np.array(ordered_doors_2['door1']['right'])
            
        elif pos =='vert':
            axis_x = ordered_doors_2['door1']['left'][0] + (ordered_doors_2['door1']['right'][0]-ordered_doors_2['door1']['left'][0])/2
            updated_rectangle_corners = reflect_rectangle(updated_rectangle_corners,axis_x,mode='vert')
            if window_pos is not None:
                window_pos =  window_pos = transform_window(door_points, window_pos, updated_rectangle_corners, mode='vert')
        
            A = np.array(ordered_doors_2['door1'][points[0]])
            B = np.array(ordered_doors['door1'][points[0]])

            # линия 2
            C = np.array(ordered_doors['door1'][points[1]])
            D = np.array(ordered_doors_2['door1'][points[1]])
            print(A,B,C,D)

    else:
        print('wehere')
        # Определяем порядок точек двери
        doors_pos = {'door1': door_points}
        ordered_doors_2 = determine_left_right(doors_pos)

        doors_pos = {'door1': door_info['door_points']}
        ordered_doors = determine_left_right(doors_pos)

        # проверяем нужно ли отзеркаливать
        # Определение линий
        # линия 1
        A = np.array(ordered_doors_2['door1'][points[0]])
        B = np.array(ordered_doors['door1'][points[0]])

        # линия 2
        C = np.array(ordered_doors['door1'][points[1]])
        D = np.array(ordered_doors_2['door1'][points[1]])
        print(A,B,C,D)
    # # Проверка пересечения линий
    if  do_intersect(A, B, C, D):
        print("Линии пересекаются. Отзеркаливаем прямоугольник.")
        
        if ordered_doors_2['door1']['left'][0] == ordered_doors_2['door1']['right'][0]:
            axis_y = ordered_doors_2['door1']['left'][1] + (ordered_doors_2['door1']['right'][1]-ordered_doors_2['door1']['left'][1])/2
            updated_rectangle_corners = reflect_rectangle(updated_rectangle_corners,axis_y,mode='hor')
            if window_pos is not None:
                window_pos = transform_window(door_points, window_pos, updated_rectangle_corners, mode='hor')
        elif ordered_doors_2['door1']['left'][1] == ordered_doors_2['door1']['right'][1]:
            # Отзеркаливание прямоугольника относительно оси х
            axis_x = ordered_doors_2['door1']['left'][0] + (ordered_doors_2['door1']['right'][0]-ordered_doors_2['door1']['left'][0])/2
            updated_rectangle_corners = reflect_rectangle(updated_rectangle_corners,axis_x,mode='vert')
            if window_pos is not None:
                window_pos = transform_window(door_points, window_pos, updated_rectangle_corners, mode='vert')

    else:
        print("Линии параллельны или не пересекаются. Ничего не делаем.")
        # проверим не лежат ли точки комнаты не с той стороны
        

    # ищем смещение
    if do_intersect(A, B, C, D) and pos is None:
        bias_x = ordered_doors_2['door1']['left'][0] - ordered_doors['door1']['left'][0]
        bias_y = ordered_doors_2['door1']['left'][1] - ordered_doors['door1']['left'][1]
    elif do_intersect(A, B, C, D) and pos=='hor':
        bias_x = ordered_doors_2['door1']['left'][0] - ordered_doors['door1']['left'][0]
        bias_y = ordered_doors_2['door1']['left'][1] - ordered_doors['door1']['left'][1]
    elif do_intersect(A, B, C, D) and pos =='vert':
        bias_x = ordered_doors_2['door1']['left'][0] - ordered_doors['door1']['right'][0]
        bias_y = ordered_doors_2['door1']['left'][1] - ordered_doors['door1']['right'][1]
    elif not do_intersect(A, B, C, D) and pos =='vert':
        bias_x = ordered_doors_2['door1']['left'][0] - ordered_doors['door1']['right'][0]
        bias_y = ordered_doors_2['door1']['left'][1] - ordered_doors['door1']['right'][1]
    elif not do_intersect(A, B, C, D) and pos =='hor':
        bias_x = ordered_doors_2['door1']['left'][0] - ordered_doors['door1']['right'][0]
        bias_y = ordered_doors_2['door1']['left'][1] - ordered_doors['door1']['right'][1]
    else:
        bias_x = ordered_doors_2['door1']['left'][0] - ordered_doors['door1']['left'][0]
        bias_y = ordered_doors_2['door1']['left'][1] - ordered_doors['door1']['left'][1]

    # print(bias_x, bias_y)
    for i,point in enumerate(updated_rectangle_corners):
        updated_rectangle_corners[i][0] = updated_rectangle_corners[i][0] - bias_x
        updated_rectangle_corners[i][1] = updated_rectangle_corners[i][1] - bias_y
    
    if window_pos is not None:
        for i in range(2):
            window_pos[i][0] = window_pos[i][0] - bias_x
            window_pos[i][1] = window_pos[i][1] - bias_y
    return  updated_rectangle_corners, window_pos


def transform_window(door_pos, window_pos, updated_rectangle_corners, mode='vert'):
    if mode == 'vert':
        if window_pos[0][0] ==  window_pos[1][0]:
            if door_pos[0][0] == door_pos[1][0]:
                # дверь и окна находятся на параллельных стенах
                new_x = [item[0] for item in updated_rectangle_corners if int(item[0])!=int(door_pos[0][0])]
                new_x = np.unique(new_x)
                new_x = np.max(new_x)
                window_pos[0][0] = new_x
                window_pos[1][0] = new_x
            elif door_pos[0][1] == door_pos[1][1]:
                print('дверь и окно на перпендикулярных стенах')
                x_center = np.min(door_pos,axis=0)[0] + (door_pos[0][0] - door_pos[1][0])/2
                window_pos[0][0] = x_center + (x_center - window_pos[0][0])
                window_pos[1][0] = x_center + (x_center - window_pos[1][0])
        elif window_pos[0][1] ==  window_pos[1][1]:
                if door_pos[0][0] == door_pos[1][0]:
                    # дверь и окно на перпендикулярных стенах
                    window_size = abs(window_pos[0][0] - window_pos[1][0])
                    if np.max(updated_rectangle_corners,axis=0)[0] > door_pos[0][0]:
                        # флип вправо
                        window_pos[0][0] = door_pos[0][0] + abs(door_pos[0][0] - np.max(window_pos,axis=0)[0])
                        window_pos[1][0] = window_pos[0][0] + window_size
                    elif np.max(updated_rectangle_corners,axis=0)[0] < door_pos[0][0]:
                        # флип влево
                        window_pos[0][0] = door_pos[0][0] - abs(door_pos[0][0]- np.min(window_pos,axis=0)[0])
                        window_pos[1][0] = window_pos[0][0] - window_size
                elif door_pos[0][0] == door_pos[1][0]:
                    print('все таки может вертикальный флип при горизонатальной двери и горизонтальном окне')
                pass
    elif mode == 'hor':
        if window_pos[0][1] ==  window_pos[1][1]:
            if door_pos[0][1] == door_pos[1][1]:
                #дверь и окна находятся на параллельных стенах
                new_y = [item[1] for item in updated_rectangle_corners if item[1]!=door_pos[0][1]]
                print('new_y',new_y)
                new_y = np.unique(new_y)
                new_y = np.max(new_y)
                window_pos[0][1] = new_y
                window_pos[1][1] = new_y
            elif door_pos[0][0] == door_pos[1][0]:
                print('сделать обновление окна для горизонтального флипа вертикальной двери')
        elif window_pos[0][0] ==  window_pos[1][0]:
            if door_pos[0][1] == door_pos[1][1]:
                # дверь и окно на перпендикулярных стенах
                pass
            elif door_pos[0][0] == door_pos[1][0]:
                #дверь и окна находятся на параллельных стенах
                print('горизонтальном отражении вертикальном окне дверь вертикально')
                y_center = np.min(door_pos,axis=0)[1] + (door_pos[0][1] - door_pos[1][1])/2
                window_pos[0][1] = y_center + (y_center + window_pos[0][1])
                window_pos[1][1] = y_center + (y_center + window_pos[1][1])
                # window_size = abs(window_pos[0][1] - window_pos[1][1])
                # y_center = np.max(door_pos,axis=0)[1] + (door_pos[0][1] - door_pos[1][1])/2
                # step_1 = y_center - window_pos[0][1]
                # step_2 = y_center - window_pos[1][1]
                # if abs(step_1)>abs(step_2):
                #     step = step_2
                # else:
                #     step = step_1
                # if step>0:
                #     window_pos[0][1] = y_center+step+window_size
                #     window_pos[1][1] = y_center+step+window_size
                # else:
                #     window_pos[0][1] = y_center+step-window_size
                #     window_pos[1][1] = y_center+step-window_size
                
    return window_pos


def orientation(p, q, r):
    """Определяет ориентацию трех точек."""
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # Коллинеарные
    elif val > 0:
        return 1  # Часовая
    else:
        return 2  # Против часовой

def on_segment(p, q, r):
    """Проверяет, находится ли точка q на отрезке pr."""
    return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

def do_intersect(p1, q1, p2, q2):
    """Проверяет, пересекаются ли отрезки p1q1 и p2q2."""
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # Общий случай
    if o1 != o2 and o3 != o4:
        return True

    # Специфические случаи
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False

def reflect_rectangle(rectangle, k ,mode = 'vert'):
    """Отразить прямоугольник относительно линии, проходящей через k.
    горизонтальной или вертикальной в зависимости от mode"""
    reflected_rectangle = []
    for point in rectangle:
        if mode == 'vert':
            # Отражаем по оси X = axis_x
            new_x = 2 * k - point[0]
            reflected_rectangle.append([new_x, point[1]])  # Y-координаты остаются неизменными
        else:
            new_y = 2 * k - point[1]  # Отражаем относительно Y = k
            reflected_rectangle.append([point[0], new_y]) 
    return np.array(reflected_rectangle)

def determine_left_right(doors_pos_centers):
    """
    Определяет порядок двух точек на стороне прямоугольника.

    :param doors_pos_centers: Словарь со значениями {двери: [x, y]}
    :return: Названия точек, где 'left' - левая точка, 'right' - правая точка
    """
    # Предполагаем, что каждая 'door' содержит 'door_points' с двумя точками.
    result = {}
    
    for door, points in doors_pos_centers.items():
        point1, point2 = points

        if point1[0] < point2[0]:  # Сравниваем x координаты
            result[door] = {'left': point1, 'right': point2}
        else:
            result[door] = {'left': point2, 'right': point1}

    return result

def detect_position_door(ordered_points,door_points,points_on_rectangle_side,updated_room_points):
    """определяем нужно ли отражать комнату
        ordered_points - точки коридора
        door_points - координаты  двери комнаты
        points_on_rectangle_side - координаты  двери комнаты на стороне коридора
    """
    pos = None
    points = None
    if door_points[0][0]==door_points[1][0] and points_on_rectangle_side[0][0] == points_on_rectangle_side[1][0]:
        # определим где точки 
        if points_on_rectangle_side[0][0] ==  np.max(ordered_points,axis=0)[0]:
            # дверь на правой стороне
            print('таргетная дверь на правой стороне')
            if int(door_points[0][0]) == int(np.min(updated_room_points,axis=0)[0]):
                print('двери на противоположных сторонах')
                pass
            elif int(door_points[0][0]) ==  int(np.max(updated_room_points,axis=0)[0]):
                print('обе двери на правой стороне')
                pos = 'vert'
                points =  ('left','right')

        elif points_on_rectangle_side[0][0] ==  np.min(ordered_points,axis=0)[0]:
            print('таргетная дверь на левой стороне')
            # таргетная дверь на левой стороне
            if int(door_points[0][0]) == int(np.min(updated_room_points,axis=0)[0]):
                # дверь комнаты тоже на левой стороне, отражаем вертикально
                pos = 'vert'
                print('обе двери на левой стороне')
            if int(door_points[0][0]) ==  int(np.max(updated_room_points,axis=0)[0]):
                print(' двери на противоположных сторонах')
                # дверь комнаты  на правой стороне стороне, отражать не надо
                pass

    elif door_points[0][1]==door_points[1][1] and points_on_rectangle_side[0][1] == points_on_rectangle_side[1][1]:

        if int(points_on_rectangle_side[0][1]) ==  int(np.min(ordered_points,axis=0)[1]):
            # дверь на верхней стороне
            print('таргетная дверь на верхней стороне')
            if int(door_points[0][1]) == int(np.max(updated_room_points,axis=0)[1]):
                # дверь на нижней стороне комнаты
                pos = None
                points =  ('left','right')
                print('дверь  на нижней части комнаты')
            elif int(door_points[0][1]) == int(np.min(updated_room_points,axis=0)[1]):
                print('дверь на верхней чатси комнаыь')
                pos = 'hor'
            else:
                pass
            # if door_points[0][0] < points_on_rectangle_side[0][0] and door_points[1][1] < points_on_rectangle_side[0][0]:
        elif int(points_on_rectangle_side[0][1]) ==  int(np.max(ordered_points,axis=0)[1]):
            # дверь на нижней стороне
            print('таргетная дверь на нижней стороне')
            if int(door_points[0][1]) == int(np.max(updated_room_points,axis=0)[1]):
                print('дверь  на нижней части комнаты')
                pos = 'hor'
                # points = ()
            elif int(door_points[0][1]) == int(np.min(updated_room_points,axis=0)[1]):
                print('дверь на верхней чатси комнаыь')
                pos = None
                points = ('left','left')
    return pos, points

# построцессинг для  склейки границ комнат

def get_wall_orient(wall):

    if wall[0][0]==wall[1][0]:
        orient = 'vert'
    elif wall[0][1]==wall[1][1]:
        orient = 'hor'
    else:
        orient = None
        print('стена не является вертикальной или горизонтальной')
    return orient

def get_connected_walls(orient,doors_pos_centers,wall):
    connected_walls = []
    idx = 1 if orient=='hor' else 0

    for img in doors_pos_centers.keys():
        if img!='ordered_points':
            for door in doors_pos_centers[img].keys():
                if 'connect_room_points' in  doors_pos_centers[img][door].keys():
                    if int(wall[0][idx]) in [int(point[idx]) for point in doors_pos_centers[img][door]['door_points']]:
                        connected_walls.append((img,door))
    return connected_walls

def update_coords(doors_pos_centers, connected_walls, coord_to_change,idx,new_coord):
                for i,item in enumerate(doors_pos_centers[connected_walls[0][0]][connected_walls[0][1]]['connect_room_points']):
                    if int(item[idx])==int(coord_to_change):
                        doors_pos_centers[connected_walls[0][0]][connected_walls[0][1]]['connect_room_points'][i][idx] = new_coord
                return doors_pos_centers

def stick_walls(doors_pos_centers):
    # будем идти по часовой


    room_for_slide = doors_pos_centers['ordered_points']

    # берем верхний левый угол
    left_top_point = [np.min(room_for_slide,axis=0)[0],np.min(room_for_slide,axis=0)[1]]

    start_idx = 0

    for i,item in enumerate(room_for_slide):
        print(item)
        print(left_top_point)
        if all(item == left_top_point):
            start_idx = i

    # сортируем точки поставив start_idx на 0 позицию
    if start_idx!=0:
        room_for_slide = room_for_slide[start_idx:] + room_for_slide[:start_idx]
        start_idx = 0

    for i, room_poinr in enumerate(room_for_slide):

        # берем след точку чтобы образовать стену
        wall = [room_for_slide[i],room_for_slide[i+1 if i+1 != len(room_for_slide) else 0]]

        # опредляем ориентацию комнаты
        orient = get_wall_orient(wall)
        print(f'Ориентация стены {orient}')

        # ищем комнату или комнаты прилегающие к этой стене
        connected_walls = get_connected_walls(orient,doors_pos_centers,wall)

        if len(connected_walls) == 1:
            print('к стене прилегает только одна комната')

            # берем следующую стену
            new_idx = i+2
            if  new_idx == len(room_for_slide):
                new_idx=0
            elif new_idx == len(room_for_slide)+1:
                new_idx=1
            wall2 = [room_for_slide[i+1 if i+1 != len(room_for_slide) else 0], room_for_slide[new_idx]]
            orient2 = get_wall_orient(wall2)
            print(f'Ориентация следующей стены {orient2}')
            connected_walls2 = get_connected_walls(orient2,doors_pos_centers,wall2)
            if len(connected_walls2)==0:
                print('К след стене не прилегает комната')
            elif len(connected_walls2)==1:
                print('К след стене  прилегает одна комната')
                # определяем координату для примыкания
                coord_to_change=None
                idx = 0 if orient=='hor' else 1
                if  orient=='hor' and np.min(room_for_slide,axis=0)[1]==wall[0][1]:
                    coord_to_change = np.max(doors_pos_centers[connected_walls[0][0]][connected_walls[0][1]]['connect_room_points'],axis=0)[idx]
                elif orient=='hor' and np.max(room_for_slide,axis=0)[1]==wall[0][1]:
                    coord_to_change = np.min(doors_pos_centers[connected_walls[0][0]][connected_walls[0][1]]['connect_room_points'],axis=0)[idx]
                elif orient=='vert' and np.min(room_for_slide,axis=0)[0]==wall[0][0]:
                    pass
                if coord_to_change:
                    doors_pos_centers = update_coords(doors_pos_centers, connected_walls, coord_to_change,idx,wall[1][idx] )

            else:
                print('К след стене  прилегает неск комнат')
            return doors_pos_centers

# постпроцессинг на пересекающиеся 

def get_min_max_corners(rectangle):
    """Возвращает минимальные и максимальные координаты x и y"""
    x_coords = rectangle[:, 0]
    y_coords = rectangle[:, 1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    return min_x, max_x, min_y, max_y

def rectangles_overlap(rect_a, rect_b):
    """Проверяет, пересекаются ли два прямоугольника."""
    min_a_x, max_a_x, min_a_y, max_a_y = get_min_max_corners(rect_a)
    min_b_x, max_b_x, min_b_y, max_b_y = get_min_max_corners(rect_b)

    # Проверка на пересечение
    return not (max_a_x < min_b_x or max_b_x < min_a_x or max_a_y < min_b_y or max_b_y < min_a_y)

def update_rectangle_to_touch(rect_a, rect_b):
    """Обновляет rect_a так, чтобы он соприкасался с rect_b."""
    min_a_x, max_a_x, min_a_y, max_a_y = get_min_max_corners(rect_a)
    min_b_x, max_b_x, min_b_y, max_b_y = get_min_max_corners(rect_b)
    print(min_a_x, max_a_x, min_a_y, max_a_y )
    print(min_b_x, max_b_x, min_b_y, max_b_y)

    # Проверка на изменение координат
    if max_a_x < min_b_x:  # A слева от B
        rect_a[:, 0] += (min_b_x - max_a_x)
    elif min_a_x > max_b_x:  # A вправо от B
        rect_a[:, 0] -= (min_a_x - max_b_x)
    print(min_a_y,max_b_y)
    if max_a_y < min_b_y:  # A ниже B
        rect_a[:, 1] += (min_b_y - max_a_y)
        
    elif min_a_y > max_b_y:  # A выше B
        print('here')
        rect_a[:, 1] -= (min_a_y - max_b_y)
    elif max_a_y>min_b_y: # А ниже верхней границы B
        for i,point in enumerate(rect_a):
            rect_a[i][1] = min_b_y if rect_a[i][1]==max_a_y else rect_a[i][1]



    return rect_a

def remove_cross_walls(doors_pos_centers):

    rect_a = np.array(doors_pos_centers['12']['door_1']['connect_room_points'])
    print(rect_a)

    rect_b = np.array(doors_pos_centers['12']['door_2']['connect_room_points'])
    print(rect_b)
    if rectangles_overlap(rect_a, rect_b):
        print("Прямоугольники пересекаются. Обновляем rect_a...")
        doors_pos_centers['12']['door_1']['connect_room_points'] = update_rectangle_to_touch(rect_a, rect_b)
        print("Обновленный rect_a:")
        print(doors_pos_centers['12']['door_1']['connect_room_points'])
    else:
        print("Прямоугольники не пересекаются.")
    return doors_pos_centers

# постпроцессинг на внешние стены
def stick_out_doors(doors_pos_centers):
    tresh = 160
    room_for_slide = doors_pos_centers['ordered_points']

    # берем верхний левый угол
    left_top_point = [np.min(room_for_slide,axis=0)[0],np.min(room_for_slide,axis=0)[1]]

    start_idx = 0

    for i,item in enumerate(room_for_slide):
        print(item)
        print(left_top_point)
        if all(item == left_top_point):
            start_idx = i

    for i, room_poinr in enumerate(room_for_slide):

        # берем след точку чтобы образовать стену
        wall = [room_for_slide[i],room_for_slide[i+1 if i+1 != len(room_for_slide) else 0]]

        # опредляем ориентацию комнаты
        orient = get_wall_orient(wall)
        print(f'Ориентация стены {orient}')

        # ищем комнату или комнаты прилегающие к этой стене
        connected_walls = get_connected_walls(orient,doors_pos_centers,wall)
        # берем следующую стену
        new_idx = i+2
        if  new_idx == len(room_for_slide):
                new_idx=0
        elif new_idx == len(room_for_slide)+1:
                new_idx=1
        wall2 = [room_for_slide[i+1 if i+1 != len(room_for_slide) else 0], room_for_slide[new_idx]]
        orient2 = get_wall_orient(wall2)
        print(f'Ориентация следующей стены {orient2}')
        connected_walls2 = get_connected_walls(orient2,doors_pos_centers,wall2)

        if len(connected_walls)==0:
            print('К  стене не прилегает комната')
            if len(connected_walls2)==1:
                print('К след стене  прилегает одна комната')

                if orient=='vert' and np.max(room_for_slide,axis=0)[0]==wall[0][0]:
                        # правая стена
                        # берем правую стену из connected_walls2
                        coord_to_change = np.max(doors_pos_centers[connected_walls2[0][0]][connected_walls2[0][1]]['connect_room_points'],axis=0)[0]
                        dist = wall[0][0] - coord_to_change
                        print(f'расстояние между стенами - {dist}')
                        if abs(dist)<tresh:
                            print('Расстояние меньше трещхолда двигаем стену')
                            for idx,item in enumerate(doors_pos_centers[connected_walls2[0][0]][connected_walls2[0][1]]['connect_room_points']):
                                if int(item[0])==int(coord_to_change):
                                    doors_pos_centers[connected_walls2[0][0]][connected_walls2[0][1]]['connect_room_points'][idx][0] = wall[0][0]
                if orient =='hor' and  np.max(room_for_slide,axis=0)[1]==wall[0][1]:
                    # нижняя стена
                    coord1 = np.max(doors_pos_centers[connected_walls2[0][0]][connected_walls2[0][1]]['connect_room_points'],axis=0)[1]
                    coord2 = wall[0][1]
                    dist = coord1-coord2
                    print(f'расстояние между стенами - {dist}')
                    if abs(dist)<tresh:
                        doors_pos_centers = update_coords(doors_pos_centers, connected_walls2, coord1, 1, coord2 )




        elif len(connected_walls) == 1:
            print('к стене прилегает только одна комната')
            
            
            if len(connected_walls2)==0:
                print('К след стене не прилегает комната')
                # считаем расстояние между стеной комнаты и след стеной коридора
                if  orient=='hor' and np.min(room_for_slide,axis=0)[1]==wall[0][1]:
                    pass
                elif orient=='hor' and np.max(room_for_slide,axis=0)[1]==wall[0][1]:
                    pass
                elif orient=='vert' and np.max(room_for_slide,axis=0)[0]==wall[0][0]:
                    pass
                elif orient=='vert' and np.min(room_for_slide,axis=0)[0]==wall[0][0]:
                    # левая стена
                    print(connected_walls)
                    if np.min(room_for_slide,axis=0)[1]==wall2[0][1]:
                        tag = 'min'
                        dist = wall2[0][1]-np.min(doors_pos_centers[connected_walls[0][0]][connected_walls[0][1]]['connect_room_points'],axis=0)[1]
                    else:
                        tag = 'max'
                        dist = wall2[0][1]-np.max(doors_pos_centers[connected_walls[0][0]][connected_walls[0][1]]['connect_room_points'],axis=0)[1]
                    print(f'расстояние между стенами - {dist}')
                    
                    if abs(dist)<tresh:
                        print('Расстояние меньше трещхолда двигаем стену')
                        if tag == 'min':
                            coord_to_change = np.min(doors_pos_centers[connected_walls[0][0]][connected_walls[0][1]]['connect_room_points'],axis=0)[1]
                        else:
                            coord_to_change = np.max(doors_pos_centers[connected_walls[0][0]][connected_walls[0][1]]['connect_room_points'],axis=0)[1]
                        
                        for idx,item in enumerate(doors_pos_centers[connected_walls[0][0]][connected_walls[0][1]]['connect_room_points']):
                            if int(item[1])==int(coord_to_change):
                                doors_pos_centers[connected_walls[0][0]][connected_walls[0][1]]['connect_room_points'][idx][1] = wall2[0][1]
                            

                # dist = 
            elif len(connected_walls2)==1:
                print('К след стене  прилегает одна комната')
                # определим положение стены
                if orient == 'vert':
                    if wall2[0][1] == np.min(room_for_slide,axis=0)[1]:
                        coord1 = np.min(doors_pos_centers[connected_walls[0][0]][connected_walls[0][1]]['connect_room_points'],axis=0)[1]
                        coord2 = np.min(doors_pos_centers[connected_walls2[0][0]][connected_walls2[0][1]]['connect_room_points'],axis=0)[1]
                        dist = coord1-coord2
                        print(f'расстояние между стенами - {dist}')
                        if abs(dist)<tresh:
                            doors_pos_centers = update_coords(doors_pos_centers, connected_walls, coord1,1, coord2 )
                elif orient == 'hor':
                    if wall2[0][0] == np.min(room_for_slide,axis=0)[0]:
                        coord1 = np.max(doors_pos_centers[connected_walls[0][0]][connected_walls[0][1]]['connect_room_points'],axis=0)[1]
                        coord2 = np.max(doors_pos_centers[connected_walls2[0][0]][connected_walls2[0][1]]['connect_room_points'],axis=0)[1]
                        dist = coord1-coord2
                        print(f'расстояние между стенами - {dist}')
                    elif wall2[0][0] == np.max(room_for_slide,axis=0)[0]:
                        coord1 = np.min(doors_pos_centers[connected_walls[0][0]][connected_walls[0][1]]['connect_room_points'],axis=0)[1]
                        coord2 = np.min(doors_pos_centers[connected_walls2[0][0]][connected_walls2[0][1]]['connect_room_points'],axis=0)[1]
                        dist = coord1-coord2
                        print(f'расстояние между стенами - {dist}')
                        if abs(dist)<tresh:
                            doors_pos_centers = update_coords(doors_pos_centers, connected_walls, coord1,1, coord2 )
          
            else:
                print('К след стене  прилегает неск комнат')
        else:
            print('К стене  прилегает неск комнат')
    return doors_pos_centers


import math
def distance(point1, point2):
    """
    Вычисляет евклидово расстояние между двумя точками.

    :param point1: tuple, первая точка (x1, y1)
    :param point2: tuple, вторая точка (x2, y2)
    :return: расстояние между точками
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_center(points):
    """
    Вычисляет центр многоугольника по заданным точкам.

    :param points: Список точек многоугольника.
    :return: Центр многоугольника.
    """
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    
    center_x = sum(x_coords) / len(points)
    center_y = sum(y_coords) / len(points)
    
    return [center_x, center_y]

def scale_position(points, final_points):
    """
    Масштабируем позиции относительно нового многоугольника.

    :param points: Исходные точки для масштабирования.
    :param final_points: Новые координаты вершин многоугольника.
    :return: Новые координаты точек.
    """
    center = calculate_center(final_points)
    scaled_points = []

    for point in points:
        new_x = center[0] + (point[0] - center[0]) * scale_factor
        new_y = center[1] + (point[1] - center[1]) * scale_factor
        scaled_points.append([new_x, new_y])

    return scaled_points

def scale_final_points(final_points, scale_factor):
    """
    Масштабирует final_points, исходя из заданного коэффициента масштабирования.

    :param final_points: Исходные координаты конечных точек.
    :param scale_factor: Коэффициент масштабирования.
    :return: Масштабированные координаты точек.
    """
    center = calculate_center(final_points)
    scaled_points = []

    for point in final_points:
        new_x = center[0] + (point[0] - center[0]) * scale_factor
        new_y = center[1] + (point[1] - center[1]) * scale_factor
        scaled_points.append([new_x, new_y])

    return scaled_points