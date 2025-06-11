import numpy as np
from .utils import (get_angle_between_planes,
                    project_line_on_plane,
                    find_point_on_line,
                    get_angle_btw_lines,
                    make_rotate_stride,
                    detect_near_point,
                    get_normal,
                    get_cross_point)
from src.steps.visual_utils.viz_plan import mask_from_points

def change_len_wall(final_points,idx1,idx2, angle):
    point_A = tuple(int(item) for item in final_points[idx1])
    point_B = tuple(int(item) for item in final_points[idx2])
    angle = -1* angle  if angle<0 else angle
    
    original_length, projected_length = project_line_on_plane(point_A, point_B, angle)
    print(f"Длина оригинальной линии: {original_length:.2f}")
    print(f"Длина проекции линии на вторую плоскость: {abs(projected_length):.2f}")

    P1 = point_A
    P2 = point_B
    P = P1 # Заданная точка
    distance = projected_length  # Заданные расстояние

    point1, point2 = find_point_on_line(point_A, point_B, point_A, distance)
    new_point = detect_near_point(point1, point2, P2)

    return new_point


def angle_between_line_and_plane(line_direction, plane_normal):
    # Преобразуем входные данные в numpy массивы
    v_line = np.array(line_direction)
    v_normal = np.array(plane_normal[:2])
    
    # Вычисляем длины векторов
    norm_line = np.linalg.norm(v_line)
    norm_normal = np.linalg.norm(v_normal)
    
    # Вычисляем скалярное произведение векторов
    dot_product = np.dot(v_line, v_normal)
    
    # Находим косинус угла между вектором линии и нормалью плоскости
    cos_theta = dot_product / (norm_line * norm_normal)
    
    # Вычисляем угол в радианах
    angle_with_normal = np.arccos(cos_theta)
    
    # Угол между линией и плоскостью
    angle_with_plane = np.pi / 2 - angle_with_normal  # 90° - угол с нормалью
    
    # Преобразуем радианы в градусы
    angle_with_plane_degrees = np.degrees(angle_with_plane)
    
    return angle_with_plane_degrees


def make_plan_postprocess(final_points, order, target, planes):
    print()
    print('Построцессинг')
    # узнаем угол между плоскостями
    x_fixed_value = planes[target]['cross_point'][0][0]
    coefficients = [1,0,0,-x_fixed_value] # коэф плоскости

    N1 = coefficients[:-1]  
    N2 = planes[target]['floor'][:-1]

    # angle = get_angle_between_planes(N1, N2)
    # if len(order)!=3:
    #     angle = angle
    # print(f"Угол между плоскостями: {angle:.2f} градусов")

    direction_vector = np.array(final_points[order.index(target)]) - np.array(final_points[order.index(target)-1])
    angle = angle_between_line_and_plane(direction_vector,N2)
    if abs(angle)>1:
        final_points[order.index(target)-1] = change_len_wall(final_points, order.index(target),order.index(target)-1,angle)

    direction_vector = -1 * np.array(final_points[order.index(target)]) - np.array(final_points[order.index(target)+1])
    angle = angle_between_line_and_plane(direction_vector,N2)
    if abs(angle)>1:
        final_points[order.index(target)+1] = change_len_wall(final_points, order.index(target),order.index(target)+1,angle//2)
    

    point4,pointl,pointt,pointr = final_points

    ar = get_angle_btw_lines([pointr,pointt],[[0,0],[500,0]])
    al = get_angle_btw_lines([pointl,pointt],[[0,0],[500,0]])
    final_angle = np.min([ar,al])
    i=0
    while final_angle>1 and i<4:
        i+=1

        pointt = make_rotate_stride(final_angle,0,pointt)
        pointl = make_rotate_stride(final_angle,0,pointl)
        pointr = make_rotate_stride(final_angle,0,pointr)
        point4 = make_rotate_stride(final_angle,0,point4)

        ar = get_angle_btw_lines([pointr,pointt],[[0,0],[500,0]])
        al = get_angle_btw_lines([pointl,pointt],[[0,0],[500,0]])
        final_angle = np.min([ar,al])

    if len(order)!=3:
        l_normal = get_normal(np.array(pointl),np.array(pointt))
        r_normal = get_normal(np.array(pointr),np.array(pointt))
        point4 = get_cross_point(l_normal,r_normal)

    if len(order)==4:
        ar = get_angle_btw_lines([pointl,pointt],[pointt,pointr])
        if ar != 90:
            pointl = [pointt[0],point4[1]]
        
        ar = get_angle_btw_lines([pointl,pointt],[pointt,pointr])
        if ar != 90:
            pointr = [point4[0],pointt[1]]
    
    mask = mask_from_points([point4,pointl,pointt,pointr], order)

    # points = [point4,pointl,pointt,pointr]
    # final_points = {}
    # for i,item in enumerate(order):
    #     final_points[item] = points[i]

    return mask, [point4,pointl,pointt,pointr]