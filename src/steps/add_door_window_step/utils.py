import math
import numpy as np

def get_distance(box, point):
    bottom_points = sorted(box, key=lambda x: x[1], reverse=True)[:2]
    distance = np.linalg.norm(np.array(bottom_points[0]) - np.array(bottom_points[1]))

    distances_to_third_point = [np.linalg.norm(np.array(pt) - np.array(point)) for pt in bottom_points]
    closest_point_index = np.argmin(distances_to_third_point)
    closest_distance = distances_to_third_point[closest_point_index]
    return distance, closest_distance

def get_object_points(point1, point2, door_size, dist_from_point):
    # Преобразуем входные точки в numpy массивы
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    # Вычисление вектора направления
    direction_vector = point2 - point1
    vector_length = np.linalg.norm(direction_vector)

    # Нормализация вектора и получение единичного вектора
    if vector_length > 0:
        unit_vector = direction_vector / vector_length
    else:
        raise ValueError("Две точки совпадают.")
    
    # Вычисление смещенной точки
    offset_point = point1 + dist_from_point * unit_vector
    
    # Новая точка на заданном расстоянии от первой точки
    new_point_on_line = offset_point + door_size * unit_vector

    # Проверка, что новая точка находится в пределах point1 и point2
    x_diff = np.abs(point2[0] - point1[0])
    y_diff = np.abs(point2[1] - point1[1])
    if x_diff > y_diff:
        if not (point1[0] <= new_point_on_line[0] <= point2[0] or point2[0] <= new_point_on_line[0] <= point1[0]):
            # Если точка вне пределов по x, берем центр и вычисляем новую точку
            center_point = (point1 + point2) / 2
            new_point_on_line = center_point + (door_size / 2) * unit_vector
            offset_point = center_point - (door_size / 2) * unit_vector
    else:
        if not (point1[1] <= new_point_on_line[1] <= point2[1] or point2[1] <= new_point_on_line[1] <= point1[1]):
            # Если точка вне пределов по y, берем центр и вычисляем новую точку
            center_point = (point1 + point2) / 2
            new_point_on_line = center_point + (door_size / 2) * unit_vector
            offset_point = center_point - (door_size / 2) * unit_vector


    return offset_point, new_point_on_line

def is_point_in_rectangle(point, A, B, C, D):
    """
    Проверяет, находится ли заданная точка внутри прямоугольника, заданного четырьмя точками.

    :param point: координаты точки (px, py)
    :param A: координаты первой точки (x1, y1)
    :param B: координаты второй точки (x2, y2)
    :param C: координаты третьей точки (x3, y3)
    :param D: координаты четвертой точки (x4, y4)
    :return: True, если точка находится внутри или на границе, иначе False
    """
    def vector_area(v1, v2):
        """Вычисляет площадь (векторное произведение) между двумя векторами"""
        return v1[0] * v2[1] - v1[1] * v2[0]

    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)
    P = np.array(point)

    # Векторы
    AB = B - A
    AP = P - A

    BC = C - B
    BP = P - B

    CD = D - C
    CP = P - C

    DA = A - D
    DP = P - D

    # Проверяем, находится ли точка P по одну сторону от всех сторон прямоугольника
    return (vector_area(AB, AP) >= 0 and  # Переход из A в B
            vector_area(BC, BP) >= 0 and  # Переход из B в C
            vector_area(CD, CP) >= 0 and  # Переход из C в D
            vector_area(DA, DP) >= 0)  

def calculate_distance(point1, point2):
    """
    Вычисляет расстояние между двумя точками в 2D пространстве.

    :param point1: (x1, y1) - координаты первой точки.
    :param point2 (x2, y2) - координаты второй точки.
    :return: Расстояние между точками.
    """
    x1, y1 = point1
    x2, y2 = point2

    # Вычисление расстояния по формуле
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance