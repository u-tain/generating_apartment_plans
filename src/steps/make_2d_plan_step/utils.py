import numpy as np
import cv2
import os
import math
from src.steps.key_points_step.points_utils import loadNPZ


def get_angle_stride(H):
    angle_rad = - np.arctan2(H[0, 1], H[0, 0])
    angle_deg = np.degrees(angle_rad)

    # print(f"Угол поворота: {angle_deg:.2f} градусов")

    t = np.array([H[0, 2],H[1, 2]])
    # print(f"Вектор смещения: {t}")
    return angle_deg, t


def make_rotate_stride(angle,stride,point):
    theta = np.radians(angle)
    # делаем  со смещением
    if angle<10 and angle>-10:
        point = np.array([point[0] * np.cos(theta) - point[1] * np.sin(theta),point[0] * np.sin(theta) + point[1] * np.cos(theta)]) + stride
    # делаем только поворот
    elif angle<-110 or angle>100:
        theta =  int(theta)//2 #np.radians(90)
        point = np.array([point[0] * np.cos(theta) - point[1] * np.sin(theta),point[0] * np.sin(theta) + point[1] * np.cos(theta)])
    else:
        point = np.array([point[0] * np.cos(theta) - point[1] * np.sin(theta),point[0] * np.sin(theta) + point[1] * np.cos(theta)]) 
    return point


def get_angle_btw_lines(line1,line2):
    # Определяем координаты точек
    p1,p2 = np.array(line1[0]),np.array(line1[1])
    p3,p4 = np.array(line2[0]),np.array(line2[1])

    # Направляющие векторы
    vector_a = p2 - p1  # Вектор для линии 1
    vector_b = p4 - p3  # Вектор для линии 2

    # Вычисляем скалярное произведение и длины векторов
    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    # Вычисляем косинус угла
    cos_theta = dot_product / (magnitude_a * magnitude_b)

    # Вычисляем угол в радианах и затем в градусах
    theta = np.arccos(cos_theta)
    theta_degrees = np.degrees(theta)

    return int(theta_degrees)

def get_H(img1, img2, path):
    npz = f"{img1}_{img2}_matches.npz"
    if not os.path.exists(os.path.join(path,'dump_match_pairs', npz)):
        npz = f"{img2}_{img1}_matches.npz"
    point_set1, point_set2 = loadNPZ(npz, path)
    H, status = cv2.findHomography(point_set1, point_set2, cv2.RANSAC, 7.0) 
    return H

def get_angle_between_planes(n1,n2):
    dot_product = np.dot(n1, n2)

    # Длины векторов
    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)

    # Вычисление косинуса угла
    cos_theta = dot_product / (norm_n1 * norm_n2)

    # Вычисление угла в радианах и затем в градусах
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def get_original_lenght(point1, point2):
    # Распаковка точек
    x1, y1 = point1
    x2, y2= point2
    
    # Вычисление длины оригинальной линии
    L = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 )
    return L

def project_line_on_plane(point1, point2, alpha):
    L = get_original_lenght(point1, point2)

    # Перевод угла в радианы
    alpha_rad = math.radians(alpha)

    # Вычисление длины проекции
    L_prime = L * math.sin(alpha_rad)
    
    return L, L_prime

def find_point_on_line(P1, P2, P, distance):
    # Извлечение координат
    x1, y1 = P1
    x2, y2 = P2
    x0, y0 = P

    # Вычисление вектора направления
    d = np.array([x2 - x1, y2 - y1])
    
    # Нормализация вектора направления
    length = np.linalg.norm(d)
    d_hat = d / length

    # Найдите точки на заданном расстоянии
    point1 = np.array([x0, y0]) + distance * d_hat
    point2 = np.array([x0, y0]) - distance * d_hat

    return point1, point2

def detect_near_point(point1,point2,point3):
    # Вычисляем расстояния
    distance1 = np.linalg.norm(point1 - point3)
    distance2 = np.linalg.norm(point2 - point3)

    # Определяем ближайшую точку
    if distance1 < distance2:
        nearest_point = point1
        nearest_distance = distance1
    else:
        nearest_point = point2
        nearest_distance = distance2

    return nearest_point

def get_normal(point0,point1,step = 0):

    slope = (point1[1] - point0[1]) / (point1[0] - point0[0])  
    slope_perpendicular = -1 / slope

    b = point0[1] - slope_perpendicular * point0[0]

    x_vals_perpendicular = np.array([point0[0], point0[0]-500])  # два значения x для построения линии
    y_vals_perpendicular = slope_perpendicular * x_vals_perpendicular + b
    return [np.array([point0[0], y_vals_perpendicular[0]]),[point0[0]-500,y_vals_perpendicular[1]]]


def get_cross_point(line1,line2):
    A = np.array(line1[0])  # Точка A (x1, y1)
    B = np.array(line1[1])  # Точка B (x2, y2)

    C = np.array(line2[0])  # Точка C (x3, y3)
    D = np.array(line2[1])  # Точка D (x4, y4)

    # Функция для нахождения угловых коэффициентов и свободных членов
    def line_equation(point1, point2):
        if point1[0] == point2[0]:  # Угловой коэффициент = бесконечность, вертикальная линия
            slope = np.inf
            intercept = point1[0]  # x = intercept
        else:
            slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
            intercept = point1[1] - slope * point1[0]
        return slope, intercept

    # Получаем угловые коэффициенты и пересечения
    m1, b1 = line_equation(A, B)  # Для первой линии
    m2, b2 = line_equation(C, D)  # Для второй линии

    # Решаем уравнения для нахождения точки пересечения
    if m1 == m2:
        print("Линии параллельны и не имеют точки пересечения.")
        return None
    else:
        if m1 == np.inf:  # Первая линия вертикальная
            x = b1  # x-позиция пересечения по первой линии
            y = m2 * x + b2  # Находим y по второй линии
        elif m2 == np.inf:  # Вторая линия вертикальная
            x = b2  # x-позиция пересечения по второй линии
            y = m1 * x + b1  # Находим y по первой линии
        else:
            # Находим x и затем y
            x = (b2 - b1) / (m1 - m2)
            y = m1 * x + b1

    return [x,y]

def calculate_angle(center, point):
    delta_y = point[1] - center[1]
    delta_x = point[0] - center[0]
    return math.atan2(delta_y, delta_x)  # Угол в радианах

# Функция для сортировки точек
def sort_points(points):
    # Вычисляем центроид
    center = np.mean(points, axis=0)
    
    # Сортируем точки по углу относительно центроида
    sorted_points = sorted(points, key=lambda p: calculate_angle(center, p))
    
    return sorted_points