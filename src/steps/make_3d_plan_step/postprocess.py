import numpy as np 
import cv2

def get_point(planes):
    plane1 = planes[0]  
    plane2 = planes[1]
    plane3 = planes[-1]
    # Создание матрицы коэффициентов A для системы уравнений
    A = np.array([plane1[:3], plane2[:3], plane3[:3]])
    det_A = np.linalg.det(A)

    # Создание матрицы для каждого столбца свободных членов
    A_x = np.array([[-plane1[3], plane1[1], plane1[2]], [-plane2[3], plane2[1], plane2[2]], [-plane3[3], plane3[1], plane3[2]]])
    A_y = np.array([[plane1[0], -plane1[3], plane1[2]], [plane2[0], -plane2[3], plane2[2]], [plane3[0], -plane3[3], plane3[2]]])
    A_z = np.array([[plane1[0], plane1[1], -plane1[3]], [plane2[0], plane2[1], -plane2[3]], [plane3[0], plane3[1], -plane3[3]]])

    # Нахождение определителей каждой модифицированной матрицы
    det_Ax = np.linalg.det(A_x)
    det_Ay = np.linalg.det(A_y)
    det_Az = np.linalg.det(A_z)

    # Нахождение значений x, y, z
    intersection_point = [det_Ax/det_A, det_Ay/det_A, det_Az/det_A]
    return intersection_point

def get_center_of_mass(mask):
    """Возвращает координаты центра масс для маски."""
    # Находим контуры маски
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centers = []
    for cnt in contours:
        # Находим момент (moment) для вычисления центра масс
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            centers.append((cX, cY))
    
    return centers

def sort_masks_by_position(mask_files):
    mask_centers = []

    for mask_file in mask_files:
        # Загружаем и обрабатываем маску
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        
        # Получаем центр масс
        centers = get_center_of_mass(mask)
        
        # Добавляем в список центров масс
        if centers:
            avg_center = np.mean(centers, axis=0)
            mask_centers.append((mask_file, avg_center[0]))
    
    # Сортируем списки масок по X координате центра масс
    sorted_masks = sorted(mask_centers, key=lambda x: x[1])
    
    return sorted_masks

def angle_between_planes(normal1, normal2):
    """Определяет угол между двумя плоскостями, заданными их нормалями."""
    # Нормализация нормалей
    n1 = normal1 / np.linalg.norm(normal1)
    n2 = normal2 / np.linalg.norm(normal2)

    # Вычисление косинуса угла
    cos_theta = np.dot(n1, n2)

    # Избегаем числовых ошибок, если cos_theta выходит за пределы [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Вычисление угла в радианах
    angle_rad = np.arccos(cos_theta)

    # Конвертация в градусы
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def get_cross_points(walls_path, planes):
    sorted_masks = sort_masks_by_position(walls_path)
    sorted_idx = [walls_path.index(item[0]) for item in sorted_masks]
    sorted_planes = [planes[item] for item in sorted_idx]
    angles = []
    cross_points = []
    final_planes = []

    # вычисляем углы и удаляем лишние плоскости
    for i in range(len(sorted_planes)-1):
        angle = angle_between_planes(sorted_planes[i][:-1],sorted_planes[i+1][:-1])
        if angle>10 and angle<167:
            point = get_point([sorted_planes[i],sorted_planes[i+1],planes[-1]])
            cross_points.append(point)
            if len(final_planes)>0:
                if final_planes[-1] != sorted_planes[i]:
                    final_planes.append(sorted_planes[i])
            else:
                final_planes.append(sorted_planes[i])
            final_planes.append(sorted_planes[i+1])

    final_planes.append(planes[-1])
    print('Всего получили ',len(cross_points),' точек пересечения')


    return cross_points, final_planes




