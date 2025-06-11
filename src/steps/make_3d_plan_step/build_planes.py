import cv2 
import numpy as np
import plotly.graph_objects as go


def find_coeffs(walls_path,dpth_path, viz = False):
    def get_linsp(points):
        points = np.array(points)
        min_coords = np.min(points, axis=0) - 200
        max_coords = np.max(points, axis=0) + 200

        x = np.linspace(min_coords[0], max_coords[0], 100)
        y = np.linspace(min_coords[1], max_coords[1], 100)
        linsp = [x,y]
        return linsp

    nonzero_points = []
    w1 = []
    w1_points = []
    coefs = []
    nearest_points = []
    for plane in range(len(walls_path)):
        wall = cv2.cvtColor(cv2.imread(walls_path[plane]), cv2.COLOR_BGR2RGB)[:,:,-1]/255
        nonzero_point1, nearest_points1,specified_distance_1 = find_points(wall)
        nonzero_points.append(nonzero_point1)
        linsp = get_linsp(nonzero_points)

        w1x,w1y,w1z,w1_pointx,w1_pointy,w1_pointz,A1,B1,C1,D1 = make_plane(dpth_path,nonzero_point1, nearest_points1,linsp)
        counter = 0
        while (A1 == B1 or [A1,B1,C1,D1].count(0)>1) and counter<3:
            nonzero_point1, nearest_points1,_ = find_points(wall,specified_distance=specified_distance_1-5,spec_p=3)
            nonzero_points[-1]=nonzero_point1
            linsp = get_linsp(nonzero_points)
            w1x,w1y,w1z,w1_pointx,w1_pointy,w1_pointz,A1, B1, C1, D1= make_plane(dpth_path,nonzero_point1, nearest_points1,linsp)
            counter +=1
            specified_distance_1 -=5
  
        w1.append([w1x,w1y,w1z])
        w1_points.append([w1_pointx,w1_pointy,w1_pointz])
        coefs.append([A1, B1, C1, D1])
        nearest_points.append(nearest_points1)
    if viz:
        # Создание данных для плоскост

        linsp = get_linsp(nonzero_points)
        
        fig = go.Figure()
        colors = ['red','green','yellow','orange','red']*2 + ['blue']
        for plane in range(len(walls_path)):
            if plane == len(walls_path)-1:
                plane=-1
            w1x,w1y,w1z,w1_pointx,w1_pointy,w1_pointz,A1, B1, C1, D1= make_plane(dpth_path,nonzero_points[plane], nearest_points[plane],linsp)
            fig.add_trace(go.Surface(x=w1x, y=w1y, z=w1z))
            fig.add_trace(go.Scatter3d(x=w1_pointx, 
                                    y=w1_pointy, 
                                    z=w1_pointz, 
                                    mode='markers', 
                                    marker=dict(size=5, color=colors[plane], symbol='circle')))

        fig.update_layout(scene=dict(
                            xaxis_title='X',
                            yaxis_title='Y',
                            zaxis_title='Z'),
                            )

        # Отображение графика
        fig.show()
    
    return coefs, w1_points

def find_points(mask, specified_distance = 40,spec_p = 2):

    # Поиск ненулевой точки на маске
    y_indices, x_indices = np.where(mask == 1)  # y - строки, x - столбцы
    # Если есть хотя бы одна "единица" в маске:
    if len(x_indices) > 0 and len(y_indices) > 0:
        # Вычислить центроид
        x_center = int(np.mean(x_indices))
        y_center = int(np.mean(y_indices))
    nonzero_point = (y_center,x_center)   
    # Поиск двух ненулевых точек на заданном расстоянии от найденной точки
    nearest_points = []
    while len(nearest_points)!=2:
        specified_distance -= 10
        if specified_distance <= 0: 
            print('Не удалось найти подходящие точки')
            print('берем любые')
            nearest_points = [(x, y) for y, x in zip(x_indices, y_indices) if x != x_center or y != y_center]
            nearest_points = nearest_points[:2]
            break
        for point in np.transpose(np.nonzero(mask)):
            distance = np.linalg.norm(np.array(point) - np.array(nonzero_point))
            if distance == specified_distance and nonzero_point[1]!=point[1]:
                nearest_points.append(point)
            if len(nearest_points) == 2:
                break
        
    return nonzero_point, nearest_points,specified_distance

def make_plane(depth_path,nonzero_point, nearest_points,linsp):
    depth = np.load(depth_path)
    # Заданные точки
    point1 = [nonzero_point[0], nonzero_point[1],int(depth[nonzero_point[0], nonzero_point[1]])]
    point2 = [nearest_points[0][0],  nearest_points[0][1],int(depth[nearest_points[0][0],  nearest_points[0][1]])]
    point3 = [nearest_points[1][0],  nearest_points[1][1],int(depth[nearest_points[1][0],  nearest_points[1][1]])]

    # Найти уравнение плоскости
    # Векторы направления
    v1 = np.array(point2) - np.array(point1)
    v2 = np.array(point3) - np.array(point1)
    # Векторное произведение для нормали к плоскости
    normal = np.cross(v1, v2)
    d = -np.dot(normal, np.array(point1))
    # Уравнение плоскости: Ax + By + Cz + D = 0
    A, B, C = normal
    D = d
    # points = np.array([point1, point2, point3])
    # min_coords = np.min(points, axis=0) - 10
    # max_coords = np.max(points, axis=0) + 10

    # Создание данных для плоскост
    x = linsp[0]#np.linspace(min_coords[0], max_coords[0], 100)
    y = linsp[1]#np.linspace(min_coords[1], max_coords[1], 100)
    x, y = np.meshgrid(x, y)
    z = (-A * x - B * y - D) / C


    pointx = [point1[0], point2[0], point3[0]]
    pointy = [point1[1], point2[1], point3[1]]
    pointz = [point1[2], point2[2], point3[2]]
    return x,y,z, pointx,pointy,pointz,A,B,C,D