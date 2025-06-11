import cv2
import numpy as np
from sklearn.cluster import KMeans

ker=np.ones((10,5))
def get_instances(nor_img,mask_path,kernel):
    # Загрузка изображения
    image = nor_img*255 # Загружаем изображение в оттенках серого
    gray_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # Применение фильтрации для удаления шума
    # gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # ищем контуры и утолщаем
    edges = cv2.Canny(gray_image, 30, 130)  
    
    # fatedge=edges
    print(kernel.shape)
    fatedge=cv2.dilate(edges, kernel)
    # fatedge = cv2.erode(fatedge, (item//2 for item in kernel), iterations=1)

    # Отображение контуров
    n,comp=cv2.connectedComponents((fatedge==0).astype(np.uint8))
 
    mask_path = cv2.imread(mask_path)[:,:,-1]
    # comp = cv2.dilate(comp, ker)
    comp *= mask_path
    
    return comp

def make_clastering(normal,wall):
    normal = cv2.GaussianBlur(normal, (25, 25), 0)
    h, w, _ = normal.shape
    normal_reshaped = normal.reshape((-1, 3))  # Преобразуем в 2D
    kmeans = KMeans(n_clusters=4)  # Задайте количество плоскостей
    kmeans.fit(normal_reshaped)

    # Получаем метки для каждого пикселя
    labels = kmeans.labels_.reshape((h, w))

    # Шаг 3: Создание масок для каждой плоскости
    masks = []
    for i in range(4):  # Для каждой плоскости
        mask = (labels == i).astype(np.uint8) * 255  # 0 или 255
        mask = wall*mask
        if mask.sum()>500:
            masks.append(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    min_area = 1000
    for i,mask in enumerate(masks):
        # Удаление мелких объектов
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Удаление шумов
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)  # Заполнение пробелов

        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask_cleaned)

        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(clean_mask, [contour], -1, (255), thickness=cv2.FILLED)

        masks[i] = clean_mask
    masks = [item for item in masks if item.sum()!=0]
    return masks

def separate_masks(wall,masks):
    all_masks = []
    for item in masks:
        contours, _ = cv2.findContours(item, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Создение масок и вычисление центров
        for contour in contours:
            # Создаем отдельную маску для текущего объекта
            obj_mask = np.zeros_like(wall)
            cv2.drawContours(obj_mask, [contour], -1, 255, -1)  # Рисуем контур
            all_masks.append(obj_mask)
    return all_masks


def get_separated_walls(wide_contours,walls,num_walls):
    results = []
    walls = cv2.cvtColor(cv2.imread(walls),cv2.COLOR_BGR2RGB)[:,:,-1]
    max_values = []

    result = wide_contours * walls
    result = result[10:-10,10:-10]
    unique_values, counts = np.unique(result, return_counts=True)
    res = dict(zip(unique_values[1:], counts[1:]))
    vl = res.values()
    vl = list(vl)

    for item in range(num_walls):
        if len(vl)!=0:
            max_values.append(max(vl))
            vl.remove(max(vl))

    max_keys = [key for key, value in res.items() if value in  max_values]
    for item in range(len(max_values)):
        res = np.where(result==max_keys[item],255,0)
        res = cv2.dilate(res.astype(np.uint8), ker)//255
        results.append(np.where(result==max_keys[item],1,0))
    return results