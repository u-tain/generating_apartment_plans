import numpy as np
import cv2
import os
from .utils import get_angle_stride, make_rotate_stride, get_angle_btw_lines, get_H,sort_points,get_normal,get_cross_point
from src.steps.visual_utils.viz_plan import mask_from_points


def make_room_plan(path, order, target, points):
    """
    Метод строит план комнаты по найденным точкам пересечения плоскостей и с помощью матрицы гомографии
    path - путь до комнаты
    order - порядок выстроения фотографий для склеивания (на основе матч поинтов)
    target - картинка в чью ск переводим (должно быть не крайнее фото для удобства)

    return 
    mask - маска с результатом
    points - координаты углов комнаты
    """
    if len(order) == 4:

        # берем соседние фото
        i = order.index(target)
        l = order[i-1]
        r = order[i+1]

        # получаем матрицы гомографий
        Hl = get_H(l,target,path)
        Hr = get_H(r,target,path)

        # получаем повороты и смещения
        angle_l, stride_l = get_angle_stride(Hl)
        angle_r, stride_r = get_angle_stride(Hr)

        #получаем координаты точек
        pointl = points[l][0]
        pointr = points[r][0]
        pointt = points[target][0]

        pointl = make_rotate_stride(angle_l,stride_l,pointl)
        pointr = make_rotate_stride(angle_r,stride_r,pointr)

        # # # если получили угол не 90 градусов
        # a = get_angle_btw_lines([pointr,pointt],[pointt,pointl])
        # k = 0
        # while a > 100 and k<5:
        #     k+=1
        #     print(f'угол между линиями {a}')
        #     if a>0 and a>91:
        #         print(a)
        #         a = (90 - (180 - a))*2
        #         print(a)
        #         if angle_l<angle_r:
        #             pointr = make_rotate_stride(a,0,pointr)
        #         else:
        #             pointl = make_rotate_stride(a,0,pointl)
        #         a = get_angle_btw_lines([pointl,pointt],[pointt,pointr])
        #     print(get_angle_btw_lines([pointl,pointt],[pointt,pointr]))

        # # находим последнюю точку
        # l_normal = get_normal(np.array(pointl),np.array(pointt))
        # r_normal = get_normal(np.array(pointr),np.array(pointt))
        # point4 = get_cross_point(l_normal,r_normal)

        point4 = points[order[0]][0]
        # print()
        # print('Последняя картинка')
        if  os.path.exists(os.path.join(path,'dump_match_pairs', f"{order[0]}_{order[i-1]}_matches.npz")):
            H4 = get_H(order[0], order[i-1], path)
            angle_4, stride_4 = get_angle_stride(H4)
            # print('last',angle_l/2+angle_4/2,stride_l/2+stride_4/2)
            point4 = make_rotate_stride(angle_4,-stride_4,point4)
            point4 = make_rotate_stride(angle_l,stride_l,point4)
        else:
            l_normal = get_normal(np.array(pointt),np.array(pointl))
            r_normal = get_normal(np.array(pointt),np.array(pointr))
            point4 = get_cross_point(l_normal,r_normal)
        # point4 = get_cross_point(l_normal,r_normal)

        # делаем финальный поворот
        # ar = get_angle_btw_lines([pointr,pointt],[[0,0],[500,0]])
        # al = get_angle_btw_lines([pointl,pointt],[[0,0],[500,0]])
        # final_angle = np.min([ar,al])

        # pointt = make_rotate_stride(final_angle,0,pointt)
        # pointl = make_rotate_stride(final_angle,0,pointl)
        # pointr = make_rotate_stride(final_angle,0,pointr)
        # point4 = make_rotate_stride(final_angle,0,point4)
    else:
        # print()
        # target = '4'
        i = order.index(target)
        cross_points = []
        for j,item in enumerate(order):
            if i!= j:
                Hm = get_H(item,target,path)
                angle, stride = get_angle_stride(Hm)
                point = points[item][0]
                point = make_rotate_stride(angle,stride,point)

                if i < j:
                    a = get_angle_btw_lines([point,points[target][1]],[points[target][1],points[target][0]])
                else:
                    a = get_angle_btw_lines([points[target][1],points[target][0]],[points[target][0],point])

                print(f'угол между линиями {a}')
                if a>0 and (a>91 or a<70):
                    print(a)
                    a = (90 - (180 - a))*2
                    print(a)
                    point = make_rotate_stride(a,0,point)

                cross_points.append(point)
            else:
                cross_points+=points[target]


        point4,pointl,pointt,pointr = cross_points
        # point4,pointl,pointt,pointr = sort_points(cross_points)
        #     a = get_angle_btw_lines([pointr,pointt],[pointt,pointl])
        #     print(f'угол между линиями {a}')
        #     if a>0 and a>91:
        #         print(a)
        #         a = (90 - (180 - a))*2
        #         print(a)
        #         pointl = make_rotate_stride(a,0,pointl)


    mask = mask_from_points([point4,pointl,pointt,pointr], order)

    return mask, [point4,pointl,pointt,pointr]
