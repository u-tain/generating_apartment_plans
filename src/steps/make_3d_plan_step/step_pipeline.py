import os
from .build_planes import find_coeffs
from .postprocess import get_cross_points
class Make_3d_Plan_step:
    def __init__(self,flat_path, room):
        self.flat_path = flat_path
        self.room = room
        self.room_path = os.path.join(self.flat_path, 'by_rooms', self.room)
        self.planes = {}

    def run(self, viz=False):
        imgs = os.listdir(self.room_path)
        imgs = [item[:-4] for item in imgs if '.' in item]

        # points = os.listdir(os.path.join(self.room_path,'dump_match_pairs'))
        # points = [(int(item.split('_')[0]),int(item.split('_')[1])) for item in points if '.png' in item]
        # sorted_data = sorted(points, key=lambda x: (x[0], x[1]), reverse=True)

        # для каждой картинки получим плоскости
        for item in imgs:
            print(f'строим для фото {item}')
            paths = []
            paths = [os.path.join(self.room_path,'separeted_walls',item1) for item1 in os.listdir(os.path.join(self.room_path,'separeted_walls')) if f'{item}_' in item1]
            # for nw in range(num_walls[item]):
            #     paths.append(os.path.join(self.room_path,'separeted_walls',item+f'_wall_{nw+1}.png'))
            # добавляем пол
            paths.append(os.path.join(self.room_path,'segms',item+'_floor.png'))

            planes,walls_points = find_coeffs(paths,os.path.join(self.room_path,'depth',f'{item}_depth.npy'),viz=viz)

            # определяем порядок стен; удаляем лищние; находим точки пересечения
            num_planes = len(planes)-1
            walls_path = paths[:-1]
            # print(f'на фото {num_planes} стен')

            cross_points, planes = get_cross_points(walls_path, planes)


            self.planes[item] = {'cross_point': cross_points,
                                'has_door': len([item for item in  os.listdir(os.path.join(self.room_path,'segms')) if 'door' in item])>0,
                                'has_window': os.path.exists(os.path.join(self.room_path,'segms',item+'_window.png'))}
            
            for nw in range(len(planes)):

                if nw ==len(planes)-1:
                    self.planes[item][f'floor']  = planes[nw]
                    self.planes[item][f'floor_points']=walls_points[nw]
                else:
                    self.planes[item][f'wall{nw+1}']  = planes[nw]
                    self.planes[item][f'wall{nw+1}_points']=walls_points[nw]

        return self.planes
