import os
import cv2
import matplotlib.pyplot as plt
from .super_glue import Super_Glue

class Key_Points_step:
    def __init__(self,flat_path, room):
        self.flat_path = flat_path
        self.room = room
        self.room_path = os.path.join(self.flat_path, 'by_rooms', self.room)


    def run(self, viz=False):
        imgs = [item for item in os.listdir( self.room_path) if os.path.isfile(os.path.join( self.room_path, item))]
        if len(imgs)!= 1:
            # запуск модели
            m = Super_Glue(self.room_path)
            m.preprocess_stage()
            m.prediction_stage()
            m.postprocess_stage()

            # проверяем количество найденных матчей
            if len(os.listdir(os.path.join(self.room_path,'dump_match_pairs')))==1:
                 print('Не найдено матчей между фото')
            else:
                m.viz_result()

            if viz:
                if os.path.exists(os.path.join(self.room_path,'graphs','key_points.png')):
                    img = os.path.join(self.room_path,'graphs','key_points.png')
                    img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.show()
        else:
            print('Не достаточно фото для поиска ключевых точек')
