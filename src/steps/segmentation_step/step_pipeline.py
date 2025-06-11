import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .predict_utils import make_predict
from .postprocess import detect_door

class Segment_step:
    def __init__(self,flat_path, room):
        self.flat_path = flat_path
        self.room = room
        self.room_path = os.path.join(self.flat_path, 'by_rooms', self.room)
        self.segm_path = os.path.join(self.room_path, 'segms')
    
    def run(self,viz=False):
        # делаем предсказания для каждой картинке и сохраняем отдельно полы и стены
        os.makedirs(self.segm_path,exist_ok=True)
        imgs = os.listdir(self.room_path)
        imgs = [item for item in imgs if '.' in item]

        for item in imgs:
            img = cv2.imread(os.path.join(self.room_path,item))
            preds = make_predict(img)
            
            mask_floor = np.where(preds ==3,1,0)+np.where(preds ==28,1,0) # пол
            mask_wall = np.where(preds ==0,1,0) # +np.where(preds == 18,1,0) # стены
            mask_door = np.where(preds ==14,1,0) # дверь
            mask_window = np.where(preds ==8,1,0) if np.where(preds ==8,1,0).sum() !=0 else np.where(preds ==18,1,0) # окно

            mask_door,mask_floor,mask_wall = detect_door(mask_door,mask_floor,mask_wall,self.room_path,item)

            if not os.path.exists(os.path.join(self.segm_path,item[:-4]+'_floor.png')) and not os.path.exists(os.path.join(self.segm_path,item[:-4]+'_wall.png')):
                cv2.imwrite(os.path.join(self.segm_path,item[:-4]+'_floor.png'),mask_floor*255)
                cv2.imwrite(os.path.join(self.segm_path,item[:-4]+'_wall.png'),mask_wall*255)
    
                if mask_window.sum()>0:
                    if mask_window.sum()>0:
                        cv2.imwrite(os.path.join(self.segm_path,item[:-4]+'_window.png'),mask_window*255)
                
        if viz:
            num_images = len(imgs)*5
            cols = 5  # Количество столбцов для отображения
            if len(imgs)!=1:
                rows = (num_images // cols) + (num_images % cols > 0)  # Число строк

                # Создание фигуры для отображения изображений
                fig, axes = plt.subplots(rows, cols, figsize=(30, 5 * rows))

                # Плоский список осей для удобного перебора
                for i in range(len(imgs)):
                    axes[i][0].imshow(cv2.cvtColor(cv2.imread(os.path.join(self.room_path,imgs[i])), cv2.COLOR_BGR2RGB))
                    axes[i][1].imshow(cv2.cvtColor(cv2.imread(os.path.join(self.segm_path,imgs[i][:-4]+'_floor.png')), cv2.COLOR_BGR2RGB)/255)
                    axes[i][1].set_title('floor')
                    axes[i][2].imshow(cv2.cvtColor(cv2.imread(os.path.join(self.segm_path,imgs[i][:-4]+'_wall.png')), cv2.COLOR_BGR2RGB)/255)
                    axes[i][2].set_title('walls')
                    if os.path.exists(os.path.join(self.segm_path,imgs[i][:-4]+'_window.png')):
                        axes[i][3].imshow(cv2.cvtColor(cv2.imread(os.path.join(self.segm_path,imgs[i][:-4]+'_window.png')), cv2.COLOR_BGR2RGB)/255)
                        axes[i][3].set_title('window')
                    if os.path.exists(os.path.join(self.segm_path,imgs[i][:-4]+'_door.png')):
                        axes[i][4].imshow(cv2.cvtColor(cv2.imread(os.path.join(self.segm_path,imgs[i][:-4]+'_door.png')), cv2.COLOR_BGR2RGB)/255)
                        axes[i][4].set_title('door')
                    axes[i][0].axis('off') 
                    axes[i][1].axis('off') 
                    axes[i][2].axis('off') 
                    axes[i][3].axis('off') 
                    axes[i][4].axis('off') 
                    axes[i][0].set_title(os.path.basename(imgs[i]))  # Заголовок с названием файла

                axes = axes.flatten() if num_images > 1 else [axes]
                for ax in axes[num_images:]:
                    ax.remove()
            else:
                rows = 1
                fig, axes = plt.subplots(rows, cols, figsize=(30, 5 * rows))
                i=0

                axes[0].imshow(cv2.cvtColor(cv2.imread(os.path.join(self.room_path,imgs[i])), cv2.COLOR_BGR2RGB))
                axes[1].imshow(cv2.cvtColor(cv2.imread(os.path.join(self.segm_path,imgs[i][:-4]+'_floor.png')), cv2.COLOR_BGR2RGB)/255)
                axes[1].set_title('floor')
                axes[2].imshow(cv2.cvtColor(cv2.imread(os.path.join(self.segm_path,imgs[i][:-4]+'_wall.png')), cv2.COLOR_BGR2RGB)/255)
                axes[2].set_title('walls')
                if os.path.exists(os.path.join(self.segm_path,imgs[i][:-4]+'_window.png')):
                    axes[3].imshow(cv2.cvtColor(cv2.imread(os.path.join(self.segm_path,imgs[i][:-4]+'_window.png')), cv2.COLOR_BGR2RGB)/255)
                    axes[3].set_title('window')
                if os.path.exists(os.path.join(self.segm_path,imgs[i][:-4]+'_door.png')):
                    axes[4].imshow(cv2.cvtColor(cv2.imread(os.path.join(self.segm_path,imgs[i][:-4]+'_door.png')), cv2.COLOR_BGR2RGB)/255)
                    axes[4].set_title('door')
                axes[0].axis('off') 
                axes[1].axis('off') 
                axes[2].axis('off') 
                axes[3].axis('off') 
                axes[4].axis('off') 
                axes[0].set_title(os.path.basename(imgs[i]))  # Заголовок с названием файла

            plt.show()