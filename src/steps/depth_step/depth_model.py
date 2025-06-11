import os 
import cv2
import torch 
import numpy as np
from PIL import Image
from dataclasses import dataclass, field

from .depth_utils.dpt import DepthAnything
from .depth_utils.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose


@dataclass
class DepthMap:
    # класс для предсказания карты глубины
    flat_path:str
    # model = field(init=False)

    def __post_init__(self):
        os.makedirs(os.path.join(self.flat_path,'depth'), exist_ok=True)
        self.model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14")
        # self.model = torch.load('depth_model.pt')

    def preprocess(self, name):
        transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        transform2 = Compose([Resize(640,480)])

        image = Image.open(os.path.join(self.flat_path, name)).convert('RGB')
        image = np.array(image) / 255.0
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        return image

    def start_predict(self):
        imgs = os.listdir(self.flat_path)
        imgs = [item for item in imgs if os.path.isfile(os.path.join( self.flat_path, item))]
        
        for item in imgs:
            img = self.preprocess(item)
            depth = self.model(img)
            pred = self.postprocess(depth)
            np.save(os.path.join(self.flat_path,'depth',item[:-4]+'_depth.npy'),pred[0][0])
    
    def postprocess(self,img):
        prediction = torch.nn.functional.interpolate(
        img.unsqueeze(1),
        size=(480,640),
        mode="bicubic",
        align_corners=False,
    )
        
        prediction = prediction.detach().numpy()
        return prediction

