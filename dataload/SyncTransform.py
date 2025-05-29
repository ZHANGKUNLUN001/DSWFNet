from torchvision.transforms import functional as F
import random
import torchvision.transforms as transforms
from PIL import Image

class DeepSyncTransform:
    def __init__(self, crop_size=512):
        self.crop_size = crop_size

    def __call__(self, image, label):
        # 随机水平翻转
        if random.random() > 0.5:
            image = F.hflip(image)
            label = F.hflip(label)

        # 随机垂直翻转
        if random.random() > 0.5:
            image = F.vflip(image)
            label = F.vflip(label)

        # 随机旋转90/180/270度
        angle = random.choice([0, 90, 180, 270])
        image = F.rotate(image, angle, interpolation=F.InterpolationMode.BILINEAR, fill=0)
        label = F.rotate(label, angle, interpolation=F.InterpolationMode.NEAREST, fill=0)

        # 仿射增强（平移+缩放）
        if random.random() > 0.5:
            scale = random.uniform(0.95, 1.05)
            max_trans = 0.05 * min(image.size)
            translate = (
                random.uniform(-max_trans, max_trans),
                random.uniform(-max_trans, max_trans)
            )
            image = F.affine(image, angle=0, translate=translate, scale=scale, shear=0,
                             interpolation=F.InterpolationMode.BILINEAR, fill=0)
            label = F.affine(label, angle=0, translate=translate, scale=scale, shear=0,
                             interpolation=F.InterpolationMode.NEAREST, fill=0)

        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
        image = F.crop(image, i, j, h, w)
        label = F.crop(label, i, j, h, w)

        return image, label
