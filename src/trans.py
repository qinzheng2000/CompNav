import cv2
import numpy as np
import random
import torch
import torchvision.transforms as transforms
import numpy as np
import time

def rotate_and_crop(images):
    # 生成一个随机旋转角度
    # angle = np.random.uniform(-90, 90)
    time1 = time.time()
    angle = 45
    imagegoal_sensor_v2 = []
    for i in range(images['imagegoal_sensor_v2'].size(0)):
        demo = images['imagegoal_sensor_v2'][i].clone()
        # demo.cuda()
        image = demo.cpu().numpy()
        original_h, original_w = image.shape[:2]

        # 计算旋转中心和旋转矩阵
        center = (original_w / 2, original_h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 执行旋转
        rotated = cv2.warpAffine(image, M, (original_w, original_h))

        # 裁剪图像到原始尺寸
        x = (rotated.shape[1] - original_w) // 2
        y = (rotated.shape[0] - original_h) // 2
        cropped = rotated[y:y+original_h, x:x+original_w]

        imagegoal_sensor_v2.append(cropped)

    # a = np.array(imagegoal_sensor_v2)
    images['imagegoal_sensor_v2'] = torch.from_numpy(np.array(imagegoal_sensor_v2)).cuda()
    time2 = time.time()
    timeall = time2 - time1
    # print("2",timeall)
    return images