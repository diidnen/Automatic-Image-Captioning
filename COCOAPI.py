import os
import sys
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

# 导入 COCO API
sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO

# 设置数据路径
data_dir = 'data/coco'  # 使用您的数据目录路径
dataType = 'train2014'

# 初始化 COCO API 用于实例标注
instances_annFile = os.path.join(data_dir, f'annotations/instances_{dataType}.json')
coco = COCO(instances_annFile)

# 初始化 COCO API 用于标题标注
captions_annFile = os.path.join(data_dir, f'annotations/captions_{dataType}.json')
coco_caps = COCO(captions_annFile)

# 获取图像ID
ids = list(coco.anns.keys())

# 随机选择一个图像并获取对应的URL
ann_id = np.random.choice(ids)
img_id = coco.anns[ann_id]['image_id']
img = coco.loadImgs(img_id)[0]

# 尝试从本地加载图像，如果不可用则使用URL
img_path = os.path.join(data_dir, dataType, img['file_name'])
if os.path.exists(img_path):
    # 使用本地图像
    I = io.imread(img_path)
    print(f"从本地加载图像: {img_path}")
else:
    # 使用URL作为备选
    url = img['coco_url']
    print(f"本地未找到图像，从URL加载: {url}")
    I = io.imread(url)

# 显示图像
plt.figure(figsize=(8, 8))
plt.axis('off')
plt.imshow(I)
plt.show()

# 加载并显示标题
annIds = coco_caps.getAnnIds(imgIds=img['id'])
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
