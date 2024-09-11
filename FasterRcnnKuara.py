import os

import numpy as np
import torch
import torchvision
import engine
import coco_eval
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision.io import read_image
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision import tv_tensors
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
from PIL import Image
from coco_eval import CocoEvaluator

class KuaraCOCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = read_image(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = img.float() / 255.0
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)

# Modeli Tanımlama Fonksiyonu
def get_model_fasterRcnn(num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT", pretrained=False)
    num_classes = num_classes

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )
    return model

root_train =  'Kuara/train'
ann_file_train = 'Kuara/train_annotations.coco.json'

root_test = 'Kuara/test'
ann_file_test = 'Kuara/test_annotations.coco.json'

myTransforms = T.Compose([
    T.Resize((256, 256)),  # Görüntüyü sabit bir boyuta getirmek için
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize etme
])

def myCollate(batch):
    images, targets = zip(*batch)
    # images = torch.stack([t[0] for t in batch])

    # Target'ları düzenleyelim
    targets = [{k: torch.tensor(v) for k, v in t[1].items()} for t in batch]

    return images, targets

train_dataset = KuaraCOCODataset(root=root_train, annotation=ann_file_train, transforms=myTransforms)
test_dataset = KuaraCOCODataset(root=root_test, annotation=ann_file_test, transforms=myTransforms)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=utils.collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = get_model_fasterRcnn(num_classes=3)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epoch = 10
for epoch in range(num_epoch):
    train_one_epoch(model, optimizer, train_loader, device, epoch, 2)
    lr_scheduler.step()
    evaluate(model, test_loader, device=device)

print("Egitim tamamlandi!")




