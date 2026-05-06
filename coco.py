import os
import torch
import torchvision
from pycocotools.coco import COCO

class Dataset(torch.utils.data.Dataset):

    def __init__(self, json_path, image_path):
        self.coco = COCO(json_path)
        self.image_path = image_path
        self.image_ids = list(self.coco.imgs.keys())
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_index = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        self.labels = [self.coco.cats[cat_id]["name"] for cat_id in cat_ids]

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_path, image_info["file_name"])
        image = torchvision.io.read_image(image_path) / 255
        if len(image.shape) == 2:
            image = torch.repeat_interleave(image[None], 3, 0)
        if image.shape[0] == 1:
            image = torch.repeat_interleave(image, 3, 0)
        _, height, width = image.shape
        pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
        ratio = 1
        if height != 256 or width != 256:
            if height > width:
                if height != 256:
                    ratio = 256 / height
                    new_width = int(256 * width / height)
                    image = torch.nn.functional.interpolate(
                        image[None], 
                        (256, new_width))[0]
                else:
                    new_width = width
                pad_left = (256 - new_width) // 2
                pad_right = 256 - pad_left - new_width
                image = torch.nn.functional.pad(
                    image, 
                    (pad_left, pad_right, 0, 0))
            else:
                if width != 256:
                    ratio = 256 / width
                    new_height = int(256 * height / width)
                    image = torch.nn.functional.interpolate(
                        image[None], 
                        (new_height, 256))[0]
                else:
                    new_height = height
                pad_top = (256 - new_height) // 2
                pad_bottom = 256 - pad_top - new_height
                image = torch.nn.functional.pad(
                    image, 
                    (0, 0, pad_top, pad_bottom))
        annotation_ids = self.coco.getAnnIds(image_id)
        annotations = self.coco.loadAnns(annotation_ids)
        boxes = []
        labels = []
        for annotation in annotations:
            if annotation.get("iscrowd", 0) == 1:
                continue
            x, y, w, h =  annotation["bbox"]
            x1 = x * ratio + pad_left
            y1 = y * ratio + pad_top
            x2 = x1 + w * ratio
            y2 = y1 + h * ratio
            boxes.append(torch.FloatTensor([x1, y1, x2, y2]))
            labels.append([self.cat_id_to_index[annotation["category_id"]]])
        return image, boxes, labels

def collate(batch):
    return (
        torch.stack([b[0] for b in batch]), 
        [torch.stack(b[1]) if len(b[1]) > 0 else [] for b in batch], 
        [torch.LongTensor(b[2]) for b in batch])

def get_dataset(json_path, image_path, batch_size=8):
    dataset = Dataset(json_path, image_path)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate, 
        num_workers=4, 
        persistent_workers=True)
    return dataset, dataloader

if __name__ == "__main__":
    json_path = r"../annotations/instances_train2014.json"
    image_path = r"../train2014"
    dataset = Dataset(json_path, image_path)
    print(dataset.labels)
    image, boxes, labels = dataset[670]
    image = torch.nn.functional.interpolate(image[None], (512, 512))[0]
    image = torchvision.utils.draw_bounding_boxes(
        image,
        torch.stack(boxes) * 2,
        [dataset.labels[l[0]] for l in labels],
        colors="black", 
        width=2,
        font="arial.ttf",
        font_size=16,
        label_colors="white",
        fill_labels=True)
    image =  (image * 255).clamp(min=0, max=255).to(torch.uint8).cpu()
    torchvision.io.write_jpeg(image, "test.jpg")