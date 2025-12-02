
import os
import numpy
import torch
import torchvision
import torchvision.transforms.v2

class Dataset(torch.utils.data.Dataset):

    def __init__(self, background_path, object_path):
        super().__init__()
        self.labels = []
        self.objects = []
        self.backgrounds = []
        for folder in os.listdir(object_path):
            self.labels.append(folder)
            self.objects.append([])
            folder_path = f"{object_path}/{folder}"
            for file in os.listdir(folder_path):
                self.objects[-1].append(f"{folder_path}/{file}")
        for file in os.listdir(background_path):
            self.backgrounds.append(f"{background_path}/{file}")
        self.augment_object = torch.nn.Sequential(
            torchvision.transforms.v2.ToDtype(torch.float32, True),
            torchvision.transforms.v2.RandomRotation(180),
            torchvision.transforms.v2.RandomPerspective(0.5, 1),
            torchvision.transforms.v2.RandomHorizontalFlip())
        self.augment_background = torch.nn.Sequential(
            torchvision.transforms.v2.ToDtype(torch.float32, True),
            torchvision.transforms.v2.ColorJitter(0.3, 0.1, 0.1, 0.05),
            torchvision.transforms.v2.RandomHorizontalFlip(),
            torchvision.transforms.v2.Resize((256, 256)))
        self.jitter = torchvision.transforms.v2.ColorJitter(0.3, 0.1, 0.1, 0.03)
        self.offsets = (
            (-32, -32), (64, -32), (160, -32),
            (-32, 64), (64, 64), (160, 64), 
            (-32, 160), (64, 160), (160, 160))

    def __len__(self):
        return 1000

    def __getitem__(self, _):
        boxes, indices = [], []
        background = numpy.random.choice(self.backgrounds, 1)[0]
        background = torchvision.io.read_image(background)
        background = self.augment_background(background)
        background = torch.nn.functional.pad(background, (32, 32, 32, 32))
        for ox, oy in self.offsets:
            if numpy.random.rand() > 0.5: continue
            i = numpy.random.randint(len(self.objects))
            j = numpy.random.randint(len(self.objects[i]))
            s = numpy.random.randint(64, 128)
            x = ox + numpy.random.randint(128 - s) + 32
            y = oy + numpy.random.randint(128 - s) + 32
            object = torchvision.io.read_image(self.objects[i][j])
            object = torch.stack((object[:3], object[3].repeat((3, 1, 1))))
            object = self.augment_object(object)
            object = torchvision.transforms.v2.functional.resize(object, (s, s))
            alpha = torch.where(object[1] > 0.5, 1, 0)
            object = self.jitter(object[0])
            temp = background[:, y:y + s, x:x + s]
            background[:, y:y + s, x:x + s] = object * alpha + temp * (1 - alpha)
            buffer = torch.zeros_like(background)
            buffer[:, y:y + s, x:x + s] = alpha
            buffer = buffer[:, 32:-32, 32:-32]
            if torch.sum(buffer) < 1: continue
            box = torchvision.ops.masks_to_boxes(buffer)[0]
            boxes.append(box)
            indices.append([i])
        return background[:, 32:-32, 32:-32], boxes, indices

def collate(batch):
    return (
        torch.stack([b[0] for b in batch]), 
        [torch.stack(b[1]) if len(b[1]) > 0 else [] for b in batch], 
        [torch.LongTensor(b[2]) for b in batch])

def get_dataset(background_path, image_path, batch_size=8):
    dataset = Dataset(background_path, image_path)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        collate_fn=collate, 
        num_workers=4, 
        persistent_workers=True)
    return dataset, dataloader
