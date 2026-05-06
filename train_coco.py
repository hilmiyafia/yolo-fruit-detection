
import torch
import lightning
from model import YOLO
from coco import get_dataset

class Trainer(lightning.LightningModule):

    def __init__(self, yolo):
        super().__init__()
        self.yolo = yolo
        self.bce = torch.nn.functional.binary_cross_entropy_with_logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.yolo.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 1/2)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_index):
        loss_box, loss_class = 0, 0
        output = self.yolo(batch[0])

        pos_count = sum([len(sample) for sample in batch[1]])
        pos_weight = (torch.numel(output[:, 4:]) - pos_count) / pos_count
        pos_weight = torch.tensor([pos_weight]).to(output.device)

        for i in range(len(output)):    
            pos = []
            count = len(batch[1][i])

            if count == 0:
                loss_class = loss_class + self.bce(
                    input=output[i, 4:], 
                    target=torch.zeros_like(output[i, 4:]))
                continue

            with torch.no_grad():
                bx1, by1, bx2, by2 = torch.unbind(batch[1][i], 1)
                bx = (bx1 + bx2) / 2
                by = (by1 + by2) / 2
                dx = (self.yolo.centers[:, 0] - bx[:, None, None]) ** 2 
                dy = (self.yolo.centers[:, 1] - by[:, None, None]) ** 2
                dp = dx + dy
                for j in range(count):
                    py, px = torch.unravel_index(torch.argmin(dp[j]), dp[j].shape)
                    py, px = py.item(), px.item()
                    dp[:, py, px] = torch.inf
                    pos.append((py, px))
                pos = torch.LongTensor(pos)

            ox1, oy1, ox2, oy2 = torch.unbind(output[i, :4, pos[:, 0], pos[:, 1]])
            ox = (ox1 + ox2) / 2
            oy = (oy1 + oy2) / 2

            # Box loss
            iw = torch.minimum(bx2, ox2) - torch.maximum(bx1, ox1)
            ih = torch.minimum(by2, oy2) - torch.maximum(by1, oy1)
            uw = torch.maximum(bx2, ox2) - torch.minimum(bx1, ox1)
            uh = torch.maximum(by2, oy2) - torch.minimum(by1, oy1)
            ia = iw.clamp(min=0) * ih.clamp(min=0)
            ba = (bx2 - bx1) * (by2 - by1)
            oa = (ox2 - ox1) * (oy2 - oy1)
            dc = ((ox - bx) ** 2 + (oy - by) ** 2) / (uw ** 2 + uh ** 2)
            iou = ia / (ba + oa - ia)
            loss_box = loss_box + (1 - iou + dc).sum()

            # Class loss
            target = torch.zeros_like(output[i, 4:])
            target[batch[2][i].flatten(), pos[:, 0], pos[:, 1]] = iou
            loss_class = loss_class + self.bce(
                input=output[i, 4:], 
                target=target,
                pos_weight=pos_weight)
        
        loss_box = loss_box / pos_count
        loss_class = loss_class / len(output)
        self.log("box", loss_box, True)
        self.log("class", loss_class, True)
        return loss_box + loss_class

JSON_PATH = r"../annotations/instances_train2014.json"
IMAGE_PATH = r"../train2014"
BATCH_SIZE = 8
EPOCHS = 10

if __name__ == "__main__":
    dataset, dataloader = get_dataset(JSON_PATH, IMAGE_PATH, BATCH_SIZE)
    yolo = YOLO(dataset.labels)
    trainer = lightning.Trainer(max_epochs=EPOCHS, precision="16-mixed")
    trainer.fit(model=Trainer(yolo), train_dataloaders=dataloader)
    torch.save(yolo.state_dict(), "checkpoint.pt")
