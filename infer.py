
import cv2
import torch
import torchvision
from model import YOLO

if __name__ == "__main__":
    labels = ["apple", "banana", "orange", "watermelon"]
    yolo = YOLO(labels).cuda()
    yolo.load_state_dict(torch.load("checkpoint.pt"))
    video = cv2.VideoCapture("video.mp4")
    yolo = yolo.eval()
    with torch.no_grad():
        while True:
            retval, image = video.read()
            if retval == False: break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.tensor(image, dtype=torch.uint8)
            image = image.permute((2, 0, 1))[None]
            output = yolo(image.cuda() / 255).cpu()[0]
            scores = torch.amax(output[4:], 0).sigmoid().flatten().numpy()
            indices = torch.argmax(output[4:], 0).flatten().numpy()
            labels = [yolo.labels[i] for i in indices]
            boxes = output[:4].flatten(1, 2).T
            indices = cv2.dnn.NMSBoxes(boxes.numpy(), scores, 0.5, 0.5)
            image = torch.nn.functional.interpolate(image, (512, 512))[0]
            if len(indices) > 0:
                image = torchvision.utils.draw_bounding_boxes(
                    image, 
                    boxes[indices] * 2,
                    labels=[labels[i] for i in indices],
                    colors="black", 
                    width=2,
                    font="arial.ttf",
                    font_size=16,
                    label_colors="white",
                    fill_labels=True)
            image = image.permute((1, 2, 0))
            image = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
            cv2.imshow("Frame", image)
            cv2.waitKey(20)
    video.release()
    cv2.waitKey()
