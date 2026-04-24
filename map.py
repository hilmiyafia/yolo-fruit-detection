import cv2
import torch
import numpy
from model import YOLO
from utils import Dataset

class Generator:

    def __init__(self, background_path, image_path):
        self.dataset = Dataset(background_path, image_path)
        self.index = 0

    def get(self):
        data = self.dataset[self.index]
        self.index = (self.index + 1) % len(self.dataset)
        image = data[0]
        boxes = numpy.array(data[1])
        indices = numpy.array(data[2]).flatten()
        return image, boxes, indices

class Predictor:

    def __init__(self):
        labels = ["apple", "banana", "orange", "watermelon"]
        self.yolo = YOLO(labels).cuda()
        self.yolo.load_state_dict(torch.load("checkpoint.pt"))
        self.yolo.eval()

    def predict(self, image):
        with torch.no_grad():
            outputs = self.yolo(image[None].cuda())[0].cpu()
        classes = torch.argmax(outputs[4:], 0).flatten().numpy()
        boxes = outputs[:4].flatten(1, 2).T.numpy()
        scores = torch.amax(outputs[4:], 0).sigmoid().flatten().numpy()
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.5)
        if len(indices) == 0:
            return [], [], []
        return boxes[indices], classes[indices], scores[indices]

def calc_iou(pred, truth):
    bx1, by1, bx2, by2 = pred
    ox1, oy1, ox2, oy2 = truth
    iw = min(bx2, ox2) - max(bx1, ox1)
    ih = min(by2, oy2) - max(by1, oy1)
    ia = max(0, iw) * max(0, ih)
    ba = (bx2 - bx1) * (by2 - by1)
    oa = (ox2 - ox1) * (oy2 - oy1)
    iou = ia / (ba + oa - ia)
    return iou

def calc_ap(recall, precision):
    precision = [1, *precision, 0]
    recall = [0, *recall, 1]
    area = 0
    for i in reversed(range(len(precision) - 1)):
        precision[i] = max(precision[i], precision[i + 1])
        area += precision[i] * (recall[i + 1] - recall[i])
    return area

def calc_map(generate, predict, num_images, num_class, iou_thresh):
    # Get data
    predictions = []
    targets = []
    for i in range(num_images):
        image, boxes, indices = generate()
        p_boxes, p_indices, p_scores = predict(image)
        for box, index in zip(boxes, indices):
            targets.append({
                "image": i, 
                "box": box, 
                "class": index, 
                "matched": False})
        for box, index, score in zip(p_boxes, p_indices, p_scores):
            predictions.append({
                "image": i, 
                "box": box, 
                "class": index,
                "score": score,
                "matched": False})
            
    # Sort predictions
    predictions.sort(key=lambda x: x["score"], reverse=True)

    # Decide if prediction is a true positive or false positive
    for i in range(len(predictions)):
        best_iou = None
        best_index = None
        for j in range(len(targets)):
            if predictions[i]["image"] != targets[j]["image"]: continue
            if predictions[i]["class"] != targets[j]["class"]: continue
            if targets[j]["matched"]: continue
            iou = calc_iou(predictions[i]["box"], targets[j]["box"])
            if (best_iou is None or iou > best_iou) and iou > iou_thresh:
                best_iou = iou
                best_index = j
        if best_index is not None:
            targets[best_index]["matched"] = True
            predictions[i]["matched"] = True

    aps = []
    for i in range(num_class):
        # Get this class predictions and targets
        preds = [pred for pred in predictions if pred["class"] == i]
        targs = [targ for targ in targets if targ["class"] == i]

        # Calculate precision and recall curve
        precision = []
        recall = []
        actual_positive = len(targs)
        true_positive = 0
        for j in range(len(preds)):
            if preds[j]["matched"]: true_positive += 1
            precision.append(true_positive / (j + 1))
            recall.append(0 if actual_positive == 0 else (true_positive / actual_positive))

        # Calculate area under curve
        aps.append(calc_ap(recall, precision))

    return sum(aps) / len(aps)

BACKGROUND_PATH = "../dataset/photo_jpg"
IMAGE_PATH = "../dataset/yolo"

if __name__ == "__main__":
    generator = Generator(BACKGROUND_PATH, IMAGE_PATH)
    predictor = Predictor()
    map_scores = []
    for i in range(10):
        thresh = 0.5 + 0.05 * i
        map_score = calc_map(generator.get, predictor.predict, 100, 4, thresh)
        map_scores.append(map_score)
        print(f"mAP@{thresh:.2f} is {map_score:.2f}")
    average = sum(map_scores) / len(map_scores)
    print(f"mAP@0.5:0.05:0.95 is {average:.2f}")