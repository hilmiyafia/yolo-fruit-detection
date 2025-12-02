
import torch
from blocks import Conv, C2f

class YOLO(torch.nn.Module):
    
    def __init__(self, labels, size=256, d=0.33, w=0.25, r=2):
        super().__init__()
        s = 16
        k = size // s
        n = len(labels)
        self.backbone = torch.nn.Sequential(
            Conv(3, 128 * w, s=2),
            C2f(128 * w, 128 * w, n=3 * d),
            Conv(128 * w, 256 * w, s=2),
            C2f(256 * w, 256 * w, n=6 * d), 
            Conv(256 * w, 512 * w, s=2),
            C2f(512 * w, 512 * w, n=6 * d), 
            Conv(512 * w, 512 * w * r, s=2),
            C2f(512 * w * r, 512 * w * r, n=3 * d))
        self.detect_class = torch.nn.Sequential(
            Conv(512 * w * r, 512 * w * r),
            Conv(512 * w * r, 512 * w * r),
            torch.nn.Conv2d(round(512 * w * r), n, 1))
        self.detect_box = torch.nn.Sequential(
            Conv(512 * w * r, 512 * w * r),
            Conv(512 * w * r, 512 * w * r),
            torch.nn.Conv2d(round(512 * w * r), 64, 1),
            torch.nn.Unflatten(1, (16, 4)),
            torch.nn.Softmax(1))
        bins = torch.arange(16)[None, :, None, None, None] / 16 * size
        self.register_buffer("bins", bins)
        centers = torch.arange(k).repeat((k, 1)) * s + (s / 2)
        centers = torch.stack((centers, centers.T))[None]
        self.register_buffer("centers", centers)
        self.labels = labels
    
    def forward(self, x):
        backbone = self.backbone(x)
        detect_class = self.detect_class(backbone)
        detect_box = (self.detect_box(backbone) * self.bins).sum(1)
        point_min = self.centers - detect_box[:, :2]
        point_max = self.centers + detect_box[:, 2:]
        return torch.concat((point_min, point_max, detect_class), 1)

if __name__ == "__main__":
    import torchinfo, subprocess, os
    labels = ["a", "b", "c"]
    yolo = YOLO(labels)
    dummy = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model=yolo, 
        args=dummy, 
        f="o.onnx", 
        input_names=["input"], 
        output_names=["output"])
    subprocess.run(["onnxsim", "o.onnx", "yolo.onnx"])
    os.remove("o.onnx")
    torchinfo.summary(yolo, input_data=dummy)
