'''
	Use pre-trained Yolo3 to detect objects and their positions
'''

import torch

from utils.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conf_thres = 0.25
iou_thres = 0.45
half = False

''' Object detection inference '''
def detect(source, weights, imgsz=256):

	# Load model
	model = attempt_load(weights, map_location=device)  # load FP32 model
	imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
	if half:
		model.half()  # to FP16

	# Set Dataloader
	dataset = LoadImages(source, img_size=imgsz)

	# Run inference
	img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
	_ = model(img.half() if half else img) if device.type != 'cpu' else None

	# Run all the images
	preds = []
	for _, img, _, _ in dataset:
		img = torch.from_numpy(img).to(device)
		img = img.half() if half else img.float()  # uint8 to fp16/32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		# Inference
		pred = model(img, False)[0]

		# Apply NMS
		pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, \
									agnostic=False)
		preds.append(pred[0])

	return preds


# $ pycodestyle setup.py insta485generator
# $ pylint --disable=no-value-for-parameter setup.py insta485generator
# $ pydocstyle setup.py insta485generator