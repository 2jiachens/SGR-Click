from pathlib import Path
import sys

sys.path.append(str(Path(sys.path[0]).parent))

import torch
import cv2
from isegm.inference import utils
from isegm.model.modeling.pos_embed import interpolate_pos_embed_inference
from isegm.inference.predictors.base import BasePredictor
from isegm.inference.transforms import ZoomIn
from isegm.inference.clicker import Clicker, Click

torch.backends.cudnn.deterministic = True

model_folder = '../weights'
model_name = 'cocolvis_vitb.pth'
checkpoint_path = utils.find_checkpoint(model_folder, model_name)
device = torch.device(f'cuda:0')
model = utils.load_is_model(
    checkpoint_path, 
    device=device,
    eval_ritm=False, 
    cpu_dist_maps=True
)

zoom_in_params = {
    'skip_clicks': -1,
    'target_size': (448, 448)
}
zoom_in = ZoomIn(**zoom_in_params)

interpolate_pos_embed_inference(
    model.backbone,
    infer_img_size=zoom_in_params['target_size'],
    device=device
)

predictor = BasePredictor(
    model, 
    device, 
    with_flip=True,
    zoom_in=zoom_in
)

# Evaluate sample
clicker = Clicker()
click = Click(is_positive=True, coords=(300, 300))
clicker.add_click(click)

image_path = '../assets/test_imgs/sheep.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


pred_thr = 0.49
with torch.no_grad():
    predictor.set_input_image(image)
    pred = predictor.get_prediction(clicker)
    pred_mask = pred > pred_thr
