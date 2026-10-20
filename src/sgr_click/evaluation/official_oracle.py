"""Ten-branch oracle from UniGraCo (Apache-2.0); see THIRD_PARTY.md."""
import numpy as np
import torch
from .official_clicker import Clicker
from isegm.inference import utils

def evaluate_sample_oracle(image, gt_mask, predictor, max_iou_thr,
                           pred_thr=0.49, min_clicks=1, max_clicks=20,
                           sample_id=None, callback=None):
    clicker = Clicker(gt_mask=gt_mask)
    ious_lists = []
    click_indxs = []
    with torch.no_grad():
        predictor.set_input_image(image)
        min_num = 100
        for gra in range(1, 11):
            cur_gra = round(gra * 0.1, 1)
            ious_list = []
            clicker.reset_clicks()
            pred_mask = np.zeros_like(gt_mask)
            predictor.prev_prediction = torch.zeros_like(predictor.original_image[:, :1, :, :])
            for click_indx in range(max_clicks):
                clicker.make_next_click(pred_mask)
                pred_probs = predictor.get_prediction(clicker, gra=cur_gra)

                pred_mask = pred_probs > pred_thr
                iou = utils.get_iou(gt_mask, pred_mask)
                if callback is not None:
                    callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

                ious_list.append(iou)
                if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                    min_num = min(min_num, click_indx + 1)
                    break
                if min_num <= max_clicks and click_indx + 1 > min_num:
                    break
            ious_lists.append(np.array(ious_list, dtype=np.float32))
            click_indxs.append(click_indx)
        click_indxs = np.array(click_indxs)
        tgt_idxs = np.squeeze(np.argwhere(click_indxs == np.min(click_indxs)), axis=1)
        selected_ious = [ious_lists[i] for i in tgt_idxs]
        max_index = np.argmax([ious[0] for ious in selected_ious])
        ious = selected_ious[max_index]
        tgt_idx = tgt_idxs[max_index]
    return ious, tgt_idx
