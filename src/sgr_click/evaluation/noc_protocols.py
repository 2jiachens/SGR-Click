"""Current GUI model branches; original official ten-point oracle and best-first NoC."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

FIELDS = ['states', 'probs_history', 'simple_probs_history', 'click_context_history',
          '_init_mask', '_requested_granularity', '_quality_path_ready',
          '_quality_edge_cache', '_quality_lab', '_quality_constraints']


def digest(mask):
    return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()


def click_dict(click):
    return dict(x=int(click.coords[1]), y=int(click.coords[0]), positive=bool(click.is_positive))


def threshold_metrics(ious, maximum=20):
    curve = np.asarray(ious, dtype=np.float32)
    result = {}
    for threshold in (85, 90):
        hits = np.flatnonzero(curve >= np.float32(threshold / 100))
        result[str(threshold)] = dict(noc=int(hits[0])+1 if len(hits) else maximum,
                                      reached=bool(len(hits)))
    return result


def best_tick(ious):
    # Earliest native tick wins an exact tie, including 0 and 100.
    if len(ious) != 101:
        raise ValueError('Expected every native tick 0..100')
    return int(np.argmax(np.asarray(ious)))


def load_official():
    from .official_oracle import evaluate_sample_oracle
    from .official_clicker import Clicker
    from isegm.inference.utils import get_iou
    return SimpleNamespace(Clicker=Clicker, iou=get_iou, oracle=evaluate_sample_oracle)


def snapshot(c):
    source = c.granularity_refiner.source
    return dict(fields=copy.deepcopy({k: getattr(c, k) for k in FIELDS if hasattr(c, k)}),
        clicker=copy.deepcopy(c.clicker.get_state()),
        predictor=copy.deepcopy(c.predictor.get_states()),
        refiner=c.granularity_refiner.__dict__.copy(), source=source.__dict__.copy(),
        external=source._external_trajectory.__dict__.copy(),
        semantic_cache=c.semantic_cache.copy(), edge_cache=c.edge_cache.copy())


def restore(c, state):
    for key, value in copy.deepcopy(state['fields']).items():
        setattr(c, key, value)
    c.clicker.set_state(copy.deepcopy(state['clicker']))
    c.predictor.set_states(copy.deepcopy(state['predictor']))
    c.granularity_refiner.__dict__.update(state['refiner'])
    c.granularity_refiner.source.__dict__.update(state['source'])
    c.granularity_refiner.source._external_trajectory.__dict__.update(state['external'])
    c.semantic_cache = state['semantic_cache'].copy()
    c.edge_cache = state['edge_cache'].copy()


class Branches:
    def __init__(self, controller, image, target, official):
        self.c, self.target, self.official = controller, target, official
        self.image = image
        clicks = self.new_clicker()
        first = clicks.get_clicks()[0]
        controller.set_image(image)
        controller.set_granularity(0.)
        controller.add_click(int(first.coords[1]), int(first.coords[0]), bool(first.is_positive))
        if not controller.granularity_ready:
            raise RuntimeError('No first-click semantic path')
        self.state = snapshot(controller)
        self.first_click = click_dict(first)

    def new_clicker(self):
        clicker = self.official.Clicker(gt_mask=self.target)
        clicker.make_next_click(np.zeros_like(self.target, dtype=bool))
        return clicker

    def start(self, tick):
        restore(self.c, self.state)
        self.c.set_granularity(tick / 100.)
        return (self.c.current_object_prob > self.c.prob_thresh).copy()

    def next(self, clicks, check_handoff=False):
        click = clicks.get_clicks()[-1]
        predictor = self.c.predictor
        original = predictor.get_prediction
        captured = []
        expected = (self.c.current_object_prob > self.c.prob_thresh).copy() if check_handoff else None
        if check_handoff:
            def checked(*args, **kwargs):
                actual = predictor.prev_prediction.detach().cpu().numpy()[0, 0] > self.c.prob_thresh
                np.testing.assert_array_equal(actual, expected,
                    err_msg='Correction discarded the selected slider mask')
                captured.append(digest(actual))
                return original(*args, **kwargs)
            predictor.get_prediction = checked
        try:
            self.c.add_click(int(click.coords[1]), int(click.coords[0]), bool(click.is_positive))
        finally:
            predictor.get_prediction = original
        if check_handoff and not captured:
            raise RuntimeError('Correction predictor was not called')
        return (self.c.current_object_prob > self.c.prob_thresh).copy(), captured

    def best_first(self, maximum=20):
        restore(self.c, self.state)
        scan = []
        for tick in range(101):
            self.c.set_granularity(tick / 100.)
            scan.append(float(self.official.iou(self.target, self.c.current_object_prob > self.c.prob_thresh)))
        tick = best_tick(scan)
        mask = self.start(tick)
        initial_hash = digest(mask)
        clicks = self.new_clicker()
        ious = [float(self.official.iou(self.target, mask))]
        recorded_clicks = [click_dict(clicks.get_clicks()[0])]
        handoff = []
        while len(ious) < maximum and ious[-1] < .90:
            clicks.make_next_click(mask)
            mask, captured = self.next(clicks, check_handoff=len(ious)==1)
            handoff += captured
            ious.append(float(self.official.iou(self.target, mask)))
            recorded_clicks.append(click_dict(clicks.get_clicks()[-1]))
        assert np.isclose(ious[0], max(scan))
        return dict(selected_tick=tick, scan_ticks=list(range(101)), initial_ious=scan,
                    ious=ious, clicks=recorded_clicks, noc=threshold_metrics(ious, maximum),
                    initial_mask_sha256=initial_hash, correction_input_sha256=handoff,
                    final_mask_sha256=digest(mask))

    def ten(self, stop_threshold, maximum=20):
        traces = []
        branches = self
        class PredictorBridge:
            def set_input_image(self, image):
                self.original_image = torch.zeros((1, 1, *image.shape[:2]))
            def get_prediction(self, clicks, gra):
                if len(clicks) == 1:
                    mask = branches.start(int(round(gra * 100)))
                    traces.append(dict(tick=int(round(gra*100)), ious=[], clicks=[], correction_input_sha256=[]))
                else:
                    mask, captured = branches.next(clicks, check_handoff=len(clicks)==2)
                    traces[-1]['correction_input_sha256'] += captured
                return mask.astype(np.float32)
        def callback(image, target, probability, sample_id, step, clicks):
            trace = traces[-1]
            trace['ious'].append(float(self.official.iou(target, probability > .5)))
            trace['clicks'].append(click_dict(clicks[-1]))
            if step == 0:
                trace['initial_mask_sha256'] = digest(probability > .5)
        curve, index = self.official.oracle(self.image, self.target, PredictorBridge(), stop_threshold,
            pred_thr=.5, min_clicks=1, max_clicks=maximum, callback=callback)
        assert [r['tick'] for r in traces] == list(range(10, 101, 10))
        np.testing.assert_array_equal(curve, np.asarray(traces[index]['ious'], np.float32))
        return dict(stop_threshold=stop_threshold, selected_tick=traces[index]['tick'],
                    ious=curve.tolist(), noc=threshold_metrics(curve, maximum), trajectories=traces,
                    original_official_function_executed=True)


def measure(controller, image, target, is_part, official, validate=False):
    branches = Branches(controller, image, target, official)
    primary = branches.ten(.85 if is_part else .90)
    extension = branches.ten(.90) if is_part else None
    ten_metrics = copy.deepcopy(primary['noc'])
    if is_part:
        ten_metrics['90'] = extension['noc']['90']
    best = branches.best_first()
    validation = []
    if validate:
        # Both routines must restore the same first-click model state.
        repeated_best = branches.best_first()
        assert best == repeated_best, 'Best-first correction changed with branch history'
        repeated_ten = branches.ten(.85 if is_part else .90)
        assert primary == repeated_ten, 'Ten-point correction changed after best-first evaluation'
        for tick in [10, 50, 100]:
            a = branches.start(tick)
            fresh = Branches(controller, image, target, official).start(tick)
            np.testing.assert_array_equal(a, fresh)
        validation = ['best-first replay identical', 'ten-oracle replay identical',
                      'fresh first-click masks at g=10/50/100 identical']
    return dict(first_click=branches.first_click,
                ten=dict(noc=ten_metrics, primary=primary, extension90=extension),
                best101=best, validation=validation, construction_failed=False)
