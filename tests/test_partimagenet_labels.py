import zipfile
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from sgr_click.evaluation.partimagenet_labels import corrected_sample, part_values
from sgr_click.evaluation.semantic_coverage_data import SemanticHierarchyDataset


def test_original_png_category_ids():
    cases = [('quadruped_head', 0), ('quadruped_body', 1), ('quadruped_foot', 2),
             ('bird_wing', 15), ('bird_foot', 16), ('bicycle_body', 27),
             ('bicycle_head', 28), ('bottle_mouth', 38), ('bottle_body', 39)]
    for name, expected in cases:
        row = {'source_labels': name, 'source_values': '1'}
        assert part_values(row) == [expected]
        assert corrected_sample(row)['source_values'] == str(expected)
    with pytest.raises(ValueError, match='background'):
        part_values({'source_values': '40'})


def test_whole_does_not_include_background_or_drop_class_zero(tmp_path):
    parts = np.array([[40, 40, 40], [0, 1, 40], [40, 2, 40]], dtype=np.uint8)
    independent_whole = np.array([[158, 158, 158], [0, 0, 158], [158, 158, 158]], dtype=np.uint8)
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    archive = tmp_path / 'annotations.zip'
    with zipfile.ZipFile(archive, 'w') as file:
        for name, array in [('images/test/example.JPEG', image),
                            ('annotations/test/example.png', parts),
                            ('annotations/test_whole/example.png', independent_whole)]:
            extension = '.jpg' if name.endswith('JPEG') else '.png'
            file.writestr('PartImageNet/' + name, cv2.imencode(extension, array)[1].tobytes())
    args = SimpleNamespace(dataset='PartImageNet', partimagenet_zip=archive)
    sample = dict(split='test', sample_id='example', source_labels='quadruped_head', source_values='1')
    dataset = SemanticHierarchyDataset(args)
    try:
        _, (part, whole), _ = dataset.load(sample)
        np.testing.assert_array_equal(part, parts == 0)
        np.testing.assert_array_equal(whole, parts < 40)
        assert np.all(~part | whole)
        args.partimagenet_whole_source = 'official'
        _, (_, whole), _ = dataset.load(sample)
        np.testing.assert_array_equal(whole, independent_whole < 158)
    finally:
        dataset.close()
