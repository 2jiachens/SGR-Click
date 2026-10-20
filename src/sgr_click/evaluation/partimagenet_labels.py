"""IDs in the original PartImageNet_Seg PNGs (parts 0..39, background 40).

Primary mapping: facebookresearch/VLPart, vlpart/data/datasets/partimagenet.py.
The independent whole-object PNGs use object classes 0..157, background 158.
"""
PART_NAMES = (
    'quadruped_head', 'quadruped_body', 'quadruped_foot', 'quadruped_tail',
    'biped_head', 'biped_body', 'biped_hand', 'biped_foot', 'biped_tail',
    'fish_head', 'fish_body', 'fish_fin', 'fish_tail',
    'bird_head', 'bird_body', 'bird_wing', 'bird_foot', 'bird_tail',
    'snake_head', 'snake_body', 'reptile_head', 'reptile_body', 'reptile_foot', 'reptile_tail',
    'car_body', 'car_tier', 'car_side_mirror',
    'bicycle_body', 'bicycle_head', 'bicycle_seat', 'bicycle_tier',
    'boat_body', 'boat_sail', 'aeroplane_head', 'aeroplane_body',
    'aeroplane_engine', 'aeroplane_wing', 'aeroplane_tail', 'bottle_mouth', 'bottle_body',
)
PART_IDS = dict(zip(PART_NAMES, range(40)))


def part_values(sample):
    """Resolve semantic names instead of trusting outdated exported mask IDs."""
    names = str(sample.get('source_labels', '')).replace(';', ',').split(',')
    if any(name.strip() for name in names):
        return [PART_IDS[name.strip().lower().replace(' ', '_')] for name in names if name.strip()]
    values = [int(value) for value in str(sample['source_values']).replace(';', ',').split(',') if value.strip()]
    if not values or any(value < 0 or value >= 40 for value in values):
        raise ValueError('PartImageNet part IDs must be in 0..39; 40 is background')
    return values


def corrected_sample(sample):
    corrected = dict(sample)
    values = ','.join(map(str, part_values(sample)))
    if str(sample['source_values']) != values:
        corrected['legacy_source_values'] = str(sample['source_values'])
    corrected['source_values'] = values
    return corrected
