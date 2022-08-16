import os
import cv2
import yaml
import collections
import numpy as np
import synthetic_data
from tqdm import tqdm
from pathlib import Path
from shutil import copyfile

def dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def parse_primitives(names, all_primitives):
    p = all_primitives if (names == 'all') \
            else (names if isinstance(names, list) else [names])
    assert set(p) <= set(all_primitives)
    return p


def dump_primitive_data(primitive, config):
    temp_dir = Path(data_path, "synthetic_shapes" + ('_{}'.format(config['data']['suffix'])), "original", primitive)
    synthetic_data.set_random_state(np.random.RandomState(config['data']['generation']['random_seed']))

    print(f"Generating dataset for Pimitive {primitive}")
    for split, size in config['data']['generation']['split_sizes'].items():
        im_dir, pts_dir = [Path(temp_dir, i, split) for i in ['images', 'points']]
        im_dir.mkdir(parents=True, exist_ok=True)
        pts_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(size), desc=split, leave=False):
            image = synthetic_data.generate_background(config['data']['generation']['image_size'], **config['data']['generation']['params']['generate_background'])
            points = np.array(getattr(synthetic_data, primitive)(image, **config['data']['generation']['params'].get(primitive, {})))
            points = np.flip(points, 1)  # reverse convention with opencv

            b = config['data']['preprocessing']['blur_size']
            image = cv2.GaussianBlur(image, (b, b), 0)
            points = (points * np.array(config['data']['preprocessing']['resize'], np.float32) / np.array(config['data']['generation']['image_size'], np.float32))
            image = cv2.resize(image, tuple(config['data']['preprocessing']['resize'][::-1]), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(str(Path(im_dir, '{}.png'.format(i))), image)
            np.save(Path(pts_dir, '{}.npy'.format(i)), points)


def generate_dataset():
    primitives = parse_primitives(config['data']['primitives'], drawing_primitives)
    basepath = Path(data_path, 'synthetic_shapes' + ('_{}'.format(config['data']['suffix']) if config['data']['suffix'] is not None else ''), "original")
    basepath.mkdir(parents=True, exist_ok=True)

    splits = {s: {'images': [], 'points': []} for s in ['training', 'validation', 'test']}
    for primitive in primitives:
        if not Path(data_path, 'synthetic_shapes' + ('_{}'.format(config['data']['suffix'])), "original", primitive).exists():
            dump_primitive_data(primitive, config)

        # Gather filenames in all splits, optionally truncate
        truncate = config['data']['truncate'].get(primitive, 1)
        path = Path(data_path, 'synthetic_shapes' + ('_{}'.format(config['data']['suffix']) if config['data']['suffix'] is not None else ''), "original", primitive)
        for s in splits:
            e = [str(p) for p in Path(path, 'images', s).iterdir()]
            f = [p.replace('images', 'points') for p in e]
            f = [p.replace('.png', '.npy') for p in f]
            splits[s]['images'].extend(e[:int(truncate*len(e))])
            splits[s]['points'].extend(f[:int(truncate*len(f))])

    for s in splits:
        perm = np.random.RandomState(0).permutation(len(splits[s]['images']))
        for obj in ['images', 'points']:
            splits[s][obj] = np.array(splits[s][obj])[perm].tolist()
    return splits


if __name__ == "__main__":
    drawing_primitives = [
            'draw_lines',
            'draw_polygon',
            'draw_multiple_polygons',
            'draw_ellipses',
            'draw_star',
            'draw_checkerboard',
            'draw_stripes',
            'draw_cube',
            'gaussian_noise'
    ]

    config_path = "./magic-point_shapes.yaml"    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_path = "/home/ubuntu/Datasets/synthetic_shapes"
    total_sets = generate_dataset()
    
    training_images = total_sets["training"]["images"]
    training_points = total_sets["training"]["points"]
    
    for key in total_sets.keys():
        save_dir = f"{data_path}/synthetic_shapes_" + config["data"]["suffix"] + f"/{key}"
        
        if not os.path.isdir(save_dir):
            os.makedirs(f"{save_dir}/images")
            os.makedirs(f"{save_dir}/points")

        # for index, (png_file, np_file) in enumerate(zip(total_sets[key]["images"], total_sets[key]["points"])):
        #     copyfile(png_file, f"{save_dir}/images/{index:>09}.png")
        #     copyfile(np_file, f"{save_dir}/points/{index:>09}.npy")

        for index in tqdm(range(len(total_sets[key]["images"]))):
            png_file = total_sets[key]["images"][index]
            npy_file = total_sets[key]["points"][index]

            copyfile(png_file, f"{save_dir}/images/{index:>09}.png")
            copyfile(npy_file, f"{save_dir}/points/{index:>09}.npy")