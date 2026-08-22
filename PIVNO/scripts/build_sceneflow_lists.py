import argparse
import random
from pathlib import Path


def collect_dataset(root, dataset_name, image_pass, split=None):
    dataset_root = root / dataset_name
    image_root = dataset_root / image_pass
    disparity_root = dataset_root / 'disparity'
    if split is not None:
        image_root = image_root / split
        disparity_root = disparity_root / split

    samples = []
    for left in sorted(image_root.rglob('left/*.png')):
        relative = left.relative_to(image_root)
        right = left.parent.parent / 'right' / left.name
        disparity = disparity_root / relative.with_suffix('.pfm')
        if not right.is_file() or not disparity.is_file():
            raise FileNotFoundError(
                f'incomplete stereo sample: {left}, {right}, {disparity}'
            )
        samples.append((left, right, disparity))

    if not samples:
        raise RuntimeError(f'no samples found below {image_root}')
    return samples


def write_list(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as stream:
        for left, right, disparity in samples:
            stream.write(f'{left}\t{right}\t{disparity}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        type=Path,
        default=Path('/home/share/yijiayi/sceneflow'),
    )
    parser.add_argument('--output', type=Path, default=Path('./stereo'))
    parser.add_argument(
        '--image-pass',
        choices=('frames_finalpass', 'frames_cleanpass'),
        default='frames_finalpass',
    )
    parser.add_argument('--val-count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    flying_train = collect_dataset(
        args.root, 'FlyingThings3D', args.image_pass, split='TRAIN'
    )
    monkaa = collect_dataset(args.root, 'Monkaa', args.image_pass)
    driving = collect_dataset(args.root, 'Driving', args.image_pass)
    train = flying_train + monkaa + driving

    official_test = collect_dataset(
        args.root, 'FlyingThings3D', args.image_pass, split='TEST'
    )
    if not 0 < args.val_count < len(official_test):
        raise ValueError('val-count must be between 1 and TEST size - 1')

    random.Random(args.seed).shuffle(official_test)
    validation = official_test[:args.val_count]
    test = official_test[args.val_count:]

    write_list(args.output / 'FlowData_train.list', train)
    write_list(args.output / 'FlowData_val.list', validation)
    write_list(args.output / 'FlowData_test.list', test)
    print(
        f'wrote train={len(train)} '
        f'(FlyingThings3D={len(flying_train)}, Monkaa={len(monkaa)}, '
        f'Driving={len(driving)}), val={len(validation)}, '
        f'test={len(test)} to {args.output}'
    )


if __name__ == '__main__':
    main()
