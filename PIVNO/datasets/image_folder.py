from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from datasets.disparity_io import load_disparity, resolve_list_path


@register('image-folder')
class ImageFolder(Dataset):
    """Stereo triples listed as: left<TAB>right<TAB>disparity."""

    def __init__(self, root_path, disparity_scale=1.0):
        list_path = Path(root_path)
        if not list_path.exists():
            raise FileNotFoundError(f'{root_path} does not exist.')

        samples = []
        with open(list_path, 'r') as stream:
            for line_number, line in enumerate(stream, 1):
                fields = line.strip().split('\t')
                if not line.strip():
                    continue
                if len(fields) != 3:
                    raise ValueError(
                        f'{root_path}:{line_number} must contain three tab-separated paths'
                    )
                samples.append(tuple(resolve_list_path(p, list_path) for p in fields))

        self.samples = samples
        self.disparity_scale = disparity_scale
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        left_path, right_path, disparity_path = self.samples[idx]
        left = self.to_tensor(Image.open(left_path).convert('L'))
        right = self.to_tensor(Image.open(right_path).convert('L'))
        disparity = load_disparity(disparity_path, self.disparity_scale)
        return left, right, disparity
