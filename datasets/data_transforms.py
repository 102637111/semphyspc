import numpy as np
import torch
import random

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _get_xyz_channels(pc):
    """Get XYZ channels from point cloud. Always returns first 3 channels."""
    return pc[:, :, 0:3]


def _get_rgb_channels(pc):
    """Get RGB channels from point cloud (channels 3:6), or None if only XYZ."""
    if pc.shape[-1] >= 6:
        return pc[:, :, 3:6]
    return None


def _has_rgb(pc):
    """Check if point cloud has RGB channels."""
    return pc.shape[-1] >= 6


class PointcloudNormalize(object):
    """ Normalize the batch data, use coordinates of the block centered at origin,
            Input:
                BxNxC array
            Output:
                BxNxC array
    """
    def __call__(self, batch_data):
        B, N, C = batch_data.shape
        normal_data = torch.zeros(B, N, C)
        for b in range(B):
            pc = batch_data[b]
            centroid = torch.mean(pc[:, :3], dim=0)  # center based on XYZ only
            pc[:, :3] = pc[:, :3] - centroid
            m = torch.max(torch.sqrt(torch.sum(pc[:, :3] ** 2, dim=1)))
            pc[:, :3] = pc[:, :3] / m
            normal_data[b] = pc
        normal_data = normal_data.to(batch_data.device)
        return normal_data


class PointcloudRotate(object):
    """Rotate around Y axis. Only affects XYZ, RGB is invariant to rotation."""
    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            R = torch.from_numpy(rotation_matrix.astype(np.float32)).to(pc.device)
            pc[i, :, 0:3] = torch.matmul(pc[i, :, 0:3], R)
        return pc


class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])

            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(
                xyz2).float().cuda()

        return pc


class PointcloudJitter(object):
    """Jitter XYZ coordinates. RGB channels are not jittered."""
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            jittered_data = pc.new(pc.size(1), 3).normal_(
                mean=0.0, std=self.std
            ).clamp_(-self.clip, self.clip)
            pc[i, :, 0:3] += jittered_data

        return pc


class PointcloudScale(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])

            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())

        return pc


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])

            pc[i, :, 0:3] = pc[i, :, 0:3] + torch.from_numpy(xyz2).float().cuda()

        return pc


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.5):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
            drop_idx = np.where(np.random.random((pc.size()[1])) <= dropout_ratio)[0]
            if len(drop_idx) > 0:
                cur_pc = pc[i, :, :]
                cur_pc[drop_idx.tolist(), 0:3] = cur_pc[0, 0:3].repeat(len(drop_idx), 1)  # set to the first point
                pc[i, :, :] = cur_pc

        return pc


class RandomHorizontalFlip(object):
    def __init__(self, upright_axis='z', is_temporal=False):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.is_temporal = is_temporal
        self.D = 4 if is_temporal else 3
        self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])

    def __call__(self, coords):
        bsize = coords.size()[0]
        for i in range(bsize):
            if random.random() < 0.95:
                for curr_ax in self.horz_axes:
                    if random.random() < 0.5:
                        coord_max = torch.max(coords[i, :, curr_ax])
                        coords[i, :, curr_ax] = coord_max - coords[i, :, curr_ax]
        return coords


# ---- Color-specific augmentations (only applied when pc has >= 6 channels) ----

class ColorJitter(object):
    """Randomly jitter RGB values. Only applied to channels 3:6."""
    def __init__(self, brightness=0.2, contrast=0.2, std=0.05):
        self.brightness = brightness
        self.contrast = contrast
        self.std = std

    def __call__(self, pc):
        if pc.shape[-1] < 6:
            return pc
        bsize = pc.size()[0]
        for i in range(bsize):
            rgb = pc[i, :, 3:6]
            # Brightness jitter
            if self.brightness > 0:
                delta = np.random.uniform(-self.brightness, self.brightness)
                rgb = rgb + delta
            # Contrast jitter
            if self.contrast > 0:
                factor = np.random.uniform(1 - self.contrast, 1 + self.contrast)
                mean_rgb = torch.mean(rgb, dim=0, keepdim=True)
                rgb = (rgb - mean_rgb) * factor + mean_rgb
            # Gaussian noise
            if self.std > 0:
                noise = pc.new(rgb.shape).normal_(mean=0.0, std=self.std)
                rgb = rgb + noise
            # Clamp to valid range
            rgb = torch.clamp(rgb, 0.0, 1.0)
            pc[i, :, 3:6] = rgb

        return pc


class ColorDropout(object):
    """Randomly drop color information from some points (set to gray/mean)."""
    def __init__(self, dropout_ratio=0.1):
        self.dropout_ratio = dropout_ratio

    def __call__(self, pc):
        if pc.shape[-1] < 6:
            return pc
        bsize = pc.size()[0]
        for i in range(bsize):
            mask = torch.rand(pc.size(1)) < self.dropout_ratio
            if mask.any():
                # Set dropped RGB values to mean color
                mean_rgb = pc[i, ~mask, 3:6].mean(dim=0)
                pc[i, mask, 3:6] = mean_rgb

        return pc


class ColorNormalize(object):
    """Normalize RGB channels to zero mean and unit variance."""
    def __init__(self, mean=None, std=None):
        # Default ImageNet-like normalization for RGB
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.std = std if std is not None else [0.229, 0.224, 0.225]

    def __call__(self, pc):
        if pc.shape[-1] < 6:
            return pc
        mean = torch.tensor(self.mean, device=pc.device).float().view(1, 1, 3)
        std = torch.tensor(self.std, device=pc.device).float().view(1, 1, 3)
        pc[:, :, 3:6] = (pc[:, :, 3:6] - mean) / std
        return pc
