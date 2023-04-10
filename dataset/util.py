from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform


try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp
    
    

def build_transform(is_train, config):
        resize_im = config.DATA.IMG_SIZE > 32
        if is_train:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=config.DATA.IMG_SIZE,
                is_training=True,
                color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                re_prob=config.AUG.REPROB,
                re_mode=config.AUG.REMODE,
                re_count=config.AUG.RECOUNT,
                interpolation=config.DATA.INTERPOLATION,
            )
            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
            return transform

        t = []
        if resize_im:
            if config.TEST.CROP:
                size = int((256 / 224) * config.DATA.IMG_SIZE)
                t.append(
                    transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                    # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
            else:
                t.append(
                    transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                    interpolation=_pil_interp(config.DATA.INTERPOLATION))
                )

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)
    
    
def build_mixup(config):
    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. \
        or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, 
            cutmix_alpha=config.AUG.CUTMIX, 
            cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, 
            switch_prob=config.AUG.MIXUP_SWITCH_PROB, 
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, 
            num_classes=config.MODEL.NUM_CLASSES)