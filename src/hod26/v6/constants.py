"""Immutable constants for the independent HOD26 V6 lineage."""

from __future__ import annotations

HOD26_CLASSES = (
    "apple",
    "apple_plastic",
    "badminton",
    "banana",
    "banana_plastic",
    "car",
    "car_toy",
    "charger_head",
    "e-bike",
    "egg",
    "egg_plastic",
    "egg_wood",
    "orange",
    "orange_plastic",
    "people",
    "rubik",
    "stone_block",
    "table_tennis",
)

WEAK4_CLASSES = ("car", "e-bike", "people", "stone_block")
WEAK4_IDS = tuple(HOD26_CLASSES.index(name) for name in WEAK4_CLASSES)

# XIMEA xiSpec2 VIS3 4x4 row-major band order.  The values are buffers in the
# V6 model and are therefore carried by every standalone final checkpoint.
WAVELENGTHS_NM = (
    464.5,
    472.8,
    480.2,
    489.3,
    499.0,
    508.2,
    516.3,
    526.1,
    534.7,
    544.3,
    552.3,
    561.8,
    571.2,
    580.5,
    588.1,
    597.2,
)

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)

HOD26_TO_COCO = {
    "apple": ("apple",),
    "apple_plastic": ("apple",),
    "badminton": ("tennis racket",),
    "banana": ("banana",),
    "banana_plastic": ("banana",),
    "car": ("car",),
    "car_toy": ("car",),
    "charger_head": ("cell phone", "remote", "mouse"),
    "e-bike": ("bicycle", "motorcycle"),
    "egg": ("sports ball",),
    "egg_plastic": ("sports ball",),
    "egg_wood": ("sports ball",),
    "orange": ("orange",),
    "orange_plastic": ("orange",),
    "people": ("person",),
    "rubik": ("book", "clock", "sports ball"),
    "stone_block": ("book", "suitcase"),
    "table_tennis": ("sports ball",),
}

OFFICIAL_CODETR_COMMIT = "26653521e47157fdd819e764b4bb949057ea9f3b"
OFFICIAL_CHECKPOINT_SHA256 = (
    "733d2ccde180a55151a68a6cab7c9f42b117d24d38d6197b37caf3189243256c"
)
OFFICIAL_CHECKPOINT_URL = (
    "https://huggingface.co/zongzhuofan/co-detr-vit-large-coco/"
    "resolve/main/pytorch_model.pth"
)

# Fold0 annotation counts only.  They are recorded rather than inferred from
# validation performance, so the loss policy cannot leak Fold0 AP into train.
FOLD0_CLASS_COUNTS = (
    357, 395, 1099, 498, 295, 452, 284, 361, 369,
    380, 527, 206, 148, 248, 876, 782, 218, 561,
)
