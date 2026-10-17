import numpy as np  # type: ignore

from server.utils.image_ops import rotate_by_degrees, resize_long_side


def test_rotate_by_degrees_90_shape_swap():
    img = np.zeros((50, 100, 3), dtype=np.uint8)  # h=50, w=100
    rot = rotate_by_degrees(img, 90)
    assert rot.shape[0] == 100 and rot.shape[1] == 50  # h,w swapped


def test_resize_long_side_downscale():
    img = np.zeros((200, 100, 3), dtype=np.uint8)  # long side = 200
    out = resize_long_side(img, 150)
    assert max(out.shape[:2]) == 150

