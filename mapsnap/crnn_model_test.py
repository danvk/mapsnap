import numpy as np
import torch

from mapsnap.crnn_model import (
    CRNN_HEIGHT,
    CRNN_WIDTH,
    NUM_CLASSES,
    build_crnn,
    ctc_greedy_decode,
    decode_batch,
    encode_text,
    firing_span,
    greedy_paths,
    ink_row_center,
    locate_number,
    number_strip,
)


def test_encode_text():
    assert encode_text("0") == [1]
    assert encode_text("21") == [3, 2]
    assert encode_text("105") == [2, 1, 6]


def test_ctc_greedy_decode_collapses_repeats_and_blanks():
    # blank=0; digit d is index d+1. Path for "21": 3,3,blank,2 -> "21".
    assert ctc_greedy_decode([3, 3, 0, 2]) == "21"
    assert ctc_greedy_decode([0, 0]) == ""
    # "112": 2,0,2,3 -> "112" (blank separates the repeated 1s)
    assert ctc_greedy_decode([2, 0, 2, 3]) == "112"


def test_number_strip_shape_and_centering():
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    img[190:210, 190:210] = (0, 0, 0)  # dark marker at center
    strip = number_strip(img, 200, 200, factor=1.0)
    assert strip.shape == (CRNN_HEIGHT, CRNN_WIDTH)
    assert strip.dtype == np.uint8
    # The dark marker should darken the strip center region.
    assert strip[CRNN_HEIGHT // 2, CRNN_WIDTH // 2] < 128


def test_number_strip_handles_edge():
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    strip = number_strip(img, 0, 0, factor=1.0)  # corner, mostly off-image
    assert strip.shape == (CRNN_HEIGHT, CRNN_WIDTH)


def test_crnn_forward_and_decode_shapes():
    model = build_crnn().eval()
    x = torch.zeros(2, 1, CRNN_HEIGHT, CRNN_WIDTH)
    with torch.no_grad():
        log_probs = model(x)
    t, n, c = log_probs.shape
    assert n == 2 and c == NUM_CLASSES and t >= 7  # enough timesteps for 3 digits
    assert len(decode_batch(log_probs)) == 2
    assert len(greedy_paths(log_probs)) == 2


def test_firing_span():
    assert firing_span([0, 3, 3, 0, 2, 0]) == (1, 4)
    assert firing_span([0, 0, 0]) is None
    assert firing_span([5]) == (0, 0)


def test_locate_number_brackets_digits():
    # 48x96 strip: white with a dark digit-like block at rows 12..32, cols 40..60.
    strip = np.full((CRNN_HEIGHT, CRNN_WIDTH), 255, dtype=np.uint8)
    strip[12:32, 40:60] = 0
    # 24-step path (cell width 4): non-blank where the block is (cols 40..60 -> t 10..15).
    path = [0] * 24
    for t in range(10, 16):
        path[t] = 3
    box = locate_number(strip, path, (0, 0, CRNN_WIDTH, CRNN_HEIGHT))
    assert box is not None
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    # Box is much tighter than the full crop and brackets the dark block.
    assert min(xs) < 40 and max(xs) > 60 and min(xs) > 20 and max(xs) < 80
    assert min(ys) < 12 and max(ys) > 32


def test_ink_row_center_is_robust_to_speckle():
    # Dense digit band at rows 10..14 (center ~12), plus a stray speckle row far away.
    rows = np.zeros(48)
    rows[10:15] = 20.0
    rows[40] = 1.0  # speckle barely moves the centroid
    center = ink_row_center(rows)
    assert center is not None and 11.5 <= center <= 13.0


def test_ink_row_center_empty():
    assert ink_row_center(np.zeros(48)) is None


def test_locate_number_rejects_all_blank():
    strip = np.full((CRNN_HEIGHT, CRNN_WIDTH), 255, dtype=np.uint8)
    assert locate_number(strip, [0] * 24, (0, 0, CRNN_WIDTH, CRNN_HEIGHT)) is None
