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
