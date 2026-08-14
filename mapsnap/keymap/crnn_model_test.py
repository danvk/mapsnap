import numpy as np
import torch

from mapsnap.keymap.crnn_model import (
    CRNN_HEIGHT,
    CRNN_WIDTH,
    NUM_CLASSES,
    build_crnn,
    central_group,
    ctc_greedy_decode,
    ctc_log_likelihood,
    decode_batch,
    encode_text,
    greedy_paths,
    ink_row_center,
    locate_number,
    number_strip,
    strip_crop_box,
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


def test_strip_crop_box_tighter_half_width():
    # A smaller half_w_working narrows the source box (un-squishing a multi-digit number)
    # while leaving the height and center unchanged.
    default = strip_crop_box(4000, 4000, 2000, 2000, factor=1.0)
    tight = strip_crop_box(4000, 4000, 2000, 2000, factor=1.0, half_w_working=30)
    default_w = default[2] - default[0]
    tight_w = tight[2] - tight[0]
    assert tight_w < default_w
    assert default[1] == tight[1] and default[3] == tight[3]  # same height
    assert (default[0] + default[2]) == (tight[0] + tight[2])  # same center x


def test_crnn_forward_and_decode_shapes():
    model = build_crnn().eval()
    x = torch.zeros(2, 1, CRNN_HEIGHT, CRNN_WIDTH)
    with torch.no_grad():
        log_probs = model(x)
    t, n, c = log_probs.shape
    assert n == 2 and c == NUM_CLASSES and t >= 7  # enough timesteps for 3 digits
    assert len(decode_batch(log_probs)) == 2
    assert len(greedy_paths(log_probs)) == 2


def test_central_group_picks_cluster_nearest_center():
    # Left cluster (t2-3) far from center; right cluster (t11-13) straddles center 11.5.
    path = [0] * 24
    for t in (2, 3):
        path[t] = 3
    for t in (11, 12, 13):
        path[t] = 4
    assert central_group(path) == (11, 13)


def test_central_group_merges_within_number_gaps():
    # Digits 2 blanks apart (< GAP_STEPS) belong to one number -> a single cluster.
    path = [0] * 24
    for t in (10, 13, 16):
        path[t] = 2
    assert central_group(path) == (10, 16)


def test_central_group_all_blank_is_none():
    assert central_group([0] * 24) is None


def test_locate_number_brackets_digits():
    # 48x160 strip: white with a dark digit-like block at rows 12..32, cols 40..60.
    strip = np.full((CRNN_HEIGHT, CRNN_WIDTH), 255, dtype=np.uint8)
    strip[12:32, 40:60] = 0
    # 40-step path (cell width CRNN_WIDTH/40 = 4): cols 40..60 -> timesteps 10..15.
    box = locate_number(strip, (10, 15), 40, (0, 0, CRNN_WIDTH, CRNN_HEIGHT))
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


def test_ctc_log_likelihood_prefers_the_emitted_digit():
    # Synthetic window: blank, strong "5" (index 6), blank.
    log_probs = np.full((3, 11), -10.0)
    log_probs[0, 0] = -0.01
    log_probs[1, 6] = -0.01
    log_probs[2, 0] = -0.01
    assert ctc_log_likelihood(log_probs, "5") > ctc_log_likelihood(log_probs, "4")
    # A two-digit string cannot be emitted in this window without a big penalty.
    assert ctc_log_likelihood(log_probs, "5") > ctc_log_likelihood(log_probs, "55")


def test_ctc_log_likelihood_two_digits():
    # "47": 4 fires at t0-1, 7 at t3-4, blank between.
    log_probs = np.full((5, 11), -12.0)
    for t_step, idx in ((0, 5), (1, 5), (2, 0), (3, 8), (4, 8)):
        log_probs[t_step, idx] = -0.01
    assert ctc_log_likelihood(log_probs, "47") > ctc_log_likelihood(log_probs, "4")
    assert ctc_log_likelihood(log_probs, "47") > ctc_log_likelihood(log_probs, "48")


def test_charset_round_trips_lettered_page_keys():
    """Letters are first-class: 1499K and 9A must encode and decode (#316)."""
    from mapsnap.keymap.crnn_model import (
        BLANK_INDEX,
        ctc_greedy_decode,
        encode_text,
    )

    for text in ("1499K", "9A", "53B", "384"):
        indices = encode_text(text)
        assert len(indices) == len(text)
        # Interleave blanks so CTC collapse reproduces repeated chars exactly.
        path: list[int] = []
        for i in indices:
            path += [i, BLANK_INDEX]
        assert ctc_greedy_decode(path) == text
    # Lowercase input normalizes; junk is skipped, not crashed on.
    assert encode_text("9a") == encode_text("9A")
    assert ctc_greedy_decode([]) == ""
