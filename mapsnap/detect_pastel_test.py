import numpy as np

from mapsnap.detect_pastel import paint_pastels, pastel_mask


def solid(rgb: tuple[int, int, int], size: int = 4) -> np.ndarray:
    """A size x size RGB image filled with a single color."""
    img = np.empty((size, size, 3), dtype=np.uint8)
    img[:] = rgb
    return img


def test_paper_is_not_pastel():
    # Light yellowed paper: bright, slightly yellow, low saturation.
    assert not pastel_mask(solid((238, 232, 210))).any()


def test_black_ink_is_not_pastel():
    assert not pastel_mask(solid((20, 20, 20))).any()


def test_pastel_pink_is_detected():
    assert pastel_mask(solid((240, 170, 190))).all()


def test_pastel_blue_is_detected():
    assert pastel_mask(solid((160, 195, 225))).all()


def test_pastel_green_is_detected():
    assert pastel_mask(solid((175, 210, 175))).all()


def test_pastel_yellow_is_detected():
    # A saturated yellow region, distinct from the paler yellow of paper.
    assert pastel_mask(solid((235, 220, 120))).all()


def test_paint_pastels_marks_only_pastels_red():
    img = np.concatenate(
        [solid((238, 232, 210)), solid((240, 170, 190))], axis=1
    )  # paper | pink
    painted = paint_pastels(img)
    # Paper half is unchanged; pink half is now bright red.
    assert (painted[:, :4] == img[:, :4]).all()
    assert (painted[:, 4:] == (255, 0, 0)).all()
