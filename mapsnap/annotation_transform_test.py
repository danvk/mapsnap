import pytest

from mapsnap.annotation_transform import page_transform


def annotation(gcps: list[tuple[float, float]], transformation: str) -> dict:
    """GCPs whose ground positions are an affine of their pixels: 1 px = 1e-5 deg east, 2e-5 south."""
    return {
        "body": {
            "transformation": {"type": transformation},
            "features": [
                {
                    "properties": {"resourceCoords": [x, y]},
                    "geometry": {"coordinates": [x * 1e-5, -y * 2e-5]},
                }
                for x, y in gcps
            ],
        }
    }


def test_page_transform_fits_an_affine_for_a_polynomial():
    page = annotation([(0, 0), (100, 0), (0, 100)], "polynomial")
    transform = page_transform(page)
    assert transform is not None
    assert transform(50, 50) == pytest.approx((50e-5, -100e-5))


def test_page_transform_fits_a_similarity_for_helmert():
    page = annotation([(0, 0), (100, 0)], "helmert")
    transform = page_transform(page)
    assert transform is not None
    # A similarity keeps the page square, whatever the GCPs' aspect.
    assert transform(0, 100) == pytest.approx((0, -100e-5), abs=1e-9)


def test_page_transform_needs_two_gcps():
    assert page_transform(annotation([(0, 0)], "helmert")) is None
