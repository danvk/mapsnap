"""The pixel -> (lon, lat) transform an IIIF georeference annotation describes."""

import math
from collections.abc import Callable

import numpy as np

Transform = Callable[[float, float], tuple[float, float]]


def page_transform(annotation: dict) -> Transform | None:
    """A pixel -> (lon, lat) function for one annotation, fitted as its transformation says.

    Fitted in a local metric frame (longitude scaled by cos latitude), as Allmaps
    fits in projected coordinates: a similarity for ``helmert`` (or with only two
    control points), an affine for a first-order polynomial. None with fewer than two.
    """
    pairs = [
        (feature["properties"]["resourceCoords"], feature["geometry"]["coordinates"])
        for feature in annotation["body"]["features"]
        if feature.get("properties", {}).get("resourceCoords")
        and feature.get("geometry")
    ]
    if len(pairs) < 2:
        return None
    pixels = np.array([pixel for pixel, _ in pairs], float)
    geo = np.array([lonlat for _, lonlat in pairs], float)
    k = math.cos(math.radians(geo[:, 1].mean()))
    metric = np.c_[geo[:, 0] * k, geo[:, 1]]
    kind = annotation["body"].get("transformation", {}).get("type")
    if kind == "helmert" or len(pairs) == 2:
        # A least-squares similarity is a complex linear map, w = a z + b. Pixel y
        # runs down and latitude up, so y is flipped first: no similarity can
        # mirror, and without the flip the fit shrinks and skews the page.
        z = pixels[:, 0] - 1j * pixels[:, 1]
        w = metric[:, 0] + 1j * metric[:, 1]
        (a, b), *_ = np.linalg.lstsq(np.c_[z, np.ones_like(z)], w, rcond=None)

        def similarity(x: float, y: float) -> tuple[float, float]:
            point = complex(a * complex(x, -y) + b)
            return (point.real / k, point.imag)

        return similarity
    coefficients, *_ = np.linalg.lstsq(
        np.c_[pixels, np.ones(len(pixels))], metric, rcond=None
    )

    def affine(x: float, y: float) -> tuple[float, float]:
        mx, my = np.array([x, y, 1.0]) @ coefficients
        return (float(mx) / k, float(my))

    return affine
