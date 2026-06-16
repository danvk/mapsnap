"""Tests for mapsnap.scale_images."""

from PIL import Image

from mapsnap import scale_images


def test_scale_writes_quarter_size_to_output_dir(tmp_path, monkeypatch):
    src = tmp_path / "p5.jpg"
    Image.new("RGB", (400, 200), "white").save(src)
    out_dir = tmp_path / "scaled"

    monkeypatch.setattr(
        "sys.argv",
        ["mapsnap scale", str(src), "--output-dir", str(out_dir)],
    )
    scale_images.main()

    out_path = out_dir / "p5.jpg"
    assert out_path.exists()
    with Image.open(out_path) as img:
        assert img.size == (100, 50)
