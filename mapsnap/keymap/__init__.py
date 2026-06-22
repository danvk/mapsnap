"""Key-map page-number recognition and georeferencing.

A Sanborn volume's key map (index map) labels each sheet's area with its page number. This
package locates and reads those numbers, then uses them to georeference the key map:

- ``keymap_patches``, ``number_model``, ``train_number_detector``, ``detect_numbers_cnn`` —
  the CNN page-number localizer (sliding-window MobileNetV3) and its training/inference.
- ``crnn_model``, ``train_crnn``, ``detect_numbers_crnn`` — the CRNN recognizer that reads
  each localized number, and the end-to-end detector that writes ``<stem>.keymap.json``.
- ``records`` — shared detection-record / page-spec helpers.
- ``score_keymap_labels`` — point-in-polygon scorer against hand labels.
- ``fit_keymap`` — RANSAC-fit a transform from page-number detections to georeferenced page
  footprints.
"""
