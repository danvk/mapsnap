"""Vocabulary-guided prefix-constrained CTC beam search for street-name recognition.

Implements a character-level trie over known street-name forms and a CTC beam search
that at each time step only extends into characters that are valid trie continuations.
This contrasts with EasyOCR's built-in ``wordbeamsearch``, which runs free beam search
first and only consults the vocabulary as a post-hoc filter.

Usage::

    from mapsnap.ctc_vocab_decode import generate_vocab_strings, patch_easyocr_reader

    vocab = generate_vocab_strings(normalized_streets)  # set[str] from build_block_index
    patch_easyocr_reader(reader, vocab)
    reader.readtext(img, decoder="wordbeamsearch", ...)  # constrained path active

After patching, ``easyocr.recognition.recognizer_predict`` is replaced with a version
that computes confidence from the constrained CTC path probability rather than greedy
max probabilities, so false positives get low confidence instead of the confidence of
the actual (non-street) image content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from mapsnap.streets import (
    DIRECTION_ABBREVS,
    HINT_STRINGS,
    ORDINAL_WORD_TO_NUM,
    STREET_ABBREVS,
)

# Matches numeric ordinal suffixes: "5TH", "12ND", "3RD", "21ST", etc.
_ORDINAL_RE = re.compile(r"^(\d+)(ST|ND|RD|TH)$")

# ---------------------------------------------------------------------------
# Vocabulary generation: canonical names → abbreviated label forms
# ---------------------------------------------------------------------------

# Inverse of DIRECTION_ABBREVS: full direction → [full, abbrev, abbrev., ...].
_DIR_TO_ABBREVS: dict[str, list[str]] = {}
for _abbrev, _full in DIRECTION_ABBREVS.items():
    if _full not in _DIR_TO_ABBREVS:
        _DIR_TO_ABBREVS[_full] = [_full]
    _DIR_TO_ABBREVS[_full].extend([_abbrev, _abbrev + "."])

# Inverse of STREET_ABBREVS: full type → [full, abbrev1, abbrev2, ...].
_TYPE_TO_ABBREVS: dict[str, list[str]] = {}
for _abbrev, _full in STREET_ABBREVS.items():
    if _full not in _TYPE_TO_ABBREVS:
        _TYPE_TO_ABBREVS[_full] = [_full]
    _TYPE_TO_ABBREVS[_full].append(_abbrev)

# HINT_STRINGS is defined in streets.py (alongside STREET_TYPES) and re-exported here
# for callers that import from ctc_vocab_decode.


def generate_vocab_strings(normalized_streets: set[str]) -> list[str]:
    """Generate all abbreviated forms that normalize to a known street name.

    For each canonical form (e.g. "EAST GRAND AVENUE"), produces the abbreviated
    label forms that might appear on Sanborn maps (e.g. "E. GRAND AV", "E GRAND",
    "GRAND"). These become the vocabulary for the constrained CTC decoder.

    The generated strings use the same character set as EasyOCR's allowlist
    (A–Z, a–z, space, period) so the CTC can actually produce them.
    """
    result: set[str] = set()

    for street in normalized_streets:
        words = street.split()
        if not words:
            continue

        # --- Determine direction-prefix variants ---
        first = words[0]
        if first in _DIR_TO_ABBREVS:
            # None means no direction prefix (bare name)
            dir_forms: list[str | None] = [None] + _DIR_TO_ABBREVS[first]
            body = words[1:]
        elif first == "SAINT":
            # "SAINT CLAIR" can appear as "ST CLAIR" (ST = Saint abbreviation)
            dir_forms = [None, "SAINT", "ST"]
            body = words[1:]
        else:
            dir_forms = [None]
            body = words

        if not body:
            continue

        # Strip a trailing direction word (e.g. "NORTHEAST" in "EIGHTH STREET NORTHEAST").
        # DC-style centerline names carry the quadrant as a suffix, but map labels omit
        # it; stripping here lets the trie emit "EIGHTH", "8TH", "EIGHTH ST", etc.
        if len(body) >= 2 and body[-1] in _DIR_TO_ABBREVS:
            body = body[:-1]

        # --- Determine street-type suffix variants ---
        last = body[-1]
        if last in _TYPE_TO_ABBREVS:
            # None means no type suffix
            type_forms: list[str | None] = [None] + _TYPE_TO_ABBREVS[last]
            name_words = body[:-1]
        else:
            type_forms = [None]
            name_words = body

        if not name_words:
            if first not in _DIR_TO_ABBREVS:
                continue
            # e.g. "WEST STREET": the direction word is the street name, not a prefix.
            # Generate all abbreviated name forms × type forms (no separate direction prefix).
            for name_form in _DIR_TO_ABBREVS[first]:
                for type_form in type_forms:
                    parts = [name_form]
                    if type_form is not None:
                        parts.append(type_form)
                    result.add(" ".join(parts))
            continue

        # e.g. "AVENUE X": type word is a leading qualifier, not a suffix.
        # Triggered when no trailing type was stripped (type_forms == [None]) but the
        # first word is itself a type. Generate all abbreviation variants of that type
        # × the remaining base name, with no separate trailing type.
        if (
            type_forms == [None]
            and len(name_words) > 1
            and name_words[0] in _TYPE_TO_ABBREVS
        ):
            leading_type_forms: list[str | None] = [None] + _TYPE_TO_ABBREVS[
                name_words[0]
            ]
            rest_name = " ".join(name_words[1:])
            for dir_form in dir_forms:
                for lt_form in leading_type_forms:
                    parts: list[str] = []
                    if dir_form is not None:
                        parts.append(dir_form)
                    if lt_form is not None:
                        parts.append(lt_form)
                    parts.append(rest_name)
                    result.add(" ".join(parts))
            continue

        base_name = " ".join(name_words)

        # CRAFT often splits a multi-word name (e.g. "VAN BRUNT") into one box per word,
        # especially on stretched or vertical labels. Add each individual name word so a
        # split box decodes to itself instead of snapping to the nearest unrelated vocab
        # word (e.g. "VAN" -> "IVAN", "DYKE" -> "DARE"); the split parts are recombined
        # later by georef_from_labels.assemble_multiword_streets. Words shorter than 3
        # characters are skipped to avoid adding ambiguous fragments.
        if len(name_words) >= 2:
            result.update(word for word in name_words if len(word) >= 3)

        # Also generate the numeric ordinal form so the CTC trie contains "S. 4TH ST"
        # alongside "S. FOURTH ST" — without this, the constrained decoder cannot
        # produce the abbreviated numeric label that appears on the map image.
        base_names = [base_name]
        if base_name in ORDINAL_WORD_TO_NUM:
            base_names.append(ORDINAL_WORD_TO_NUM[base_name])

        # Sanborn maps print ordinals with a typographic gap between the digit
        # and the suffix (e.g. "5 TH", not "5TH"). The CTC model sees that gap
        # as a space character, so "5TH" in the trie gets a near-zero path
        # probability (blank must pass through frames dominated by space).
        # Adding "5 TH" gives the decoder a matching path at ~1.0 probability.
        spaced = [
            f"{m.group(1)} {m.group(2)}"
            for bn in base_names
            if (m := _ORDINAL_RE.match(bn))
        ]
        base_names = base_names + [s for s in spaced if s not in base_names]

        # Cross-product of (direction prefix) × (base name) × (type suffix)
        for dir_form in dir_forms:
            for bn in base_names:
                for type_form in type_forms:
                    parts: list[str] = []
                    if dir_form is not None:
                        parts.append(dir_form)
                    parts.append(bn)
                    if type_form is not None:
                        parts.append(type_form)
                    result.add(" ".join(parts))

    # Add standalone hint strings unconditionally so the CTC decoder can output
    # bare type/direction words when CRAFT splits a label into separate boxes.
    result.update(HINT_STRINGS)

    return sorted(result)


# ---------------------------------------------------------------------------
# Trie (uppercase characters only)
# ---------------------------------------------------------------------------


@dataclass
class TrieNode:
    children: dict[str, "TrieNode"] = field(default_factory=dict)
    is_end: bool = False


def build_trie(strings: list[str]) -> TrieNode:
    """Build a character trie from a list of strings (stored uppercase)."""
    root = TrieNode()
    for s in strings:
        node = root
        for char in s.upper():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    return root


# ---------------------------------------------------------------------------
# Prefix-constrained CTC beam search
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def _build_char_projection(
    char_list_tuple: tuple[str, ...],
) -> tuple[dict[str, int], np.ndarray]:
    """Build a compact uppercase-char index and CTC→uppercase projection matrix.

    Cached so the O(C²) construction runs once per unique char_list rather than once
    per crop (1880+ times per image). The projection matrix ``ctc_to_uc`` has shape
    ``(C, n_uc)`` with a 1 in column k for every CTC index i whose character
    upper-cases to ``uc_chars[k]``.  A single ``mat @ ctc_to_uc`` replaces all the
    ``sum(p[i] for i in idxs)`` calls in the hot beam-search loop.
    """
    uc_chars: list[str] = []
    uc_char_to_idx: dict[str, int] = {}
    for ch in char_list_tuple[1:]:  # index 0 is the blank sentinel
        uc = ch.upper()
        if uc not in uc_char_to_idx:
            uc_char_to_idx[uc] = len(uc_chars)
            uc_chars.append(uc)
    n_uc = len(uc_chars)
    ctc_to_uc = np.zeros((len(char_list_tuple), n_uc), dtype=np.float64)
    for i, ch in enumerate(char_list_tuple):
        if i != 0:
            ctc_to_uc[i, uc_char_to_idx[ch.upper()]] = 1.0
    return uc_char_to_idx, ctc_to_uc


def prefix_constrained_ctc(
    mat: np.ndarray,
    trie_root: TrieNode,
    char_list: list[str],
    beam_width: int = 20,
) -> tuple[str, float]:
    """Prefix-constrained CTC beam search for a single detection.

    At each time step only extends beams via characters that are valid next nodes
    in the vocabulary trie, so the decoded output is always a prefix (or full member)
    of the vocabulary. The trie is case-insensitive: 'a' and 'A' both follow the 'A'
    branch, and their CTC probabilities are summed.

    Args:
        mat: ``(T, C)`` array of normalised CTC probabilities. Index 0 is blank.
        trie_root: Root of the vocabulary trie (built from uppercase strings).
        char_list: EasyOCR's character list: ``['[blank]', char1, char2, ...]``.
        beam_width: Maximum number of beams to keep after each time step.

    Returns:
        Tuple of (decoded string (uppercase), constrained path probability pb+pnb).
        Falls back to the highest-probability partial prefix if no complete vocabulary
        word is in the final beam. Returns ("", 0.0) if beams are empty.
    """
    T = mat.shape[0]
    blank_idx = 0

    # Precompute compact uppercase-char probabilities for every time step.
    #
    # Both 'A' (CTC index i) and 'a' (CTC index j) map to the same uppercase key 'A'.
    # _build_char_projection is @lru_cache'd so the projection matrix is built only
    # once per unique char_list across all crops. The single matmul replaces the
    # per-crop Python loop + tens of millions of sum(p[i] for i in idxs) calls.
    uc_char_to_idx, ctc_to_uc = _build_char_projection(tuple(char_list))
    compact_char_probs: np.ndarray = mat @ ctc_to_uc  # (T, n_uc)
    blank_probs = mat[:, blank_idx]

    # Beam entries: prefix_str → (trie_node, last_uc, pb, pnb)
    #   last_uc: uppercase version of the last emitted character (for repeat-char rule)
    #   pb: sum of CTC-path probs that decode to this prefix and end with blank
    #   pnb: sum of CTC-path probs that decode to this prefix and end with non-blank
    beams: dict[str, tuple[TrieNode, str, float, float]] = {
        "": (trie_root, "", 1.0, 0.0)
    }

    for t in range(T):
        t_probs = compact_char_probs[t]  # (n_uc,) view; O(1) slice, no copy
        blank_p = float(blank_probs[t])
        new_beams: dict[str, tuple[TrieNode, str, float, float]] = {}

        for prefix, (node, last_uc, pb, pnb) in beams.items():
            pr_total = pb + pnb

            # Emit blank — prefix and trie node unchanged.
            blank_prob = blank_p * pr_total
            prev = new_beams.get(prefix)
            if prev is None:
                new_beams[prefix] = (node, last_uc, blank_prob, 0.0)
            else:
                new_beams[prefix] = (prev[0], prev[1], prev[2] + blank_prob, prev[3])

            # Repeat last char — decoded prefix unchanged (CTC merge rule).
            # Probability comes only from non-blank-ending paths.
            if last_uc:
                last_ci = uc_char_to_idx.get(last_uc)
                if last_ci is not None:
                    repeat_p = pnb * float(t_probs[last_ci])
                    if repeat_p > 0:
                        prev = new_beams[prefix]
                        new_beams[prefix] = (
                            prev[0],
                            prev[1],
                            prev[2],
                            prev[3] + repeat_p,
                        )

            # Extend with each trie-valid character.
            for uc, child_node in node.children.items():
                ci = uc_char_to_idx.get(uc)
                if ci is None:
                    continue  # char not in EasyOCR alphabet
                char_p = float(t_probs[ci])
                if char_p == 0:
                    continue

                if uc == last_uc:
                    # Same char as last non-blank: only blank-ending paths may emit it
                    # (non-blank paths would merely repeat, handled above).
                    extend_p = pb * char_p
                else:
                    extend_p = pr_total * char_p

                new_prefix = prefix + uc
                prev = new_beams.get(new_prefix)
                if prev is None:
                    new_beams[new_prefix] = (child_node, uc, 0.0, extend_p)
                else:
                    new_beams[new_prefix] = (
                        prev[0],
                        prev[1],
                        prev[2],
                        prev[3] + extend_p,
                    )

        # Prune to beam_width by total probability.
        if len(new_beams) > beam_width:
            beams = dict(
                sorted(
                    new_beams.items(),
                    key=lambda kv: kv[1][2] + kv[1][3],
                    reverse=True,
                )[:beam_width]
            )
        else:
            beams = new_beams

    if not beams:
        return "", 0.0

    # Only return complete vocabulary entries (is_end=True).
    # Partial prefix fallbacks would cause non-street fragments to appear
    # as high-confidence matches because many streets share a common prefix.
    complete = [(pfx, b[2] + b[3]) for pfx, b in beams.items() if b[0].is_end]
    if not complete:
        return "", 0.0
    best_pfx, best_prob = max(complete, key=lambda x: x[1])
    return best_pfx, best_prob


# ---------------------------------------------------------------------------
# Monkey-patch
# ---------------------------------------------------------------------------

# Blank probability threshold above which a CTC frame is treated as padding.
# After batching, EasyOCR pads shorter crops to the longest crop's T.
# Padding frames have near-100% blank probability (black pixels → near-zero
# CNN activations → LSTM output dominated by blank). Frames at or above this
# threshold are trimmed before beam search to restore per-crop effective T.
_PADDING_BLANK_THRESHOLD = 0.9999


def patch_easyocr_reader(
    reader,
    vocab_strings: list[str],
    beam_width: int = 20,
) -> TrieNode:
    """Patch EasyOCR for vocabulary-constrained CTC decoding and batched GPU→CPU transfers.

    Two patches are applied:

    1. ``easyocr.recognition.recognizer_predict`` — replaces text decoding with
       prefix-constrained CTC beam search (vocabulary trie) and derives confidence
       from the constrained path probability. For other decoders the original is called.

    2. ``easyocr.Reader.recognize`` — replaces the ``batch_size=1`` one-per-bbox loop
       with a group-by-T loop: bboxes with the same natural sequence length (max_width)
       are batched together, so each distinct T value produces one LSTM forward pass and
       one GPU→CPU transfer instead of one per bbox. This reduces ~1886 CUDA syncs to
       ~10–20 for a typical Sanborn map page (one per distinct crop aspect ratio).

    Both patches are module-/class-level; creating a new ``Reader`` does NOT revert them.

    Returns the trie root (for inspection / debugging).
    """
    import easyocr
    import easyocr.easyocr as _easyocr_mod
    import easyocr.recognition as _recog
    import torch
    import torch.nn.functional as F

    trie_root = build_trie(vocab_strings)
    char_list: list[str] = reader.converter.character

    original_predict = _recog.recognizer_predict

    def _patched_recognizer_predict(
        model,
        converter,
        test_loader,
        batch_max_length,
        ignore_idx,
        char_group_idx,
        decoder: str = "greedy",
        beamWidth: int = 5,  # ignored; the outer beam_width closure is used instead
        device: str = "cpu",
    ) -> list[list]:
        if decoder != "wordbeamsearch":
            return original_predict(
                model,
                converter,
                test_loader,
                batch_max_length,
                ignore_idx,
                char_group_idx,
                decoder,
                beamWidth,
                device,
            )

        model.eval()

        # Phase 1: all GPU forward passes — keep preds_prob on GPU.
        # Ignore-idx zeroing and renormalization happen on GPU to avoid the
        # extra GPU→CPU→GPU roundtrip that the original EasyOCR code does.
        all_preds: list[torch.Tensor] = []
        with torch.no_grad():
            for image_tensors in test_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)
                text_for_pred = (
                    torch.LongTensor(batch_size, batch_max_length + 1)
                    .fill_(0)
                    .to(device)
                )
                preds = model(image, text_for_pred)
                preds_prob = F.softmax(preds, dim=2)
                if ignore_idx:
                    preds_prob[:, :, ignore_idx] = 0.0
                pred_norm = preds_prob.sum(dim=2, keepdim=True).clamp(min=1e-9)
                all_preds.append(preds_prob / pred_norm)

        if not all_preds:
            return []

        # Phase 2: one GPU→CPU transfer for all batches combined.
        # (N_total, T_global, C) where T_global = batch_max_length + 1.
        all_probs: np.ndarray = torch.cat(all_preds, dim=0).cpu().numpy()  # type: ignore[reportPrivateImportUsage]

        # Phase 3: CTC beam search with per-crop T trimming.
        # When called with batch_size > 1, EasyOCR pads all crops to the
        # widest crop's T_global. Padding frames have near-100% blank
        # probability (black-pixel padding → near-zero CNN features → LSTM
        # outputs dominated by blank). Trimming those frames restores the
        # per-crop effective T, keeping CTC beam search as fast as it was
        # in the batch_size=1 path.
        T_global = all_probs.shape[1]
        result: list[list] = []
        for i in range(len(all_probs)):
            crop = all_probs[i]  # (T_global, C)
            # Scan backwards to find the last non-padding frame.
            effective_T = T_global
            for t in range(T_global - 1, 0, -1):
                if crop[t, 0] >= _PADDING_BLANK_THRESHOLD:
                    effective_T = t
                else:
                    break
            text, path_prob = prefix_constrained_ctc(
                crop[:effective_T], trie_root, char_list, beam_width
            )
            # Per-character geometric mean of the constrained path probability.
            # Analogous to EasyOCR's custom_mean but derived from the actual
            # constrained path rather than greedy max probs.
            confidence = path_prob ** (1.0 / max(len(text), 1))
            result.append([text, float(confidence)])

        return result

    _recog.recognizer_predict = _patched_recognizer_predict

    # --- Patch 2: Reader.recognize --- group bboxes by T to reduce CUDA syncs ---

    from collections import defaultdict

    import easyocr.utils as _eu
    from easyocr.recognition import get_text as _get_text

    original_recognize = easyocr.Reader.recognize

    def _patched_recognize(
        self,
        img_cv_grey,
        horizontal_list=None,
        free_list=None,
        decoder="greedy",
        beamWidth=5,
        batch_size=1,
        workers=0,
        allowlist=None,
        blocklist=None,
        detail=1,
        rotation_info=None,
        paragraph=False,
        contrast_ths=0.1,
        adjust_contrast=0.5,
        filter_ths=0.003,
        y_ths=0.5,
        x_ths=1.0,
        reformat=True,
        output_format="standard",
    ):
        # Only optimize the GPU wordbeamsearch path without rotation.
        # Fall back to original for CPU (where CUDA sync overhead doesn't exist),
        # non-wordbeamsearch decoders, and rotation_info paths.
        # Note: batch_size is intentionally ignored — all crops sharing the same T
        # are batched together regardless of the caller's batch_size value.
        if decoder != "wordbeamsearch" or rotation_info or self.device == "cpu":
            return original_recognize(
                self,
                img_cv_grey,
                horizontal_list,
                free_list,
                decoder,
                beamWidth,
                batch_size,
                workers,
                allowlist,
                blocklist,
                detail,
                rotation_info,
                paragraph,
                contrast_ths,
                adjust_contrast,
                filter_ths,
                y_ths,
                x_ths,
                reformat,
                output_format,
            )

        # Replicate recognize()'s standard preprocessing.
        if reformat:
            _, img_cv_grey = _eu.reformat_input(img_cv_grey)
            assert img_cv_grey is not None

        if allowlist:
            ignore_char = "".join(set(self.character) - set(allowlist))
        elif blocklist:
            ignore_char = "".join(set(blocklist))
        else:
            ignore_char = "".join(set(self.character) - set(self.lang_char))

        if horizontal_list is None and free_list is None:
            y_max, x_max = img_cv_grey.shape
            horizontal_list = [[0, x_max, 0, y_max]]
            free_list = []

        img_h: int = _easyocr_mod.imgH

        # Group bboxes by max_width (= each crop's natural sequence length T×10).
        # Crops with the same max_width share a T value so they can go through
        # the LSTM together without any padding overhead. Each group produces one
        # LSTM forward pass and one GPU→CPU transfer instead of one per bbox.
        groups: dict[int, list] = defaultdict(list)
        for bbox in horizontal_list or []:
            image_list, max_width = _eu.get_image_list(
                [bbox], [], img_cv_grey, model_height=img_h
            )
            if image_list:
                groups[int(max_width)].extend(image_list)
        for bbox in free_list or []:
            image_list, max_width = _eu.get_image_list(
                [], [bbox], img_cv_grey, model_height=img_h
            )
            if image_list:
                groups[int(max_width)].extend(image_list)

        result = []
        for max_width, image_list in groups.items():
            group_result = _get_text(
                self.character,
                img_h,
                max_width,
                self.recognizer,
                self.converter,
                image_list,
                ignore_char,
                decoder,
                beamWidth,
                len(image_list),  # one LSTM call for the whole T group
                contrast_ths,
                adjust_contrast,
                filter_ths,
                workers,
                self.device,
            )
            result.extend(group_result)

        if paragraph:
            result = _eu.get_paragraph(result, x_ths=x_ths, y_ths=y_ths, mode="ltr")  # type: ignore[reportArgumentType]

        assert detail != 2, "detail=2 is not supported by the batched recognize patch"
        if detail == 0:
            return [item[1] for item in result]
        elif output_format == "dict":
            if paragraph:
                return [{"boxes": item[0], "text": item[1]} for item in result]
            return [
                {"boxes": item[0], "text": item[1], "confident": item[2]}
                for item in result
            ]
        elif output_format == "json":
            import json as _json

            if paragraph:
                return [
                    _json.dumps(
                        {
                            "boxes": [list(map(int, lst)) for lst in item[0]],
                            "text": item[1],
                        },
                        ensure_ascii=False,
                    )
                    for item in result
                ]
            return [
                _json.dumps(
                    {
                        "boxes": [list(map(int, lst)) for lst in item[0]],
                        "text": item[1],
                        "confident": item[2],
                    },
                    ensure_ascii=False,
                )
                for item in result
            ]
        elif output_format == "free_merge":
            return _eu.merge_to_free(result, free_list)
        else:
            return result

    easyocr.Reader.recognize = _patched_recognize

    return trie_root
