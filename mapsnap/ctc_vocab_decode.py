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

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Vocabulary generation: canonical names → abbreviated label forms
# ---------------------------------------------------------------------------

# Inverse of DIRECTION_ABBREVS: full direction → all forms that normalize to it.
_DIR_TO_ABBREVS: dict[str, list[str]] = {
    "NORTH": ["NORTH", "N", "N."],
    "SOUTH": ["SOUTH", "S", "S."],
    "EAST": ["EAST", "E", "E."],
    "WEST": ["WEST", "W", "W."],
    "NORTHEAST": ["NORTHEAST", "NE", "NE."],
    "NORTHWEST": ["NORTHWEST", "NW", "NW."],
    "SOUTHEAST": ["SOUTHEAST", "SE", "SE."],
    "SOUTHWEST": ["SOUTHWEST", "SW", "SW."],
}

# Inverse of STREET_ABBREVS: full type → all abbreviation forms.
_TYPE_TO_ABBREVS: dict[str, list[str]] = {
    "STREET": ["STREET", "ST"],
    "AVENUE": ["AVENUE", "AVE", "AV"],
    "BOULEVARD": ["BOULEVARD", "BLVD"],
    "DRIVE": ["DRIVE", "DR"],
    "PLACE": ["PLACE", "PL"],
    "COURT": ["COURT", "CT"],
    "ROAD": ["ROAD", "RD"],
    "LANE": ["LANE", "LN"],
    "TERRACE": ["TERRACE", "TER", "TERR"],
    "HIGHWAY": ["HIGHWAY", "HWY"],
    "PARKWAY": ["PARKWAY", "PKWY"],
    "CIRCLE": ["CIRCLE", "CIR"],
    "EXPRESSWAY": ["EXPRESSWAY", "EXPY"],
}


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
            continue

        base_name = " ".join(name_words)

        # Cross-product of (direction prefix) × (type suffix)
        for dir_form in dir_forms:
            for type_form in type_forms:
                parts: list[str] = []
                if dir_form is not None:
                    parts.append(dir_form)
                parts.append(base_name)
                if type_form is not None:
                    parts.append(type_form)
                result.add(" ".join(parts))

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

    # Map uppercase character → list of CTC class indices.
    # Both 'A' (index i) and 'a' (index j) map to key 'A'.
    uc_to_idxs: dict[str, list[int]] = {}
    for idx, ch in enumerate(char_list):
        if idx == 0:
            continue  # skip blank sentinel
        uc = ch.upper()
        uc_to_idxs.setdefault(uc, []).append(idx)

    # Beam entries: prefix_str → (trie_node, last_uc, pb, pnb)
    #   last_uc: uppercase version of the last emitted character (for repeat-char rule)
    #   pb: sum of CTC-path probs that decode to this prefix and end with blank
    #   pnb: sum of CTC-path probs that decode to this prefix and end with non-blank
    beams: dict[str, tuple[TrieNode, str, float, float]] = {
        "": (trie_root, "", 1.0, 0.0)
    }

    for t in range(T):
        p = mat[t]
        new_beams: dict[str, tuple[TrieNode, str, float, float]] = {}

        for prefix, (node, last_uc, pb, pnb) in beams.items():
            pr_total = pb + pnb

            # Emit blank — prefix and trie node unchanged.
            blank_prob = p[blank_idx] * pr_total
            prev = new_beams.get(prefix)
            if prev is None:
                new_beams[prefix] = (node, last_uc, blank_prob, 0.0)
            else:
                new_beams[prefix] = (prev[0], prev[1], prev[2] + blank_prob, prev[3])

            # Repeat last char — decoded prefix unchanged (CTC merge rule).
            # Probability comes only from non-blank-ending paths.
            if last_uc:
                idxs = uc_to_idxs.get(last_uc, [])
                repeat_p = pnb * sum(p[i] for i in idxs)
                if repeat_p > 0:
                    prev = new_beams[prefix]
                    new_beams[prefix] = (prev[0], prev[1], prev[2], prev[3] + repeat_p)

            # Extend with each trie-valid character.
            for uc, child_node in node.children.items():
                idxs = uc_to_idxs.get(uc, [])
                if not idxs:
                    continue  # char not in EasyOCR alphabet
                char_p = sum(p[i] for i in idxs)
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


def patch_easyocr_reader(
    reader,
    vocab_strings: list[str],
    beam_width: int = 20,
) -> TrieNode:
    """Patch easyocr.recognition.recognizer_predict with constrained CTC decoding.

    For ``decoder='wordbeamsearch'`` calls, replaces both the text decoding and the
    confidence score with values derived from the constrained CTC path probability,
    so false positives (non-street text forced to a vocabulary word) get low
    confidence rather than the greedy-decode confidence of the actual image content.

    For other decoders the original function is called unchanged.

    The patch is module-level; creating a new ``Reader`` does NOT revert it.

    Returns the trie root (for inspection / debugging).
    """
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
        beamWidth: int = 5,
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
        result: list[list] = []
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
                preds_prob_np = preds_prob.cpu().detach().numpy()

                # Zero-out ignored characters (same normalisation as original).
                preds_prob_np[:, :, ignore_idx] = 0.0
                pred_norm = preds_prob_np.sum(axis=2)
                preds_prob_np = preds_prob_np / np.expand_dims(pred_norm, axis=-1)

                for i in range(batch_size):
                    text, path_prob = prefix_constrained_ctc(
                        preds_prob_np[i], trie_root, char_list, beam_width
                    )
                    # Per-character geometric mean of the constrained path probability.
                    # Analogous to EasyOCR's custom_mean but derived from the actual
                    # constrained path rather than greedy max probs.
                    confidence = path_prob ** (1.0 / max(len(text), 1))
                    result.append([text, float(confidence)])

        return result

    _recog.recognizer_predict = _patched_recognizer_predict
    return trie_root
