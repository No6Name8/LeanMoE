"""IncrementalDecoder (src/engine/runtime/text_stream.py): streaming text never splits a character.

    python -m unittest discover -s tests -v

LeanMoE tests. Standard library only. A fake byte-level tokenizer (one token
per byte, decoded with errors="replace", like a byte-level BPE tokenizer cut
in the middle of a character) stands in for the real one.
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.runtime.text_stream import REPLACEMENT, IncrementalDecoder  # noqa: E402


class ByteTokenizer:
    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


def stream(data: bytes) -> tuple[list[str], str, IncrementalDecoder]:
    d = IncrementalDecoder(ByteTokenizer())
    deltas = [d.push(b) for b in data]
    return deltas, d.finish(), d


class IncrementalDecoderTest(unittest.TestCase):
    def test_ascii_is_released_immediately(self) -> None:
        deltas, tail, d = stream(b"x = 2")
        self.assertEqual(deltas, list("x = 2"))
        self.assertEqual(tail, "")
        self.assertEqual(d.released, "x = 2")

    def test_two_byte_character_waits_for_its_second_byte(self) -> None:
        deltas, tail, _ = stream("é".encode())
        self.assertEqual(deltas, ["", "é"])
        self.assertEqual(tail, "")

    def test_arabic_streams_without_replacement_characters(self) -> None:
        text = "عاصمة اليابان طوكيو"
        deltas, tail, d = stream(text.encode())
        self.assertTrue(all(REPLACEMENT not in x for x in deltas))
        self.assertEqual("".join(deltas) + tail, text)
        self.assertEqual(d.released, text)

    def test_four_byte_character(self) -> None:
        deltas, tail, _ = stream("∫ 🙂".encode())
        self.assertEqual("".join(deltas) + tail, "∫ 🙂")
        self.assertEqual(deltas[-4:], ["", "", "", "🙂"])

    def test_random_splits_of_mixed_text(self) -> None:
        text = "Solve $x^2 - 5x + 6 = 0$: x = 2 or x = 3. الجواب ٢ و ٣. ∑ 🙂 end"
        data = text.encode()
        rng = random.Random(7)
        for _ in range(50):
            cuts = sorted(rng.sample(range(1, len(data)), k=rng.randint(1, 20)))
            pieces = [data[a:b] for a, b in zip([0, *cuts], [*cuts, len(data)])]
            d = IncrementalDecoder(ByteTokenizer())
            deltas = []
            for piece in pieces:  # a "token" may now hold several bytes
                d.ids.extend(piece[:-1])
                deltas.append(d.push(piece[-1]))
            out = "".join(deltas) + d.finish()
            self.assertEqual(out, text)
            self.assertTrue(all(REPLACEMENT not in x for x in deltas))

    def test_dangling_bytes_are_released_by_finish(self) -> None:
        deltas, tail, d = stream(b"ok \xd8")  # a lead byte that never gets its continuation
        self.assertEqual("".join(deltas), "ok ")
        self.assertEqual(tail, REPLACEMENT)
        self.assertEqual(d.released, "ok " + REPLACEMENT)

    def test_invalid_byte_inside_text_is_shown_once_followed_by_text(self) -> None:
        deltas, tail, _ = stream(b"a\xffb")
        self.assertEqual("".join(deltas) + tail, "a" + REPLACEMENT + "b")


if __name__ == "__main__":
    unittest.main()
