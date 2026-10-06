"""Regression tests for Xiaohongshu screenshot comment extraction."""

import unittest

from screenshot_ocr import extract_comment_lines


def make_ocr_line(
    text,
    left,
    top,
    width=220,
    height=22,
    confidence=0.99,
):
    """Create one synthetic OCR line with screen coordinates."""

    return {
        "text": text,
        "confidence": confidence,
        "box": {
            "left": float(left),
            "top": float(top),
            "right": float(left + width),
            "bottom": float(top + height),
        },
    }


class ScreenshotCommentExtractionTests(unittest.TestCase):
    """Verify that only top-level comment bodies are returned."""

    def test_empty_ocr_input_returns_empty_list(self):
        """An empty screenshot result should produce no comments."""

        self.assertEqual(extract_comment_lines([]), [])

    def test_filters_headers_metadata_replies_and_image_text(self):
        """The parser should keep root comments and remove interface noise."""

        ocr_lines = [
            # Section header. It must not cause the first comment to disappear.
            make_ocr_line("27 comments", 20, 0),

            # First top-level comment.
            make_ocr_line("user_one", 70, 41),
            make_ocr_line("first real comment", 70, 66),
            make_ocr_line("2h ago California", 70, 94),
            make_ocr_line("Like Reply", 70, 119),

            # Second top-level comment with two wrapped body lines.
            make_ocr_line("user_two", 70, 170),
            make_ocr_line("multiline comment part one", 70, 195),
            make_ocr_line("part two", 70, 218),
            make_ocr_line("3h ago New York", 70, 247),
            make_ocr_line("Like 5 Reply", 70, 272),

            # Third top-level comment.
            make_ocr_line("user_three", 70, 325),
            make_ocr_line("third real comment", 70, 350),
            make_ocr_line("1d ago Texas", 70, 379),
            make_ocr_line("Like Reply", 70, 404),

            # An indented reply. Replies are displayed separately in a future
            # enhancement and must not enter the current Layer 3 score.
            make_ocr_line("reply_user", 110, 450),
            make_ocr_line("indented reply text", 110, 475),

            # OCR text found inside an attached product image.
            # This must not be treated as a top-level comment.
            make_ocr_line("NYX", 180, 540),
            make_ocr_line("butter", 185, 565),
            make_ocr_line("gloss", 189, 585),

            # Controls below the attached reply image.
            make_ocr_line("2d ago California", 110, 650),
            make_ocr_line("Like Reply", 111, 675),
            make_ocr_line("View 2 replies", 110, 700),

            # Fourth top-level comment after the image reply.
            make_ocr_line("user_four", 70, 755),
            make_ocr_line("fourth real comment", 70, 780),
            make_ocr_line("4h ago Washington", 70, 809),
            make_ocr_line("Like Reply", 70, 834),
        ]

        comments = extract_comment_lines(ocr_lines)
        extracted_texts = [
            comment["text"]
            for comment in comments
        ]

        # Only the four top-level comment bodies should remain.
        self.assertEqual(
            extracted_texts,
            [
                "first real comment",
                "multiline comment part onepart two",
                "third real comment",
                "fourth real comment",
            ],
        )

        # Explicitly verify that common false positives were removed.
        combined_text = "\n".join(extracted_texts)

        self.assertNotIn("27 comments", combined_text)
        self.assertNotIn("ago", combined_text)
        self.assertNotIn("Like", combined_text)
        self.assertNotIn("Reply", combined_text)
        self.assertNotIn("butter", combined_text)
        self.assertNotIn("gloss", combined_text)
        self.assertNotIn("indented reply", combined_text)


if __name__ == "__main__":
    # This allows the file to run directly as well as through discovery.
    unittest.main()
