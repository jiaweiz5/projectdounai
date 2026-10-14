import asyncio
import json
import unittest

from comment_vision import CommentVisionError, extract_comments, parse_result


class CommentVisionTests(unittest.TestCase):
    def test_complete_comments_and_duplicates_survive(self):
        result = parse_result(json.dumps({"comments": [{"text": "第一行\n第二行"}, {"text": "重复"}, {"text": "重复"}], "lines": ["用户名", "第一行", "第二行"]}))
        self.assertEqual(result["comments"][0]["text"], "第一行\n第二行")
        self.assertEqual(len(result["comments"]), 3)
        self.assertEqual(result["lines"][0], {"text": "用户名"})

    def test_invalid_output_is_not_silently_empty(self):
        for content in ('{}', '{"comments":null,"lines":[]}', '{"comments":[{"text":123}],"lines":[]}', 'not json'):
            with self.assertRaises(CommentVisionError):
                parse_result(content)

    def test_no_comments_is_valid(self):
        self.assertEqual(parse_result('{"comments":[],"lines":[]}')["comments"], [])

    def test_invalid_image_rejected_before_network(self):
        with self.assertRaises(CommentVisionError):
            asyncio.run(extract_comments(b"not an image"))


if __name__ == "__main__":
    unittest.main()
