import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from hot_topic.ad_prompt import (
    generate_chunk_responses,
    merge_spans,
    parse_json_object,
    validate_spans,
)


class ParseJsonObjectTests(unittest.TestCase):
    def test_assumes_no_spans_for_valid_json_without_spans_field(self):
        response = 'model output {"answer": "no ads"}'
        payload = parse_json_object(response, lambda text: text)

        self.assertEqual(payload, {"spans": []})

    def test_returns_none_for_invalid_json(self):
        response = 'model output {"start": 2, "end": 4'

        payload = parse_json_object(response, lambda text: text)

        self.assertIsNone(payload)

    def test_prefers_a_spans_dictionary(self):
        response = '{"start": 2, "end": 4, "reason": "fallback"} {"spans": []}'

        payload = parse_json_object(response, lambda text: text)

        self.assertEqual(payload, {"spans": []})


class SpanValidationTests(unittest.TestCase):
    def test_validates_closed_spans_and_truncates_end_at_chunk_boundary(self):
        payload = {"spans": [{"start": 1, "end": 10, "reason": "sponsor"}]}

        spans = validate_spans(payload, 4, 20, Path("response.txt"))

        self.assertEqual(spans, [{
            "start": 21,
            "end": 23,
            "reason": "sponsor",
        }])

    def test_merge_spans_merges_adjacent_closed_spans(self):
        spans = merge_spans([
            {"start": 3, "end": 4, "reason": "second"},
            {"start": 1, "end": 2, "reason": "first"},
        ])

        self.assertEqual(spans, [{"start": 1, "end": 4, "reason": "first; second"}])


class EmptyResponseTests(unittest.TestCase):
    def test_retries_empty_chunk_by_splitting_at_line_boundary(self):
        rows = [{"text": f"line {index}"} for index in range(4)]

        async def generate(
            _client, _model, chunk, _max_new_tokens, _reasoning_effort, _request_id
        ):
            if len(chunk) == 4 or (len(chunk) == 2 and chunk[0]["text"] == "line 0"):
                return ""
            return '{"spans": []}'

        with (
            patch("hot_topic.ad_prompt.generate_text", side_effect=generate) as mocked,
            self.assertLogs("hot_topic.ad_prompt", level="DEBUG") as logs,
        ):
            responses = asyncio.run(
                generate_chunk_responses(
                    None, "model", 10, rows, 100, "low", Path("episode.csv")
                )
            )

        self.assertEqual(
            responses,
            [],
        )
        self.assertEqual(mocked.call_count, 5)
        self.assertEqual(
            logs.output,
            [
                "WARNING:hot_topic.ad_prompt:Splitting empty-response chunk for "
                "transcript episode.csv at line 12",
                "WARNING:hot_topic.ad_prompt:Splitting empty-response chunk for "
                "transcript episode.csv at line 11",
            ],
        )

    def test_retries_invalid_json_chunk_by_splitting_at_line_boundary(self):
        rows = [{"text": f"line {index}"} for index in range(4)]

        async def generate(
            _client, _model, chunk, _max_new_tokens, _reasoning_effort, _request_id
        ):
            if len(chunk) == 4:
                return '{"spans":'
            return '{"spans": []}'

        with (
            patch("hot_topic.ad_prompt.generate_text", side_effect=generate),
            self.assertLogs("hot_topic.ad_prompt", level="WARNING") as logs,
        ):
            responses = asyncio.run(
                generate_chunk_responses(
                    None, "model", 10, rows, 100, "low", Path("episode.csv")
                )
            )

        self.assertEqual(
            responses,
            [],
        )
        self.assertEqual(
            logs.output,
            [
                "WARNING:hot_topic.ad_prompt:Splitting missing-JSON chunk for "
                "transcript episode.csv at line 12",
            ],
        )


if __name__ == "__main__":
    unittest.main()