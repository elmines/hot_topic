import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from hot_topic.ad_prompt import (
    extract_json_object,
    generate_chunk_responses,
    merge_spans,
    validate_spans,
)


class ExtractJsonObjectTests(unittest.TestCase):
    def test_falls_back_to_span_dictionaries_in_any_field_order(self):
        response = (
            'model output {"reason": "sponsor", "end": 4, "start": 2} '
            '{"start": 8, "reason": "another", "end": 10}'
        )

        payload = extract_json_object(response, Path("response.txt"), lambda text: text)

        self.assertEqual(payload, {
            "spans": [
                {"start": 2, "end": 4, "reason": "sponsor"},
                {"start": 8, "end": 10, "reason": "another"},
            ],
        })

    def test_falls_back_to_an_incomplete_final_span_dictionary(self):
        response = 'model output {"reason": "sponsor", "start": 2, "end": 4'

        payload = extract_json_object(response, Path("response.txt"), lambda text: text)

        self.assertEqual(payload, {
            "spans": [{"start": 2, "end": 4, "reason": "sponsor"}],
        })

    def test_falls_back_to_a_complete_span_without_a_reason(self):
        response = 'model output {"end": 4, "start": 2}'

        payload = extract_json_object(response, Path("response.txt"), lambda text: text)

        self.assertEqual(payload, {
            "spans": [{"start": 2, "end": 4, "reason": ""}],
        })

    def test_allows_an_incomplete_final_span_without_a_reason(self):
        response = 'model output {"end": 4, "start": 2'

        payload = extract_json_object(response, Path("response.txt"), lambda text: text)

        self.assertEqual(payload, {
            "spans": [{"start": 2, "end": 4, "reason": ""}],
        })

    def test_prefers_a_spans_dictionary(self):
        response = '{"start": 2, "end": 4, "reason": "fallback"} {"spans": []}'

        payload = extract_json_object(response, Path("response.txt"), lambda text: text)

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
            [(offset, len(chunk), response) for offset, chunk, response in responses],
            [
                (10, 1, '{"spans": []}'),
                (11, 1, '{"spans": []}'),
                (12, 2, '{"spans": []}'),
            ],
        )
        self.assertEqual(mocked.call_count, 5)
        self.assertEqual(
            logs.output,
            [
                "DEBUG:hot_topic.ad_prompt:Splitting empty-response chunk for "
                "transcript episode.csv at line 12",
                "DEBUG:hot_topic.ad_prompt:Splitting empty-response chunk for "
                "transcript episode.csv at line 11",
            ],
        )


if __name__ == "__main__":
    unittest.main()