import asyncio
import unittest
from unittest.mock import patch

from pathlib import Path

from hot_topic.ad_prompt import (
    merge_spans,
    parse_json_object,
    validate_spans,
)
from hot_topic.llm_utils import generate_chunk_responses


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


def span_messages(chunk):
    return [{"role": "user", "content": "\n".join(row["text"] for row in chunk)}]


def parse_spans(text):
    return parse_json_object(text, lambda item: item)


class EmptyResponseTests(unittest.TestCase):
    def test_retries_empty_chunk_by_splitting_at_line_boundary(self):
        rows = [{"text": f"line {index}"} for index in range(4)]

        async def generate(
            _client, _model, messages, _max_new_tokens, _reasoning_effort, _request_id,
            request_semaphore=None,
        ):
            lines = messages[-1]["content"].splitlines()
            if len(lines) == 4 or (len(lines) == 2 and lines[0] == "line 0"):
                return ""
            return '{"spans": []}'

        with (
            patch("hot_topic.llm_utils.generate_text", side_effect=generate) as mocked,
            self.assertLogs("hot_topic.llm_utils", level="DEBUG") as logs,
        ):
            responses = asyncio.run(
                generate_chunk_responses(
                    None,
                    "model",
                    10,
                    rows,
                    100,
                    "low",
                    Path("episode.csv"),
                    span_messages,
                    parse_spans,
                )
            )

        self.assertEqual(responses, [
            (10, rows[:1], {"spans": []}),
            (11, rows[1:2], {"spans": []}),
            (12, rows[2:], {"spans": []}),
        ])
        self.assertEqual(mocked.call_count, 5)
        self.assertEqual(
            logs.output,
            [
                "WARNING:hot_topic.llm_utils:Splitting empty-response chunk for "
                "transcript episode.csv at line 12",
                "WARNING:hot_topic.llm_utils:Splitting empty-response chunk for "
                "transcript episode.csv at line 11",
            ],
        )

    def test_retries_invalid_json_chunk_by_splitting_at_line_boundary(self):
        rows = [{"text": f"line {index}"} for index in range(4)]

        async def generate(
            _client, _model, messages, _max_new_tokens, _reasoning_effort, _request_id,
            request_semaphore=None,
        ):
            lines = messages[-1]["content"].splitlines()
            if len(lines) == 4:
                return '{"spans":'
            return '{"spans": []}'

        with (
            patch("hot_topic.llm_utils.generate_text", side_effect=generate),
            self.assertLogs("hot_topic.llm_utils", level="WARNING") as logs,
        ):
            responses = asyncio.run(
                generate_chunk_responses(
                    None,
                    "model",
                    10,
                    rows,
                    100,
                    "low",
                    Path("episode.csv"),
                    span_messages,
                    parse_spans,
                )
            )

        self.assertEqual(responses, [
            (10, rows[:2], {"spans": []}),
            (12, rows[2:], {"spans": []}),
        ])
        self.assertEqual(
            logs.output,
            [
                "WARNING:hot_topic.llm_utils:Splitting invalid-JSON chunk for "
                "transcript episode.csv at line 12",
            ],
        )


if __name__ == "__main__":
    unittest.main()