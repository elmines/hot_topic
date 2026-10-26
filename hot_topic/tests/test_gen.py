import asyncio
import csv
import io
import json
import unittest
from collections import OrderedDict
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from hot_topic.gen import (
    SYSTEM_PROMPT,
    TOPIC_SCHEMA,
    ad_span_file,
    discover_batch,
    drop_ad_span_lines,
    incorporate_topics,
    load_ad_spans,
    messages_for_chunk,
    parse_topics_payload,
    prepare_transcript,
    process_episode,
    select_episode_topics,
    topic_prompt,
)
from hot_topic.llm_utils import generate_chunk_responses
from hot_topic.utils import align_quote


class TopicSchemaTests(unittest.TestCase):
    def test_system_prompt_includes_the_schema(self):
        self.assertIn(json.dumps(TOPIC_SCHEMA, indent=2), SYSTEM_PROMPT)
        self.assertEqual(TOPIC_SCHEMA["required"], ["topics"])

    def test_messages_include_topics_noise_and_numbered_lines(self):
        messages = messages_for_chunk(
            [{"text": "Hello there"}],
            "[1] politics : mentions legislation, elections, or regulation",
        )

        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        user = messages[1]["content"]
        self.assertIn("[0] Hello there", user)
        self.assertIn("[1] politics : mentions legislation, elections, or regulation", user)
        self.assertIn("agriculture", user)
        self.assertNotIn("{Topics}", user)
        self.assertNotIn("{Noise}", user)
        self.assertNotIn("{Document}", user)


class AdSpanFilterTests(unittest.TestCase):
    def test_closed_spans_remove_inclusive_lines(self):
        rows = [{"text": f"line {index}"} for index in range(4)]

        kept = drop_ad_span_lines(rows, [{"start": 1, "end": 2}, {"start": 0, "end": 0}])

        self.assertEqual([row["text"] for row in kept], ["line 3"])

    def test_span_filter_then_music_and_ad_regexes(self):
        rows = [
            {"text": "The senate passed a new election law."},
            {"text": "Brought to you by T-Mobile today."},
            {"text": "[MUSIC]"},
            {"text": "This episode is sponsored by shoes."},
            {"text": "Voters discussed the regulation afterward."},
        ]

        kept = prepare_transcript(rows, [{"start": 3, "end": 3}])

        self.assertEqual(
            [row["text"] for row in kept],
            [
                "The senate passed a new election law.",
                "Voters discussed the regulation afterward.",
            ],
        )

    def test_span_path_mirrors_the_transcript_relative_path(self):
        span = ad_span_file(
            "/data/transcripts",
            "/data/spans",
            "/data/transcripts/show/episode.csv",
        )

        self.assertEqual(span.name, "episode.json")
        self.assertEqual(span.parent.name, "show")

    def test_missing_span_file_warns_and_returns_no_spans(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript = root / "transcripts" / "show" / "episode.csv"
            buffer = io.StringIO()
            with redirect_stderr(buffer):
                spans = load_ad_spans(transcript, root / "transcripts", root / "spans", 1)

        self.assertEqual(spans, [])
        self.assertIn("no ad spans", buffer.getvalue())


class TopicSelectionTests(unittest.TestCase):
    def test_keeps_only_quotes_that_align_to_the_transcript(self):
        text = "The senate passed a new election law yesterday."
        mentions = [
            ("politics", "first description", "senate passed a new election law"),
            ("politics", "second description", "election law yesterday"),
            ("sports", "games", "unrelated basketball championship quote"),
        ]

        selected = select_episode_topics(mentions, text, 95)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][0], "politics")
        self.assertEqual(selected[0][1], "first description")
        self.assertIn(selected[0][2], text)
        self.assertIn("senate passed a new election law", selected[0][2])
        self.assertIsNone(align_quote("unrelated basketball championship quote", text, 95))

    def test_batch_keeps_the_first_description_and_records_every_quote(self):
        topic_to_desc = OrderedDict([("politics", "elections")])
        new_topics, quotes = incorporate_topics(topic_to_desc, [
            ("a.csv", [("farming", "crops", "q1"), ("politics", "ignored", "q2")]),
            ("b.csv", [("farming", "other", "q3"), ("shipping", "freight", "q4")]),
        ])

        self.assertEqual(new_topics, [["farming", "crops"], ["shipping", "freight"]])
        self.assertEqual(quotes, [
            ["a.csv", "farming", "q1"],
            ["a.csv", "politics", "q2"],
            ["b.csv", "farming", "q3"],
            ["b.csv", "shipping", "q4"],
        ])
        self.assertEqual(topic_to_desc["farming"], "crops")

    def test_prompt_updates_after_the_batch_not_during_it(self):
        topic_to_desc = OrderedDict([("politics", "elections")])
        topic_str = topic_prompt(topic_to_desc)
        seen = []

        async def process(path, prompt):
            seen.append((path, prompt))
            if path == "a.csv":
                return path, [("farming", "crops and livestock", "quote")]
            return path, [("shipping", "movement of goods", "quote")]

        updated, new_topics, _quotes = asyncio.run(
            discover_batch(process, topic_to_desc, topic_str, ["a.csv", "b.csv"])
        )

        self.assertEqual([prompt for _path, prompt in seen], [topic_str, topic_str])
        self.assertNotIn("farming", topic_str)
        self.assertIn("[1] farming : crops and livestock", updated)
        self.assertIn("[1] shipping : movement of goods", updated)
        self.assertEqual(
            [topic for topic, _description in new_topics],
            ["farming", "shipping"],
        )


class EmptyResponseTests(unittest.TestCase):
    def test_prompt_debug_overwrites_with_latest_user_prompt(self):
        rows = [{"text": "first line"}]
        with TemporaryDirectory() as temporary:
            debug_path = Path(temporary) / "prompt.txt"

            async def generate(_client, _model, messages, *_args, **_kwargs):
                return "{}"

            with patch("hot_topic.llm_utils.generate_text", side_effect=generate):
                asyncio.run(
                    generate_chunk_responses(
                        None,
                        "model",
                        0,
                        rows,
                        32,
                        "low",
                        Path("episode.csv"),
                        lambda chunk: [{"role": "user", "content": chunk[0]["text"]}],
                        parse_topics_payload,
                        prompt_debug_path=debug_path,
                    )
                )
                rows[0]["text"] = "second line"
                asyncio.run(
                    generate_chunk_responses(
                        None,
                        "model",
                        0,
                        rows,
                        32,
                        "low",
                        Path("episode.csv"),
                        lambda chunk: [{"role": "user", "content": chunk[0]["text"]}],
                        parse_topics_payload,
                        prompt_debug_path=debug_path,
                    )
                )

            self.assertEqual(debug_path.read_text(encoding="utf-8"), "second line")

    def test_empty_response_splits_and_empty_topic_object_does_not(self):
        rows = [{"text": f"line {index}"} for index in range(2)]

        async def generate(_client, _model, messages, *_args, **_kwargs):
            if "\n" in messages[-1]["content"]:
                return ""
            return "{}"

        def messages_for(chunk):
            return [{"role": "user", "content": "\n".join(row["text"] for row in chunk)}]

        with (
            patch("hot_topic.llm_utils.generate_text", side_effect=generate) as mocked,
            self.assertLogs("hot_topic.llm_utils", level="WARNING") as logs,
        ):
            responses = asyncio.run(
                generate_chunk_responses(
                    None,
                    "model",
                    0,
                    rows,
                    32,
                    "low",
                    Path("episode.csv"),
                    messages_for,
                    parse_topics_payload,
                )
            )

        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(responses, [
            (0, rows[:1], {"topics": []}),
            (1, rows[1:], {"topics": []}),
        ])
        self.assertIn("empty-response", logs.output[0])

        async def empty_object(*_args, **_kwargs):
            return "{}"

        with (
            patch("hot_topic.llm_utils.generate_text", side_effect=empty_object) as mocked,
            self.assertNoLogs("hot_topic.llm_utils", level="WARNING"),
        ):
            responses = asyncio.run(
                generate_chunk_responses(
                    None,
                    "model",
                    0,
                    rows,
                    32,
                    "low",
                    Path("episode.csv"),
                    messages_for,
                    parse_topics_payload,
                )
            )

        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(responses, [(0, rows, {"topics": []})])

    def test_invalid_json_splits(self):
        self.assertIsNone(parse_topics_payload("no json here"))
        self.assertEqual(parse_topics_payload("{}"), {"topics": []})
        self.assertEqual(parse_topics_payload('{"topics": []}'), {"topics": []})


class ProcessEpisodeTests(unittest.TestCase):
    def test_filters_ads_and_drops_unaligned_quotes(self):
        rows = [
            {"start": "0", "end": "1", "text": "The senate passed a new election law."},
            {"start": "1", "end": "2", "text": "Brought to you by T-Mobile today."},
            {"start": "2", "end": "3", "text": "[MUSIC]"},
            {"start": "3", "end": "4", "text": "This episode is sponsored by shoes."},
            {"start": "4", "end": "5", "text": "Voters discussed the regulation afterward."},
        ]
        payload = {"topics": [
            {
                "topic": "politics",
                "description": "mentions legislation, elections, or regulation",
                "quote": "senate passed a new election law",
            },
            {
                "topic": "shopping",
                "description": "retail purchases",
                "quote": "Brought to you by T-Mobile today",
            },
            {
                "topic": "civics",
                "description": "public decision making",
                "quote": "Voters discussed the regulation afterward",
            },
        ]}

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript = root / "transcripts" / "show" / "episode.csv"
            span = root / "spans" / "show" / "episode.json"
            transcript.parent.mkdir(parents=True)
            span.parent.mkdir(parents=True)
            with transcript.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["start", "end", "text"])
                writer.writeheader()
                writer.writerows(rows)
            span.write_text(json.dumps({
                "source": str(transcript),
                "line_count": len(rows),
                "spans": [{"start": 3, "end": 3, "reason": "sponsor"}],
            }), encoding="utf-8")
            captured = {}

            def fake_chunks(prepared, _renderer, _max_model_len, _max_new_tokens, _path, messages_for):
                captured["rows"] = list(prepared)
                captured["messages"] = messages_for(prepared)
                return [(0, list(prepared), None)]

            async def fake_generate(*_args, **_kwargs):
                return [(0, captured["rows"], payload)]

            with (
                patch("hot_topic.gen.transcript_chunks", side_effect=fake_chunks),
                patch("hot_topic.gen.generate_chunk_responses", side_effect=fake_generate),
            ):
                _path, matches = asyncio.run(process_episode(
                    None,
                    "model",
                    None,
                    transcript,
                    "[1] politics : mentions legislation, elections, or regulation",
                    32,
                    "low",
                    1024,
                    root / "transcripts",
                    root / "spans",
                    95,
                    None,
                ))

        self.assertEqual(
            [row["text"] for row in captured["rows"]],
            [
                "The senate passed a new election law.",
                "Voters discussed the regulation afterward.",
            ],
        )
        document = captured["messages"][1]["content"].split("[Document]", 1)[1]
        self.assertNotIn("T-Mobile", document)
        self.assertNotIn("sponsored by shoes", document)
        self.assertNotIn("[MUSIC]", document)
        self.assertEqual([topic for topic, _description, _quote in matches], ["politics", "civics"])
        self.assertTrue(all(quote in combine_kept(captured["rows"]) for _topic, _description, quote in matches))


def combine_kept(rows):
    return "".join(row["text"] for row in rows)


if __name__ == "__main__":
    unittest.main()
