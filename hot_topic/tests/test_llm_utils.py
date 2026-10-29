import argparse
import unittest
from unittest.mock import patch

from hot_topic.llm_utils import (
    add_vllm_client_arguments,
    configure_server,
    server_models,
)


class ServerModelTests(unittest.TestCase):
    def make_parser(self):
        parser = argparse.ArgumentParser()
        add_vllm_client_arguments(parser)
        return parser

    def make_args(self, model=None):
        args = self.make_parser().parse_args(
            ["--model", model] if model is not None else []
        )
        return args

    @patch(
        "hot_topic.llm_utils.server_request",
        return_value={"data": [{"id": "only-model"}]},
    )
    def test_selects_the_only_server_model_when_unspecified(self, _request):
        args = self.make_args()

        configure_server(self.make_parser(), args)

        self.assertEqual(args.model, "only-model")

    @patch(
        "hot_topic.llm_utils.server_request",
        return_value={"data": [{"id": "first"}, {"id": "second"}]},
    )
    def test_requires_model_when_server_exposes_multiple_models(self, _request):
        args = self.make_args()

        with self.assertRaises(SystemExit):
            configure_server(self.make_parser(), args)

    @patch(
        "hot_topic.llm_utils.server_request",
        return_value={"data": [{"id": "first"}, {"id": "second"}]},
    )
    def test_accepts_requested_server_model(self, _request):
        args = self.make_args("second")

        configure_server(self.make_parser(), args)

        self.assertEqual(args.model, "second")

    @patch(
        "hot_topic.llm_utils.server_request",
        return_value={"data": [{"id": "only-model"}]},
    )
    def test_rejects_model_not_exposed_by_server(self, _request):
        args = self.make_args("missing")

        with self.assertRaises(SystemExit):
            configure_server(self.make_parser(), args)

    @patch(
        "hot_topic.llm_utils.server_request",
        return_value={"data": [{"id": "first"}, {"id": "second"}]},
    )
    def test_returns_all_server_models(self, _request):
        self.assertEqual(server_models("http://server", "key"), ["first", "second"])


if __name__ == "__main__":
    unittest.main()
