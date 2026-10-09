"""Regression for the server's populated finish usage plus matching SSE trailer."""

import json
import unittest

from tools.smoke.serve_contract import ContractError, Response, parse_openai_stream


class FinishUsageRegression(unittest.TestCase):
    def setUp(self):
        self.usage = {
            "prompt_tokens": 9,
            "completion_tokens": 3,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 1},
        }
        self.role = self.choice({"role": "assistant"})
        self.content = self.choice({"content": "OK"})
        self.finish = self.choice({}, "stop", self.usage)
        self.trailer = {"choices": [], "usage": self.usage}

    @staticmethod
    def choice(delta, reason=None, usage=None):
        return {"choices": [{"index": 0, "delta": delta, "finish_reason": reason}], "usage": usage}

    @staticmethod
    def stream(events):
        body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        return Response(200, "text/event-stream", (body + "data: [DONE]\n\n").encode())

    def test_populated_finish_and_matching_trailer(self):
        result = parse_openai_stream(
            self.stream(
                [
                    self.role,
                    self.choice({"reasoning_content": "Think."}),
                    self.content,
                    self.finish,
                    self.trailer,
                ]
            )
        )
        self.assertEqual(result, ("OK", "Think.", "stop", self.usage))

    def test_trailer_only_usage_still_supported(self):
        result = parse_openai_stream(
            self.stream([self.role, self.content, self.choice({}, "stop"), self.trailer])
        )
        self.assertEqual(result[3], self.usage)

    def test_reject_premature_contradictory_or_duplicate_usage(self):
        contradictory = {"choices": [], "usage": {**self.usage, "total_tokens": 13}}
        cases = [
            [
                self.role,
                self.choice({"content": "OK"}, usage=self.usage),
                self.finish,
                self.trailer,
            ],
            [self.role, self.trailer, self.finish],
            [self.role, self.content, self.finish, contradictory],
            [self.role, self.content, self.finish, self.trailer, self.trailer],
            [self.role, self.content, self.finish, self.finish, self.trailer],
            [self.role, self.content, self.finish],
        ]
        for events in cases:
            with self.subTest(events=events), self.assertRaises(ContractError):
                parse_openai_stream(self.stream(events))


if __name__ == "__main__":
    unittest.main()
