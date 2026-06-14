# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.completion.protocol import CompletionRequest


def _make_request(logprobs):
    return CompletionRequest.model_validate(
        {
            "model": "facebook/opt-125m",
            "prompt": "Hello",
            "logprobs": logprobs,
        }
    )


@pytest.mark.parametrize("logprobs", [0, 1, 5, -1])
def test_completion_logprobs_accepts_non_negative_and_all_sentinel(logprobs):
    # -1 is the documented sentinel meaning "return logprobs for the whole
    # vocab", mirroring the prompt_logprobs handling in the same validator.
    request = _make_request(logprobs)
    assert request.logprobs == logprobs


@pytest.mark.parametrize("logprobs", [-2, -10])
def test_completion_logprobs_rejects_other_negatives(logprobs):
    with pytest.raises(ValueError, match="must be a positive value or -1"):
        _make_request(logprobs)
