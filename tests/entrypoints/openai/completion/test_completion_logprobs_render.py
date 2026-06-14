# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.logprobs import Logprob


def _render(num_output_top_logprobs):
    # Build a serving instance without running __init__ (we only exercise the
    # pure-Python logprobs renderer, no engine/tokenizer required).
    serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
    serving.return_tokens_as_token_ids = False

    sampled_token_id = 10
    step_top_logprobs = {
        10: Logprob(logprob=-0.1),
        11: Logprob(logprob=-1.2),
        12: Logprob(logprob=-2.3),
    }
    result = serving._create_completion_logprobs(
        token_ids=[sampled_token_id],
        top_logprobs=[step_top_logprobs],
        num_output_top_logprobs=num_output_top_logprobs,
        tokenizer=None,
        return_as_token_id=True,
    )
    return result.top_logprobs[0]


def test_logprobs_minus_one_returns_all_entries():
    # -1 is the documented sentinel for "return all vocab_size logprobs".
    # The renderer must not drop every entry via the `>= i` cap.
    assert len(_render(-1)) == 3


def test_logprobs_positive_count_is_capped():
    # N keeps entries at indices 0..N (the OpenAI "+1" behavior).
    assert len(_render(1)) == 2
