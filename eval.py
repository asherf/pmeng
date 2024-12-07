import logging
from langsmith import traceable, evaluate
from prompts import CODE_FILE_SUMMARIZER_PROMPTS, get_prompt, client, MAIN_MODEL, MODEL_TEMPERATURE

_logger = logging.getLogger(__name__)

DATASET_NAME = "asher-pmeng"
EXPERIMENT_PREFIX = "Asher pmeng Unit Tests"


@traceable
def code_file_summarizer_agent(inputs: dict) -> dict:
    messages = [get_prompt(), *inputs["messages"]]
    result = client.chat.completions.create(model=MAIN_MODEL, messages=messages, temperature=MODEL_TEMPERATURE)
    return {"message": {"role": "assistant", "content": result.choices[0].message.content}}


MAX_SCORE = 3


@traceable
def correctness_evaluator(run, example) -> dict:
    # Extract the original LeetCode problem from inputs
    source_file_request = run.inputs["inputs"]["messages"][-1]["content"]
    explanation = run.outputs["message"]["content"]

    # Rest of the evaluation logic remains the same
    evaluation_prompt = f"""
    Given this source file:
    {source_file_request}

    Evaluate the explanation:
    {explanation}

    Score from 0-{MAX_SCORE}:
    3 = The explanation is correct and complete.
    2 = The explanation is mostly correct and complete, but some details are missing.
    1 = The explanation is partially correct.
    0 = The explanation is incorrect or irrelevant.

    Return only the number (0-{MAX_SCORE}).
    """

    response = client.chat.completions.create(
        model=MAIN_MODEL,
        messages=[
            {"role": "system", "content": "You are a test evaluation assistant. Respond only with a number 0-3."},
            {"role": "user", "content": evaluation_prompt},
        ],
        temperature=0,
    )

    try:
        score = int(response.choices[0].message.content.strip())
        return {
            "key": "correctness score",
            "score": score / MAX_SCORE,  # Normalize to 0-1
            "explanation": f"Explanation correctness score: {score}/{MAX_SCORE}",
        }
    except ValueError:
        return {"key": "correctness score", "score": 0, "explanation": "Failed to parse score"}


evaluators = [correctness_evaluator]


results = evaluate(
    code_file_summarizer_agent, data=DATASET_NAME, evaluators=evaluators, experiment_prefix=EXPERIMENT_PREFIX
)
