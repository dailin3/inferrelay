import argparse
from dataclasses import asdict
import json
import os

from .core import Check, Feedback, Model, ModelReply, Role, Task, ThresholdRouter, run_task
from .models import ChatCompletionsModel


def verify_range_output(text: str) -> Check:
    try:
        candidate = json.loads(text)
    except ValueError:
        return Check(False, "invalid_json", "Return a JSON array of integers.")

    if not isinstance(candidate, list) or any(type(value) is not int for value in candidate):
        return Check(False, "invalid_shape", "Return a JSON array of integers.")
    expected = list(range(7, 0, -3))
    if candidate != expected:
        return Check(
            False,
            f"range_mismatch:{json.dumps(candidate)}",
            "The array differs from the Python range output.",
        )
    return Check(True, "matches", "The array matches the Python range output.")


def scripted_model(name: str, answer: str) -> Model:
    def generate(_task: Task, _feedback: Feedback | None) -> ModelReply:
        return ModelReply(name, answer, finish_reason="stop")

    return generate


def configured_model(role: Role) -> ChatCompletionsModel:
    prefix = f"AI4SE_{role.upper()}"
    return ChatCompletionsModel(
        base_url=os.environ[f"{prefix}_BASE_URL"],
        model=os.environ[f"{prefix}_MODEL"],
        api_key=os.environ[f"{prefix}_API_KEY"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the hybrid inference prototype.")
    parser.add_argument("--backend", choices=("demo", "api"), default="demo")
    parser.add_argument("--policy", choices=("fixed", "feedback"), default="feedback")
    parser.add_argument("--score", type=float, default=0.25, help="Example task difficulty.")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--max-fast-attempts", type=int, default=3)
    args = parser.parse_args()

    models: dict[Role, Model]
    if args.backend == "api":
        try:
            models = {"fast": configured_model("fast"), "strong": configured_model("strong")}
        except KeyError as error:
            parser.error(f"Missing environment variable: {error.args[0]}")
    else:
        models = {
            "fast": scripted_model("demo-fast", "[7, 4]"),
            "strong": scripted_model("demo-strong", "[7, 4, 1]"),
        }

    task = Task(
        "What does list(range(7, 0, -3)) return in Python? Reply with only a JSON array.",
        difficulty=args.score,
    )
    try:
        result = run_task(
            task,
            router=ThresholdRouter(args.threshold),
            models=models,
            verify=verify_range_output,
            policy=args.policy,
            max_fast_attempts=args.max_fast_attempts,
        )
    except ValueError as error:
        parser.error(str(error))

    print(json.dumps(asdict(result), indent=2))
    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
