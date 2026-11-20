from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal, Mapping


Role = Literal["fast", "strong"]
Policy = Literal["fixed", "feedback"]


@dataclass(frozen=True)
class Task:
    prompt: str
    difficulty: float


@dataclass(frozen=True)
class Route:
    model: Role
    score: float
    threshold: float


@dataclass(frozen=True)
class ThresholdRouter:
    threshold: float

    def __call__(self, task: Task) -> Route:
        if not 0 <= task.difficulty <= 1 or not 0 <= self.threshold <= 1:
            raise ValueError("Difficulty and threshold must be between 0 and 1.")
        model: Role = "strong" if task.difficulty >= self.threshold else "fast"
        return Route(model, task.difficulty, self.threshold)


@dataclass(frozen=True)
class Check:
    passed: bool
    code: str
    summary: str


@dataclass(frozen=True)
class Feedback:
    candidate: str | None
    check: Check


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class ModelReply:
    model: str
    text: str | None
    finish_reason: str | None = None
    usage: Usage = Usage()


class ModelError(Exception):
    """A transport or response error, distinct from a failed candidate check."""


Model = Callable[[Task, Feedback | None], ModelReply]


@dataclass(frozen=True)
class Attempt:
    role: Role
    model: str | None
    model_ms: float
    finish_reason: str | None
    usage: Usage | None
    check: Check | None
    decision: str
    reason: str


@dataclass(frozen=True)
class Run:
    route: Route
    policy: Policy
    attempts: tuple[Attempt, ...]
    status: Literal["passed", "failed", "model_error"]
    output: str | None
    elapsed_ms: float


def run_task(
    task: Task,
    *,
    router: Callable[[Task], Route],
    models: Mapping[Role, Model],
    verify: Callable[[str], Check],
    policy: Policy = "feedback",
    max_fast_attempts: int = 3,
) -> Run:
    if max_fast_attempts < 1:
        raise ValueError("The fast model must be allowed at least one attempt.")
    if policy not in ("fixed", "feedback"):
        raise ValueError(f"Unknown policy: {policy}")

    started = perf_counter()
    route = router(task)
    role = route.model
    attempts: list[Attempt] = []
    feedback = None
    fast_attempts = 0
    output = None
    status: Literal["passed", "failed", "model_error"] = "failed"

    while True:
        call_started = perf_counter()
        try:
            reply = models[role](task, feedback)
        except ModelError as error:
            model_ms = round((perf_counter() - call_started) * 1000, 3)
            attempts.append(
                Attempt(role, None, model_ms, None, None, None, "stop", str(error))
            )
            status = "model_error"
            break

        model_ms = round((perf_counter() - call_started) * 1000, 3)
        if reply.finish_reason not in (None, "stop", "length"):
            attempts.append(
                Attempt(
                    role, reply.model, model_ms, reply.finish_reason, reply.usage,
                    None, "stop", "unexpected_finish_reason",
                )
            )
            status = "model_error"
            break
        if reply.finish_reason == "length":
            check = Check(False, "output_truncated", "The model reached its output limit.")
        elif reply.text is None or not reply.text.strip():
            check = Check(False, "empty_output", "The model returned no candidate.")
        else:
            check = verify(reply.text)

        if role == "fast":
            fast_attempts += 1

        if check.passed:
            decision, reason = "accept", "verification_passed"
            status, output = "passed", reply.text
        elif role == "strong":
            decision, reason = "stop", "strong_verification_failed"
        elif fast_attempts >= max_fast_attempts:
            decision, reason = "escalate", "fast_attempt_limit"
        elif policy == "feedback" and feedback and feedback.check.code == check.code:
            decision, reason = "escalate", "repeated_failure"
        else:
            decision, reason = "retry", "fast_attempts_remaining"

        attempts.append(
            Attempt(role, reply.model, model_ms, reply.finish_reason, reply.usage, check, decision, reason)
        )
        if decision in ("accept", "stop"):
            break

        feedback = Feedback(reply.text, check)
        if decision == "escalate":
            role = "strong"

    elapsed_ms = round((perf_counter() - started) * 1000, 3)
    return Run(route, policy, tuple(attempts), status, output, elapsed_ms)
