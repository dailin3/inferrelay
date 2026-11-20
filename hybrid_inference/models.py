from dataclasses import dataclass
from http.client import HTTPException
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .core import Feedback, ModelError, ModelReply, Task, Usage


@dataclass(frozen=True)
class ChatCompletionsModel:
    base_url: str
    model: str
    api_key: str
    timeout_seconds: float = 60

    def __call__(self, task: Task, feedback: Feedback | None) -> ModelReply:
        messages = [{"role": "user", "content": task.prompt}]
        if feedback is not None:
            revision = {
                "previous_candidate": feedback.candidate,
                "verification": feedback.check.summary,
            }
            messages.append(
                {
                    "role": "user",
                    "content": "Revise the candidate for the original task using this result.\n"
                    + json.dumps(revision, ensure_ascii=False),
                }
            )

        request = Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps({"model": self.model, "messages": messages}).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.load(response)
        except HTTPError as error:
            error.close()
            raise ModelError(f"http_{error.code}") from error
        except (OSError, HTTPException) as error:
            raise ModelError("transport_error") from error
        except ValueError as error:
            raise ModelError("invalid_response") from error

        try:
            choice = payload["choices"][0]
            content = choice["message"].get("content")
            if content is not None and not isinstance(content, str):
                raise ModelError("invalid_response")
            usage = payload.get("usage") or {}
            completion_details = usage.get("completion_tokens_details") or {}
            return ModelReply(
                model=payload.get("model") or self.model,
                text=content,
                finish_reason=choice.get("finish_reason"),
                usage=Usage(
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    reasoning_tokens=completion_details.get("reasoning_tokens"),
                ),
            )
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise ModelError("invalid_response") from error
