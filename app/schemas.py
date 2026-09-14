from typing import Literal
from urllib.parse import urlsplit
import ipaddress

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Connection(StrictModel):
    base_url: str = Field(default="", max_length=500)
    model: str = Field(default="", max_length=200)
    timeout_seconds: int = Field(default=120, ge=5, le=600)
    num_ctx: int = Field(default=8192, ge=1024, le=32768)
    num_predict: int = Field(default=512, ge=32, le=2048)
    temperature: float = Field(default=0.4, ge=0, le=1.5)

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        if not value:
            return value
        u = urlsplit(value)
        if (u.scheme not in {"http", "https"} or not u.hostname
                or u.username or u.password or u.query or u.fragment
                or u.path not in {"", "/"}):
            raise ValueError("Нужен базовый http(s) URL без пароля, пути и параметров")
        _ = u.port  # also validates malformed ports
        if u.hostname.lower() in {"metadata.google.internal", "metadata"}:
            raise ValueError("Этот адрес не разрешён")
        try:
            address = ipaddress.ip_address(u.hostname)
        except ValueError:
            address = None
        if address and (address.is_link_local or address.is_multicast or address.is_unspecified):
            raise ValueError("Этот адрес не разрешён")
        return value.rstrip("/")


class PromptCreate(StrictModel):
    title: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=16000)


class Catalog(StrictModel):
    content: str = Field(default="", max_length=24000)


class Message(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=16000)


class Context(StrictModel):
    manager_notified: bool = False


class ReplyRequest(StrictModel):
    request_id: str = Field(min_length=8, max_length=128, pattern=r"^[a-zA-Z0-9_.:-]+$")
    chat_id: str = Field(min_length=1, max_length=128)
    messages: list[Message] = Field(min_length=1, max_length=100)
    mode: Literal["assistant", "fallback"] = "assistant"
    context: Context = Field(default_factory=Context)

    @model_validator(mode="after")
    def check_messages(self):
        if self.messages[-1].role != "user":
            raise ValueError("Последнее сообщение должно быть от клиента")
        if sum(len(m.content) for m in self.messages) > 24000:
            raise ValueError("История слишком длинная: максимум 24000 символов")
        return self


class TestRequest(ReplyRequest):
    prompt_id: int | None = Field(default=None, gt=0)


class ModelAnswer(StrictModel):
    reply: str = Field(max_length=12000)
    action: Literal["reply", "handoff", "no_reply"]
    reason: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def check_answer(self):
        if self.action == "reply" and not self.reply:
            raise ValueError("Пустой ответ")
        if self.action == "no_reply" and self.reply:
            raise ValueError("no_reply должен содержать пустой reply")
        return self


class ExampleCreate(StrictModel):
    source_request_id: str = Field(min_length=8, max_length=128)
    answer: ModelAnswer
    note: str = Field(default="", max_length=2000)


class ExamplePatch(StrictModel):
    answer: ModelAnswer
    note: str = Field(default="", max_length=2000)
