from types import SimpleNamespace

import pytest

from src import llm_client


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def fake_response(content='{"ok":true}', finish_reason='stop'):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, refusal=None),
            finish_reason=finish_reason,
        )],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    )


def test_gpt_oss_uses_one_user_message_and_low_reasoning(monkeypatch):
    fake = FakeCompletions([fake_response()])
    client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    monkeypatch.setattr(llm_client, 'get_groq_client', lambda: client)
    monkeypatch.setattr(llm_client, 'get_model_name', lambda: 'openai/gpt-oss-20b')

    text, meta = llm_client.request_json_object(
        system_prompt='system rules', user_prompt='episode events', max_completion_tokens=1400
    )

    call = fake.calls[0]
    assert len(call['messages']) == 1
    assert call['messages'][0]['role'] == 'user'
    assert call['reasoning_effort'] == 'low'
    assert call['include_reasoning'] is False
    assert text == '{"ok":true}'
    assert meta['response_mode'] == 'json_object'


def test_falls_back_to_plain_text_on_json_validate_error(monkeypatch):
    error = Exception("Error code 400 json_validate_failed failed_generation: ''")
    fake = FakeCompletions([error, fake_response('```json\n{"ok":true}\n```')])
    client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    monkeypatch.setattr(llm_client, 'get_groq_client', lambda: client)
    monkeypatch.setattr(llm_client, 'get_model_name', lambda: 'openai/gpt-oss-20b')

    text, meta = llm_client.request_json_object(
        system_prompt='system rules', user_prompt='episode events', max_completion_tokens=1400
    )

    assert len(fake.calls) == 2
    assert 'response_format' not in fake.calls[1]
    assert text == '{"ok":true}'
    assert meta['response_mode'] == 'text_json_fallback'


def test_rejects_empty_content(monkeypatch):
    fake = FakeCompletions([fake_response(content='')])
    client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    monkeypatch.setattr(llm_client, 'get_groq_client', lambda: client)
    monkeypatch.setattr(llm_client, 'get_model_name', lambda: 'openai/gpt-oss-20b')

    with pytest.raises(RuntimeError, match='empty message content'):
        llm_client.request_json_object(
            system_prompt='rules', user_prompt='events', max_completion_tokens=1000
        )
