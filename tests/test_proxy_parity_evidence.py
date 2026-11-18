"""Small persisted CC 2.1.280 records from the failing offline captures."""
import copy
import json
from pathlib import Path

import pytest

from agent_snapshot_resume.parity import content_reference
from agent_snapshot_resume.proxy.resume_proxy import check_content, rewrite

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/cc280_rendered_attachments.json').read_text())
IMAGE_MARKER = '[Image: original 3775x5101, displayed at 1480x2000. Multiply coordinates by 2.55 to map to original image.]'
RECOVERY = 'Your response above was cut off mid-stream. Resume directly from where it stops — no apology, no recap. If none of it survived, answer the request from the start.'


def user(content, **kw):
    return dict(type='user', message=dict(role='user', content=content), **kw)


@pytest.mark.parametrize('text', [IMAGE_MARKER, RECOVERY])
def test_persisted_companion_text_is_exact_and_single_use(text):
    records = [user('task'), user(text, isMeta=True, turnCompanion=True)]
    ref = content_reference(records)
    body = {'messages': [r['message'] for r in records]}
    assert check_content(body, ref, []) == []
    bad = copy.deepcopy(body)
    bad['messages'][1]['content'] = text[:-1] + '!'
    assert check_content(bad, ref, [])
    body['messages'].append(body['messages'][1])
    assert check_content(body, ref, [])


@pytest.mark.parametrize('kind', FIXTURE)
@pytest.mark.parametrize('wrapped', [False, True])
def test_recorded_reminder_is_reproduced_verbatim(kind, wrapped):
    record = FIXTURE[kind]
    recorded = record['rendered'][0]['content']
    text = recorded if wrapped else recorded.removeprefix('<system-reminder>\n').removesuffix('\n</system-reminder>')
    ref = content_reference([user('task'), record], {'fixed': [], 'slots': []})
    # Coalescing into a system turn is CC's observed resume rendering.
    body = {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'task'}]}, {'role': 'system', 'content': text}]}
    raw = copy.deepcopy(body)
    raw['messages'].append({'role': 'user', 'content': 'SENTINEL'})
    rewritten, _ = rewrite(raw, {'sentinel': 'SENTINEL', 'mode': 'same'})
    assert rewritten == body
    assert check_content(rewritten, ref, []) == []
    bad = copy.deepcopy(body)
    bad['messages'][1]['content'] = text.replace('Background shell', 'Background Shell').replace('changed on disk', 'changed on Disk')
    assert check_content(bad, ref, [])
    bad['messages'][1]['content'] = text + '\n' + text
    assert check_content(bad, ref, [])
    bad['messages'][1]['content'] = ' '.join(reversed(text.split(' ')))
    assert check_content(bad, ref, [])


def test_image_bytes_are_bound_inside_tool_result():
    content = [{'type': 'tool_result', 'tool_use_id': 'read-image', 'content': [
        {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'AAAA'}}]}]
    ref = content_reference([user(content), user(IMAGE_MARKER, isMeta=True, turnCompanion=True)])
    body = {'messages': [user(content)['message'], user(IMAGE_MARKER)['message']]}
    assert check_content(body, ref, []) == []
    changed = copy.deepcopy(body)
    changed['messages'][0]['content'][0]['content'][0]['source']['data'] = 'AAAB'
    assert check_content(changed, ref, [])


def test_renderer_evidence_is_position_bound():
    record = FIXTURE['background']
    ref = content_reference([record, user('one'), user('two'), user('three')], {'fixed': [], 'slots': []})
    body = {'messages': [user(t)['message'] for t in ('one', 'two', 'three')] + [
        {'role': 'system', 'content': record['rendered'][0]['content']}]}
    assert check_content(body, ref, [])


def test_interrupted_thinking_is_not_silently_ignored():
    records = [user('task'), {'type': 'assistant', 'message': {
        'role': 'assistant', 'id': 'interrupted', 'content': [
            {'type': 'thinking', 'thinking': 'partial response', 'signature': ''}]}},
        user(RECOVERY, isMeta=True, turnCompanion=True),
        {'type': 'assistant', 'message': {'role': 'assistant', 'id': 'retry', 'content': [
            {'type': 'text', 'text': 'completed response'}]}}]
    body = {'messages': [r['message'] for i, r in enumerate(records) if i != 1]}
    assert check_content(body, content_reference(records), [])
