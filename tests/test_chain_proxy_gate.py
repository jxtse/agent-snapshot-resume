import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent_snapshot_resume.proxy import resume_proxy as proxy

H1 = '<system-reminder>\nfirst correction\n</system-reminder>'
H2 = '<system-reminder>\nsecond correction\n</system-reminder>'


def fixture():
    return {'model': 'test-model', 'max_tokens': 128000, 'stream': False, 'metadata': {'user_id': 'dynamic'},
            'messages': [
                {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'old', 'content': 'old output'},
                                             {'type': 'text', 'text': H1}]},
                {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'new', 'name': 'Bash', 'input': {'command': 'true'}}]},
                {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'new', 'content': 'new output'},
                                             {'type': 'text', 'text': proxy.CONTINUE_TXT}]},
                {'role': 'assistant', 'content': proxy.SYNTH_TXT},
                {'role': 'user', 'content': 'SENTINEL_TWO'}]}


def spec_fixture():
    return {'sentinel': 'SENTINEL_TWO', 'mode': 'hint', 'hint': 'second correction', 'require_sentinel': True,
            'require_text_only': True, 'required_hint_blocks': [H1, H2],
            'forbidden_sentinels': ['SENTINEL_ONE', 'SENTINEL_TWO'],
            'hint_anchors': [{'block': H1, 'tool_use_id': 'old'}, {'block': H2, 'tool_use_id': 'new'}]}


def digest(body):
    return hashlib.sha256(json.dumps({k: v for k, v in body.items() if k != 'metadata'},
                                     sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


class ChainGate(unittest.TestCase):
    def test_missing_sentinel_rejected(self):
        body = fixture()
        body['messages'] = body['messages'][:-1]
        with self.assertRaises(ValueError):
            proxy.rewrite(body, spec_fixture())

    def test_preserves_exact_hints_at_ordered_tool_boundaries(self):
        out, _ = proxy.rewrite(fixture(), spec_fixture())
        self.assertEqual(out['messages'][0]['content'][-1]['text'], H1)
        self.assertEqual(out['messages'][-1]['content'][-1]['text'], H2)
        self.assertNotIn('SENTINEL', str(out))

    def test_missing_duplicate_and_wrong_anchor_rejected(self):
        for kind in ['missing', 'duplicate', 'anchor']:
            b = fixture()
            if kind == 'missing':
                b['messages'][0]['content'].pop()
            elif kind == 'duplicate':
                b['messages'][0]['content'].append({'type': 'text', 'text': H1})
            else:
                b['messages'][0]['content'][0]['tool_use_id'] = 'not-old'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                proxy.rewrite(b, spec_fixture())

    def test_old_scaffolding_rejected(self):
        b = fixture()
        b['messages'][0]['content'].append({'type': 'text', 'text': 'SENTINEL_ONE'})
        with self.assertRaises(ValueError):
            proxy.rewrite(b, spec_fixture())

    def test_custom_resume_artifacts_from_spec(self):
        body = {'messages': [
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 't', 'content': 'x'},
                                         {'type': 'text', 'text': 'Please resume.'}]},
            {'role': 'assistant', 'content': 'Acknowledged.'},
            {'role': 'user', 'content': 'S'}]}
        spec = {'sentinel': 'S', 'resume_artifacts': {'continue_text': 'Please resume.',
                                                      'synthetic_assistant_text': 'Acknowledged.'}}
        out, acts = proxy.rewrite(body, spec)
        self.assertEqual(len(out['messages']), 1)
        self.assertEqual(out['messages'][0]['content'], [{'type': 'tool_result', 'tool_use_id': 't', 'content': 'x'}])
        self.assertIn('drop_synthetic_assistant', acts)


class ChainDispatchGate(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.spec = spec_fixture()
        self.spec['upstream'] = 'fake'
        expected, _ = proxy.rewrite(fixture(), self.spec)
        self.spec['expected_first_request_sha256'] = digest(expected)
        self.specpath = root / 'run.json'
        self.specpath.write_text(json.dumps(self.spec))
        app = web.Application()
        app.update(spec_dir=str(root), log_dir=str(root / 'logs'), specs={}, counts={}, session=AsyncMock())
        app['session'].request.side_effect = AssertionError('unexpected upstream dispatch')
        app.router.add_route('*', '/r/{run_id}/{tail:.*}', proxy.handle)
        self.app = app
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.temp.cleanup()

    async def test_changed_first_request_never_dispatches_even_on_second_request(self):
        self.spec['upstream'] = 'http://should-never-dispatch.invalid'
        self.specpath.write_text(json.dumps(self.spec))
        b = fixture()
        b['max_tokens'] = 1
        for _ in range(2):
            response = await self.client.post('/r/run/v1/messages', json=b)
            self.assertEqual(response.status, 400, await response.text())
        self.app['session'].request.assert_not_awaited()

    async def test_metadata_variance_allowed_and_later_genuine_history_allowed(self):
        b = fixture()
        b['metadata']['user_id'] = 'another-runtime-id'
        r = await self.client.post('/r/run/v1/messages', json=b)
        self.assertEqual(r.status, 200, await r.text())
        b['messages'].append({'role': 'assistant', 'content': 'genuine next-turn answer'})
        r = await self.client.post('/r/run/v1/messages', json=b)
        self.assertEqual(r.status, 200, await r.text())

    async def test_spec_with_key_file_is_refused(self):
        self.spec['key_file'] = '/some/key'
        self.specpath.write_text(json.dumps(self.spec))
        r = await self.client.post('/r/run/v1/messages', json=fixture())
        self.assertEqual(r.status, 400, await r.text())
        self.app['session'].request.assert_not_awaited()
