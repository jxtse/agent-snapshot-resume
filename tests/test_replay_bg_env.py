"""Offline regressions for background lifecycle, snapshot tails, and environment-only drift."""
import json
import sys
from unittest.mock import MagicMock

import pytest

from agent_snapshot_resume import bundle, chain
from agent_snapshot_resume.replay import gate, replay_inside as replay
from agent_snapshot_resume.sandbox import source


def execute(tmp_path, monkeypatch, actions, **fields):
    plan = dict(actions=actions, initial_cwd=str(tmp_path), **fields)
    pp, rp = tmp_path / 'plan.json', tmp_path / 'report.json'
    pp.write_text(json.dumps(plan))
    monkeypatch.setattr(sys, 'argv', ['replay', str(pp), str(rp)])
    replay.main()
    return plan, json.loads(rp.read_text())


def bg(tmp_path, command, rc=124, duration=.1):
    return dict(call_id='job', tool='Bash', side_effect='exec_background',
                args={'command': command}, cwd=str(tmp_path), bg_status='failed', bg_exit_code=rc,
                bg_duration_s=duration, duration_s=120, bg_finished_before_k=True,
                background_output_file=str(tmp_path / 'job.output'), is_error=False)


@pytest.mark.parametrize('code', [124, 137, 143, 144])
def test_historical_termination(tmp_path, monkeypatch, code):
    plan, report = execute(tmp_path, monkeypatch, [bg(tmp_path, "printf 'progress 1\\n'; sleep 5", code)],
                           bg_output_trailer='\n[exited with code {rc}]\n')
    r = report['results'][0]
    assert r['rc'] == code and r['dur_s'] < 2
    assert r['fidelity'] == 'bg_termination_reproduced'
    assert 'progress 1' in (tmp_path / 'job.output').read_text()
    for policy in ('strict', 'lenient'):
        gate.validate_replay(plan, report, 0, policy=policy)


def test_fast_failure_preserves_write_order(tmp_path, monkeypatch):
    a = bg(tmp_path, 'bash missing.sh', 127, 0)
    write = dict(call_id='write', tool='Write', side_effect='file_write',
                 args=dict(file_path=str(tmp_path / 'missing.sh'), content='sleep 5'))
    plan, report = execute(tmp_path, monkeypatch, [a, write])
    assert report['results'][0]['rc'] == 127
    gate.validate_replay(plan, report, 0, policy='lenient')


def test_fast_failure_not_fabricated(tmp_path, monkeypatch):
    plan, report = execute(tmp_path, monkeypatch, [bg(tmp_path, 'sleep 5', 2, 0)])
    with pytest.raises(RuntimeError, match='background_failure_changed'):
        gate.validate_replay(plan, report, 0, policy='lenient')


def test_changed_consumed_output_blocks(tmp_path):
    a = bg(tmp_path, 'x')
    a['bg_output_read_later'] = ['read']
    r = dict(call_id='job', status='ran', rc=124)
    assert not gate.background_failure_equivalent(a, r, [r])
    assert not gate.background_failure_equivalent(a, r, [dict(call_id='read', fidelity='numbers_differ')])
    assert gate.background_failure_equivalent(a, r, [dict(call_id='read', fidelity='exact')])


def test_snapshot_tail_retains_dispatched_job():
    a = dict(call_id='job', side_effect='exec_background', obs_line=4, bg_finished_line=12)
    assert chain.tail_actions([a], 10)[0]['bg_inflight_at_snapshot']
    assert not chain.tail_actions([a], 13)
    assert chain.tail_actions([a], 12)[0]['bg_inflight_at_snapshot']
    assert 'bg_inflight_at_snapshot' not in a


def test_snapshot_rerun_before_dependent_read(tmp_path, monkeypatch):
    stopped = []
    monkeypatch.setattr(replay, 'stop_snapshot_background', lambda a: stopped.append(a['call_id']) or [42])
    a = bg(tmp_path, "echo saved > artifact; echo done", 0)
    a.update(bg_status='completed', bg_inflight_at_snapshot=True)
    read = dict(call_id='read', tool='Bash', side_effect='exec', args={'command': 'cat artifact'}, orig_text='saved')
    _, report = execute(tmp_path, monkeypatch, [a, read])
    assert stopped == ['job']
    assert report['results'][0]['background_resume'] == 'rerun_from_snapshot'
    assert report['results'][1]['fidelity'] in ('exact', 'normalized')


def test_completed_background_uses_total_duration(tmp_path, monkeypatch):
    seen = []
    def run(cmd, cwd, timeout, **kw):
        seen.append(timeout)
        return dict(rc=0, stdout='', stderr='', dur=0, timeout=False)
    monkeypatch.setattr(replay, 'run_bash', run)
    a = bg(tmp_path, 'install', 0, 147.8)
    a.update(bg_status='completed', orig_text='Exit code 143\nCommand timed out after 2m 0s')
    execute(tmp_path, monkeypatch, [a])
    assert seen == [325]


def test_harness_listing_rule():
    a = dict(args={'command': 'pip list 2>/dev/null | head -60; echo ---; ls /opt/tbs-proxy 2>/dev/null'},
             orig_text='Package Version\n---\nproxy.py\nvenv')
    r = dict(status='ran', rc=2, stdout_full='Package Version\n---\n', stderr_full='')
    assert gate.harness_path_only(a, r)
    assert not gate.harness_path_only(a, r, ['/different'])
    for command in ('touch file; ls /opt/tbs-proxy', 'echo x > file; ls /opt/tbs-proxy',
                    'pip install x; ls /opt/tbs-proxy', 'echo $(touch file); ls /opt/tbs-proxy'):
        assert not gate.harness_path_only(dict(a, args={'command': command}), r)
    assert not gate.harness_path_only(a, dict(r, stdout_full='changed\n---'))


def test_deadline_forwarded_and_default():
    sb = MagicMock()
    source.dispatch_replay(sb, '/app', 1800, 25000)
    assert '25000s' in sb.commands.run.call_args.args[0]
    source.dispatch_replay(sb, '/app', 1800)
    assert '10800s' in sb.commands.run.call_args.args[0]


def test_captured_path_after_login(tmp_path):
    directory = tmp_path / 'agent/sessions/shell-snapshots'
    directory.mkdir(parents=True)
    (directory / 'env.sh').write_text("export PATH='/custom/env/bin:/usr/bin:/bin'\nexport TOKEN='secret'\n")
    env = bundle.captured_shell_env(tmp_path)
    assert env == {'PATH': '/custom/env/bin:/usr/bin:/bin'}
    r = replay.run_bash('printf %s "$PATH"', str(tmp_path), 2, shell_env=env)
    assert r['stdout'] == env['PATH']
    (directory / 'other.sh').write_text("export PATH='/different'\n")
    assert bundle.captured_shell_env(tmp_path) == {}


def test_background_output_prefix_and_numeric_drift(tmp_path):
    a = bg(tmp_path, 'x')
    a['bg_original_output'] = 'iteration 1 loss 2.0\niteration 2 loss 1.0'
    r = dict(status='ran', rc=124, stdout_full='iteration 1 loss 2.1\n')
    assert gate.background_failure_equivalent(a, r, [])
    assert not gate.background_failure_equivalent(a, dict(r, stdout_full='different error'), [])
    assert not gate.background_failure_equivalent(a, dict(r, stdout_full=''), [])


def test_background_read_compares_actual_file(tmp_path, monkeypatch):
    output = tmp_path / 'output'
    output.write_text('value 7\n')
    a = dict(call_id='reader', tool='Read', side_effect='read_only',
             evidence_read_for='failed_background_output', args={'file_path': str(output)},
             orig_text='     1→value 6')
    _, report = execute(tmp_path, monkeypatch, [a])
    assert report['results'][0]['fidelity'] == 'numbers_differ'


def test_natural_signal_uses_shell_exit_convention(tmp_path):
    result = replay.run_bash('kill -TERM $$', str(tmp_path), 2)
    assert result['rc'] == 143
    assert not result['timeout']


def test_snapshot_ownership_check_fails_without_proc(tmp_path, monkeypatch):
    monkeypatch.setattr(replay.os.path, 'isdir', lambda p: False)
    with pytest.raises(RuntimeError, match='cannot identify'):
        replay.stop_snapshot_background(bg(tmp_path, 'echo x'))
