"""Read-only shell probes, using synthetic paths and process names."""
import json
import pytest

from agent_snapshot_resume.agents import codex, pi
from agent_snapshot_resume.trace.ir import bash_is_read_only


@pytest.mark.parametrize("command", [
    "sleep 60; tail -2 sample*.txt; ls -la results 2>/dev/null | head",
    "cat sample 2>/dev/null | head -60",
    "ps aux | grep worker | grep -v grep | wc -l",
    "find /work -maxdepth 4 -type f | sort | head",
    "cd /work && pushd nested >/dev/null && pwd; popd &>/dev/null",
    "date -u '+%F %T' || printf '%s\\n' 'date unavailable'",
    "printf '%s' 'a; b > c | d' 2>&1",
    "pgrep -af worker || true; nproc; free -m; uptime; df -h; du -sh .",
    "env -i; printenv PATH; file ./sample; stat ./sample; which bash",
    "ls *.txt &>/dev/null; ls ./sample >/dev/null 2>&1",
    "find . -name '*.x' -printf '%p\\n' | sort -k1,1 | head",
    'echo "$(date +%s)"; (cd /work && ls) | wc -l',
    "printf '%s' `pwd`; cat './semi;colon'",
    "ls \\\n  ./data # a literal comment\nhead ./summary",
    "LC_ALL=C ps -eo pid,args | grep worker",
    "date -d yesterday +%s; date -r ./stamp +%F",
    "tail -f sample",  # main omitted the blocking observer
])
def test_observers(command):
    assert bash_is_read_only(command)
    assert pi.tool_action("bash", {"command": command}, "/work")[2] == "read_only"
    payload = dict(type="function_call", name="exec_command", arguments=json.dumps({"cmd":command}))
    assert codex.tool_action(payload, "/work")[2] == "read_only"


@pytest.mark.parametrize("command", [
    "cat x 2>error.log", "ls >/dev/null.extra", "ls &> run.log", "date +%s 2>&1 >out",
    "printf '%s' hi >>file", "echo x | tee result", "ls; cd /work && rm out",
    "ls | sort -o sorted", "sort -noresult data", "sort --output=dest in",
    "find . -delete", "find . -execdir touch mark \\;", "find . -ok rm {} \\;",
    "find . -fls names", "find . -fprintf listing '%p'", "find . -fprint0 names",
    "sed -i 's/one/two/' f", "env FLAG=1 touch out", "env -S 'touch out'",
    "date -u 010100002026", "date --set=tomorrow", "unknown-probe",
    'echo "$(cat sample; touch changed)"', "printf '%s' `rm out`", "(pwd && touch sample)",
    "rg --pre helper pattern file", "uniq in out", "xxd -r hex bin",
    "du . > /dev/null && echo x >changed", "sort --compress-program=helper",
    "echo safe | sh", "ls > /dev/null 2>errors", "echo x & touch y",
    "printf -v 'arr[$(touch out)]' x", "/work/ls",
])
def test_retained_commands(command):
    assert not bash_is_read_only(command)


@pytest.mark.parametrize("command", [
    "diff <(sort first) <(sort second)", "awk -F, '{print $2}' ./records",
    "ls part{1..4}.log", "sort -t, -k2,2nr ./records", "command -v sed",
    "command grep -n item file", "echo $$; env FLAG=1", "top -b -n1",
    "sed -n '2,8p' ./output", "echo $(( $(date +%s) - 10 ))", "[ -f ./result ]",
    'grep -E "x>y|a;b" ./input',
])
def test_other_pure_probes(command):
    assert bash_is_read_only(command)


@pytest.mark.parametrize("command", [
    "find . -{delete,print}", "sort data -o result", "uniq -- -c -d", "uniq input*.txt",
    "env -- -i", "command touch out", "awk 'BEGIN {print 1 > \"out\"}'", "awk -f script.awk",
    'diff <(sort -o new old) old', "cat <input", "xxd input output",
    "find . -name '*' -okdir touch mark \\;", "printf '%s' `touch mark`",
    "sort --compress-prog=helper data", "sed -n '1w changed' input",
    "awk 'BEGIN {system(\"touch mark\")}'", "find . \"${OPT}glob\"",
])
def test_utility_and_expansion_effects(command):
    assert not bash_is_read_only(command)


def test_cc_replay_uses_later_calls_recorded_cwd(tmp_path, monkeypatch):
    from agent_snapshot_resume import bundle
    from agent_snapshot_resume.trace import claude_code
    from test_review_native import replay_actions

    nested = tmp_path/"subdir"
    nested.mkdir()
    rows = []
    for i, (command, cwd) in enumerate([("cd subdir && pwd", tmp_path), ("printf result > target", nested),
                                         ("pwd", nested)]):
        rows.append(dict(type="assistant", timestamp=f"2026-01-01T00:00:0{i}Z", cwd=str(cwd), message=dict(
            id=str(i), content=[dict(type="tool_use", id=str(i), name="Bash", input={"command":command})])))
        rows.append(dict(type="user", timestamp=rows[-1]["timestamp"], message=dict(
            content=[dict(type="tool_result", tool_use_id=str(i), content="")])))
    session = tmp_path/"agent/sessions/projects/-task/fixture.jsonl"
    session.parent.mkdir(parents=True)
    session.write_text("".join(json.dumps(r)+"\n" for r in rows))
    ir = claude_code.load_trial(str(tmp_path), 60, include_config=False)
    plan = bundle.plan_with_observations(ir, 2)
    assert [a["cwd"] for a in plan["actions"]] == [str(nested)]
    replay_actions(tmp_path, plan["actions"], monkeypatch)
    assert (nested/"target").read_text() == "result" and not (tmp_path/"target").exists()
