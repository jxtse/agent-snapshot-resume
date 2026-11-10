"""Fail-closed gates for source replay and restored task/auxiliary files.

validate_replay(plan, report, rc, reviewed_exceptions) raises unless every planned action replayed with an accepted
fidelity. Outputs that differ only in known-benign ways (hardware/memory probes, shell clocks, cProfile timings,
terminal `| grep` no-match exit codes, original timeouts) are accepted ONLY when the bundle lists a reviewed exception
bound to the exact command's sha256 (bundle.json `replay_reviewed_exceptions`: {call_id: {kind, command_sha256, ...}}).
"""

def profile_without_times(text):
    """Parse a cProfile table, retaining science, call counts and function identities."""
    import re
    from .replay_inside import norm
    text = norm(text)
    header = re.search(r'(?m)^\s*(\d+) function calls \((\d+) primitive calls\) in \d+(?:\.\d+)? seconds$', text)
    if header is None:
        return None
    science = text[:header.start()].strip()
    tail = text[header.end():].strip().splitlines()
    tail = [line.strip() for line in tail if line.strip()]
    if len(tail) < 4 or tail[0] != 'Ordered by: cumulative time':
        return None
    count = re.fullmatch(r'List reduced from (\d+) to (\d+) due to restriction <(\d+)>', tail[1])
    if count is None or count[2] != count[3] or len(tail[3:]) != int(count[2]):
        return None
    if tail[2].split() != ['ncalls','tottime','percall','cumtime','percall','filename:lineno(function)']:
        return None
    rows = []
    for row in tail[3:]:
        parsed = re.fullmatch(r'(\d+(?:/\d+)?)\s+(?:\d+\.\d+\s+){4}(.+)', row)
        if parsed is None:
            return None
        rows.append(parsed.groups())
    return science, header.groups(), count.groups(), sorted(rows)


def validate_replay(plan, report, rc, reviewed_exceptions=None):
    reviewed_exceptions = reviewed_exceptions or {}
    if str(rc).strip() != '0' or len(report.get('results', [])) != len(plan['actions']):
        raise RuntimeError('Replay did not complete all planned actions')
    allowed={'exact','normalized','same_lines_reordered','volatile_only','address_and_order_only',
             'line_alignment_and_order_only','error_reproduced','orig_error_not_applied'}
    for action, result in zip(plan['actions'],report['results']):
        if result.get('call_id') != action['call_id']:
            raise RuntimeError('Replay action identity mismatch')
        fidelity=result.get('fidelity')
        exit_reviewed = False
        if action['side_effect']=='exec_background':
            if not (action.get('bg_finished_before_k',True) and action.get('bg_status')=='completed'
                    and result.get('rc')==0 and fidelity=='bg_rerun_foreground'
                    and result.get('background_output_file')==action.get('background_output_file')
                    and result.get('background_output_sha256')):
                raise RuntimeError('Completed background output not restored: '+action['call_id'])
        elif (action['side_effect']=='task_mgmt' and action.get('tool') in ('TaskCreate','TaskUpdate')
              and fidelity=='task_mgmt_restored_from_session'):
            result['deferred_gate'] = 'session auxiliary files must be hash-verified after clone seeding, before CC starts'
        elif fidelity not in allowed:
            import hashlib, re
            review = reviewed_exceptions.get(action['call_id'], {})
            command = action.get('args', {}).get('command', '')
            old = action.get('orig_text', '').strip().splitlines()
            new = result.get('stdout_full', '').strip().splitlines()
            version_line = lambda s: re.sub(r'\(main, [^)]+\)', '(main, <build-stamp>)', s)
            hardware_only = (review.get('kind') in ('hardware_probe','hardware_probe_and_python_build_stamp') and
                             review.get('command_sha256') == hashlib.sha256(command.encode()).hexdigest() and
                             len(old) >= 9 and len(new) >= 9 and old[1:7] == new[1:7] and
                             (old[0] == new[0] or (review.get('kind') == 'hardware_probe_and_python_build_stamp' and version_line(old[0]) == version_line(new[0]))) and
                             old[7].split() == new[7].split() and old[8].startswith('Mem:') and new[8].startswith('Mem:') and
                             not result.get('stderr_full') and result.get('rc') == 0)
            memory_pattern = r'(?m)^Mem:\s+(?:\d+\s+){5}\d+\s*$'
            memory_only = (review.get('kind') == 'shell_memory_probe' and
                           review.get('command_sha256') == hashlib.sha256(command.encode()).hexdigest() and
                           len(re.findall(memory_pattern, '\n'.join(old))) == 1 and
                           len(re.findall(memory_pattern, '\n'.join(new))) == 1 and
                           re.sub(memory_pattern, 'Mem:<host-memory>', '\n'.join(old)) ==
                           re.sub(memory_pattern, 'Mem:<host-memory>', '\n'.join(new)) and
                           not result.get('stderr_full') and result.get('rc') == 0)
            terminal_grep_only = (review.get('kind') == 'terminal_grep_no_match' and
                review.get('command_sha256') == hashlib.sha256(command.encode()).hexdigest() and
                bool(re.search(r'\|\s*grep\s', command.rstrip().splitlines()[-1])) and
                result.get('rc') == 1 and not action.get('is_error') and not result.get('stderr_full') and
                action.get('orig_text', '').strip() == result.get('stdout_full', '').strip())
            from .replay_inside import norm, original_timeout
            clock = re.compile(r'^(?:real|user|sys)\s+(?:\d+m\d+(?:\.\d+)?s|\d+(?:\.\d+)?)$')
            no_clock = lambda text: '\n'.join(line for line in norm(text).splitlines() if not clock.fullmatch(line)).strip()
            bound = review.get('command_sha256') == hashlib.sha256(command.encode()).hexdigest()
            timed_grep = (review.get('kind') == 'terminal_grep_with_shell_time' and bound and
                bool(re.search(r'\|\s*grep\s', command.rstrip().splitlines()[-1])) and
                result.get('rc') == 1 and not action.get('is_error') and
                all(clock.fullmatch(line) for line in result.get('stderr_full','').splitlines() if line.strip()) and
                no_clock(action.get('orig_text','')) == norm(result.get('stdout_full','')))
            old_to = original_timeout(action.get('orig_text'))
            timeout_same = (review.get('kind') == 'original_timeout_partial_output' and bound and old_to is not None and
                action.get('is_error') and result.get('orig_timed_out') and fidelity == 'timeout_reproduced' and
                result.get('status') == 'timeout' and result.get('timeout_s') == old_to and result.get('rc') is None and
                not result.get('stderr_full') and
                no_clock('\n'.join(action['orig_text'].splitlines()[2:])) == no_clock(result.get('stdout_full','')))
            profile_old = profile_without_times(action.get('orig_text', '')) if review.get('kind') in ('cprofile_timings','cprofile_reviewed_aggregate_counts') else None
            profile_new = profile_without_times(result.get('stdout_full','')) if profile_old is not None else None
            aggregate_reviewed = (profile_old is not None and profile_new is not None and
                review.get('kind') == 'cprofile_reviewed_aggregate_counts' and
                list(profile_old[1]) == review.get('original_aggregate_counts') and
                list(profile_new[1]) == review.get('replayed_aggregate_counts') and
                profile_old[0] == profile_new[0] and profile_old[2:] == profile_new[2:])
            profile_only = (profile_old is not None and
                review.get('command_sha256') == hashlib.sha256(command.encode()).hexdigest() and
                'cProfile.Profile()' in command and result.get('rc') == 0 and not result.get('stderr_full') and
                (profile_old == profile_new or aggregate_reviewed))
            if not (hardware_only or memory_only or terminal_grep_only or profile_only or timed_grep or timeout_same):
                raise RuntimeError('Unaccepted replay fidelity '+str(fidelity)+': '+action['call_id'])
            exit_reviewed = terminal_grep_only or timed_grep or timeout_same
            result['reviewed_exception'] = ('identical full stdout; command-bound terminal grep no-match returns1 while original CC reported non-error' if terminal_grep_only else
                'same original timeout deadline and full partial scientific output; shell timing differences only' if timeout_same else
                'identical scientific stdout; command-bound terminal grep returns1 and stderr contains only shell clocks' if timed_grep else
                'identical scientific output and profiled function identities/counts; explicitly bound aggregate profiler-count difference plus timings' if profile_only and aggregate_reviewed else
                'identical scientific output, cProfile call counts and profiled function identities; performance timings/order only' if profile_only else
                'same package/version/CPU output; host-memory report only' if memory_only else
                'same Python release/compiler, scientific package versions and CPU count; reviewed Python build timestamp and host free-memory report differences')
        if action['side_effect']=='exec' and not exit_reviewed and bool(result.get('rc')) != bool(action.get('is_error')):
            raise RuntimeError('Replay exit outcome changed: '+action['call_id'])
    return True


def check_hashes(expected, actual, label):
    missing_or_different=[p for p,h in expected.items() if actual.get(p)!=h]
    if missing_or_different:
        raise RuntimeError(label+' hash mismatch: '+', '.join(missing_or_different))
    return True
