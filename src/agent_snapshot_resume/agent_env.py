"""Historical agent configuration allowed in a native replay and its resumed process."""
import json
import re
import shlex
from pathlib import Path


SECRET_NAME = re.compile(r"(KEY|SECRET|TOKEN(?!S)|PASSWORD|CREDENTIAL|AUTH)", re.I)
RESERVED_ENV = {"ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_BASE_URL",
                "OPENAI_API_KEY", "CLAUDE_CONFIG_DIR"}
_ROUTE_ENV = {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "ASR_UPSTREAM_URL",
              "ASR_MODEL", "OPENAI_API_BASE", "OPENAI_MODEL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "ANTHROPIC_MODEL"}
_SESSION_ENV = {"codex": {"CODEX_HOME"}, "pi": {"PI_CODING_AGENT_DIR", "PI_OFFLINE"}}
_NAME = r"[a-zA-Z_][a-zA-Z_0-9]*"
# The guest shell supplies these independently of agent.env.
_SHELL_ENV = {"HOME", "USER", "LOGNAME", "SHELL", "PATH", "PWD", "OLDPWD", "SHLVL", "IFS"}
_EXPANSION = re.compile(r"\$\{?[!#]?(" + _NAME + ")")
_EXPLICIT_LOOKUP = re.compile(
    r"(?:\bprintenv\s+['\"]?|\b(?:getenv|environ\.get|Sys\.getenv)\(\s*['\"]|"
    r"\benviron\[\s*['\"]|\bprocess\.env\.)(" + _NAME + ")")
_DYNAMIC_LOOKUP = re.compile(
    r"""\bprintenv\s+["']?\$|\b(?:getenv|environ\.get)\(\s*(?![\s'"])\w|\bprocess\.env\[\s*[^\s'"]""")
_QUOTED = re.compile(r"""'(?:[^']*)'|"(?:\\.|[^"\\])*"|\\.""", re.S)
_LITERAL_ASSIGNMENT = re.compile(r"""^\s*([A-Za-z_]\w*)=(?:[\w./:-]*|'[^']*')\s*[;\n]""")


# A quoted here-document consumed as source by one of these interpreters does
# not expand shell variables. Language-level environment lookups still count.
_LITERAL_SOURCE = re.compile(
    r"""(?m)^(?P<head>[ \t]*(?:Rscript|R|python[23]?)[ \t][^;&|<>\n]*<<[ \t]*"""
    r"""(?P<quote>['"])(?P<tag>\w+)(?P=quote)[ \t]*\n)[\s\S]*?^(?P=tag)$""")


# A credential can sit in a value whose name looks harmless (DATABASE_URL, REDIS_DSN, ...).
# Values that look like one are treated as unrecoverable: never written to plan.json/job.json
# and never exported to replay or the resumed agent.
_SECRET_VALUE = [
    re.compile(r"[a-z][a-z0-9+.-]*://[^/?#\s@]+@", re.I),                  # URL/DSN userinfo
    re.compile(r"(?:^|[\s;&?,])(?:password|passwd|pwd|secret|token|api[_-]?key|apikey|access[_-]?key"
               r"|private[_-]?key|client[_-]?secret|auth)\s*[=:]\s*\S", re.I),  # key=value DSN / query
    re.compile(r"\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}", re.I),        # auth header values
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}|\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_\w{20,}"
               r"|\bxox[abposr]-[A-Za-z0-9-]{10,}|\bAKIA[0-9A-Z]{16}\b|\bAIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),   # JWT
]
_OPAQUE = re.compile(r"[A-Za-z0-9+=_-]{32,}")


def secret_value(value: str) -> bool:
    """True when a value looks like a credential, whatever its variable name."""
    if any(p.search(value) for p in _SECRET_VALUE):
        return True
    # A long opaque token mixing lower case, upper case and digits. Path segments, plain
    # words and hex digests (lower case + digits) do not match.
    # Split on _ and - so identifiers like RUN_SUMMARY_v2_final do not qualify.
    words = (s for w in _OPAQUE.findall(value) for s in re.split(r"[_-]", w) if len(s) >= 24)
    return any(all(re.search(c, w) for c in (r"[a-z]", r"[A-Z]", r"[0-9]")) for w in words)


def native_agent_env(trial_dir, agent):
    reserved = RESERVED_ENV | _ROUTE_ENV | _SESSION_ENV[agent]
    path = Path(trial_dir) / "config.json" if trial_dir else None
    config = json.loads(path.read_bytes()) if path and path.is_file() else {}
    safe, unavailable = {}, set(reserved)
    for name, value in ((config.get("agent") or {}).get("env") or {}).items():
        if (SECRET_NAME.search(name) or name.upper() in reserved or not re.fullmatch(_NAME, name)
                or not isinstance(value, str) or "\x00" in value or re.search(r"\$[\w{]", value)
                or secret_value(value)):
            unavailable.add(name)
        else:
            safe[name] = value
    return safe, unavailable


def unavailable_references(command, safe, unavailable):
    """Resolve direct named dependencies. Shell literals can also contain embedded programs."""
    known = set(safe) | _SHELL_ENV
    explicit = set(_EXPLICIT_LOOKUP.findall(command))
    # Drop literal shell arguments, preserving double-quoted expansions. Check explicitly
    # forbidden names even inside nested shell/program text, whose syntax may be unknown.
    expansion_text = _LITERAL_SOURCE.sub(r"\g<head>", command)
    shell = _QUOTED.sub(lambda m: m[0] if m[0].startswith('"') else " ", expansion_text)
    used = set(_EXPANSION.findall(shell)) | explicit
    if "${!" in shell or _DYNAMIC_LOOKUP.search(command):
        used.add("computed environment variable name")
    for name in _EXPANSION.findall(command):
        if SECRET_NAME.search(name) or name in unavailable or name.upper() in unavailable:
            used.add(name)
    # A leading literal assignment is reproducible inside this shell.
    assignment = _LITERAL_ASSIGNMENT.match(command)
    if assignment:
        known.add(assignment[1])
    try:
        words = shlex.split(command)
    except ValueError:
        words = []
    if len(words) >= 3 and words[0].rsplit("/", 1)[-1] in {"bash", "sh"} and words[1] in {"-c", "-lc"}:
        used.update(unavailable_references(words[2], known, unavailable))
    return sorted(name for name in used if name not in known or SECRET_NAME.search(name)
                  or name in unavailable or name.upper() in unavailable)
