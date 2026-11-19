"""Recognize observation-only Bash command lists without running or expanding archived input."""
import posixpath
import re


BASH_CLASSIFIER_REVISION = 1


class _Retain(ValueError):
    """Shell syntax or a utility invocation that must remain a replay action."""


_OBSERVERS = frozenset("""
    ls cat head tail wc grep egrep fgrep pwd echo cd pushd popd which stat du df nproc free uname
    whoami id printenv cut tr cmp md5sum sha256sum od realpath dirname basename true false uptime
    type test [ readlink sleep zcat strings nl seq ps pgrep lscpu column
""".split())
_CONNECTOR = re.compile(r"&&|\|\||\|&|[;|\n]")
# Longest operator first so descriptor duplication is one redirection.
_REDIRECT = re.compile(r"(\d*)(&>>?|>&|>>?|<)")
_PARAM = re.compile(r"\$(?:[a-zA-Z_]\w*|\{[a-zA-Z_]\w*\}|[?0-9])")
_BRACE = re.compile(r"\{(?:[\w.]+(?:,[\w.]+)+|\d+\.\.\d+)\}")
_NUMBER = r"(?:[0-9]+|\$\(date[ \t]+\+%s\))"
_ARITHMETIC = re.compile(r"\$\(\(\s*" + _NUMBER + r"(?:\s*[-+*/%]\s*" + _NUMBER + r")*\s*\)\)")
_PLAIN = re.compile(r"""[^ \t\n\r;&|<>()\x60$'"\\{}*?\[]+""")
_GLOB = "\x01"  # Multiple pathnames, each with a known non-option prefix.
_DYNAMIC = "\0"  # Input containing NUL is rejected. An expansion's value is unknown.
_LANG_ASSIGN = re.compile(r"(?:LC_[A-Z_]+|LANG|TZ)=[\w./:+-]*\Z")


def _check_utility(words):
    index = 0
    while index < len(words) and _LANG_ASSIGN.fullmatch(words[index]):
        index += 1
    if index == len(words):
        return
    exe, *args = words[index:]
    name = posixpath.basename(exe)
    if exe not in (name, "/bin/" + name, "/usr/bin/" + name):
        raise _Retain("nonstandard executable path")
    if name in _OBSERVERS:
        if name in ("[", "test") and any(arg in ("-v", "-R") for arg in args):
            raise _Retain("test variable lookup")
        return
    if name == "tee":
        raise _Retain("tee output")
    if name == "printf":
        fmt = args[1:] if args[:1] == ["--"] else args
        if fmt and (_DYNAMIC in fmt[0] or _GLOB in fmt[0] or fmt[0].startswith("-v") or "%n" in fmt[0]):
            raise _Retain("printf variable target or dynamic format")
        return
    if name == "command":
        if args[:1] in (["-v"], ["-V"]):
            return
        if args and posixpath.basename(args[0]) != "command" and not _LANG_ASSIGN.fullmatch(args[0]):
            _check_utility(args)
            return
        raise _Retain("command invocation")
    # A field expansion may provide additional words or options. Globs with a
    # fixed non-option prefix are safe for tools taking any number of input files.
    if name in ("find", "sort", "rg", "file"):
        args = [a.replace(_GLOB, "path") for a in args]
    if any(_DYNAMIC in a or _GLOB in a for a in args):
        raise _Retain("dynamic utility arguments")
    if name == "find":
        it = iter(args)
        for a in it:
            if a in ("-name", "-iname", "-path", "-ipath", "-regex", "-iregex", "-printf"):
                next(it, None)
            elif (a in ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fls", "-cpio", "-ncpio")
               or a.startswith(("-fprint", "-fprintf"))):
                raise _Retain("find filesystem or command action")
        return
    elif name == "sort":
        flags = {"reverse", "unique", "stable", "numeric-sort", "version-sort", "human-numeric-sort",
                 "ignore-case", "general-numeric-sort"}
        valued = {"field-separator", "key", "buffer-size", "parallel", "temporary-directory"}
        it = iter(args)
        for arg in it:
            if arg == "--":
                break
            if arg.startswith("--"):
                option, equals, _ = arg[2:].partition("=")
                if option in valued and not equals:
                    next(it, None)
                elif option not in flags | valued:
                    raise _Retain("sort output option or unsupported option")
            elif arg.startswith("-"):
                for index, option in enumerate(arg[1:], 2):
                    if option in "kStT":
                        if index == len(arg):
                            next(it, None)
                        break
                    if option not in "bcCdfghiMmnRrsuvVz":
                        raise _Retain("sort output option or unsupported option")
        return
    elif name == "date":
        it = iter(args)
        for arg in it:
            if arg in ("-d", "--date", "-r", "--reference"):
                next(it, None)
            elif not (arg.startswith(("+", "--date=", "--reference=")) or
                      re.fullmatch(r"-[uR]+|-I(?:date|hours|minutes|seconds|ns)?|--utc|--rfc-email", arg)):
                raise _Retain("date operand or setting option")
        return
    elif name == "env":
        pos = 0
        while pos < len(args):
            arg = args[pos]
            if arg in ("-u", "--unset", "-C", "--chdir"):
                pos += 2
            elif arg in ("-", "-i", "-0", "--ignore-environment", "--null") or arg.startswith(("--unset=", "--chdir=")):
                pos += 1
            else:
                if arg == "--":
                    pos += 1
                break
        if all(re.match(r"^[A-Za-z_]\w*=", a) for a in args[pos:]):
            return
        raise _Retain("env invokes a command or splits arguments")
    elif name == "uniq":
        it = iter(args)
        inputs = 0
        for arg in it:
            if arg == "--":
                inputs += sum(1 for _ in it)
                break
            if arg.startswith("-") and arg != "-":
                if not re.fullmatch(r"-[cdiu]+", arg):
                    raise _Retain("uniq arguments")
            else:
                inputs += 1
        if inputs <= 1:
            return
    elif name == "rg":
        if all(not a.startswith("-") or re.fullmatch(
                r"--?|-[inlovcFHhwSseNAaBCgUuzxLmPqItT0-9]+|--(?:files|no-heading|hidden|no-ignore|"
                r"line-number|glob|count|fixed-strings|ignore-case|smart-case|files-with-matches|quiet|"
                r"word-regexp|pcre2|no-filename|with-filename)|--(?:color|type|type-not|max-count)(?:=.*)?", a)
               for a in args):
            return
    elif name == "file":
        if all(not a.startswith("-") or re.fullmatch(r"--?|-[bhiLNz]+|--mime(?:-type|-encoding)?", a) for a in args):
            return
    elif name == "awk":
        pos = 0
        while pos < len(args) and args[pos].startswith("-F"):
            pos += 2 if args[pos] == "-F" else 1
        if pos < len(args):
            program, *files = args[pos:]
            if not program.startswith("-") and all(a == "-" or not a.startswith("-") for a in files):
                # Check even string and regexp literals. Do not try to interpret an AWK
                # program containing an output operator, shell pipe or dynamic extension.
                if not re.search(r"[>|@]|\b(?:system|getline)\b", program.replace("\\\n", "")):
                    return
        raise _Retain("awk program or external I/O")
    elif name == "diff":
        if all(not a.startswith("-") or re.fullmatch(r"--?|-[abBdiNpqrsuwU0-9]+", a) for a in args):
            return
    elif name == "xxd":
        if len(args) <= 1 and all(not a.startswith("-") or a == "-" for a in args):
            return
    elif name == "top":
        if all(re.fullmatch(r"-[bcn0-9]+|[0-9]+", a) for a in args):
            return
    elif name == "nvidia-smi" and not args:
        return
    elif name == "tar" and len(args) == 2 and re.fullmatch(r"-t[zjJ]?f", args[0]):
        if not args[1].startswith("-") and ":" not in args[1]:
            return
    elif name == "unzip" and len(args) == 2 and args[0] == "-l" and not args[1].startswith("-"):
        return
    elif name == "sed":
        if len(args) >= 2 and args[0] == "-n" and re.fullmatch(r"(?:\d+,)?(?:\d+|\$)p", args[1]) and all(
                not a.startswith("-") for a in args[2:]):
            return
    else:
        raise _Retain("unclassified executable")
    raise _Retain(name + " arguments")


class _Shell:
    def __init__(self, text):
        self.text = text
        self.i = 0

    def peek(self):
        return self.text[self.i:self.i + 1]

    def word(self, depth, end):
        """Read one argument. Quoted operators are data. Check nested substitutions in place."""
        parts = []
        start = self.i
        quoted = False
        while (c := self.peek()):
            if not quoted and self.text.startswith("<(", self.i):
                self.i += 2
                self.sequence(depth + 1, ")")
                parts.append("/dev/fd/0")  # A read endpoint, irrespective of the assigned descriptor.
                continue
            if not quoted and (c in " \t\n;&|<>()" or c == end):
                break
            if c == '"':
                quoted = not quoted
                self.i += 1
            elif c == "'" and not quoted:
                close = self.text.find("'", self.i + 1)
                if close < 0:
                    raise _Retain("unfinished quote")
                parts.append(self.text[self.i + 1:close])
                self.i = close + 1
            elif c == "\\":
                self.i += 1
                escaped = self.peek()
                if not escaped:
                    raise _Retain("unfinished escape")
                if escaped != "\n":
                    parts.append(("\\" if quoted and escaped not in '"$\\`' else "") + escaped)
                self.i += 1
            elif c == "`":
                close = self.text.find("`", self.i + 1)
                body = self.text[self.i+1:close]
                if close < 0 or "\\" in body:
                    raise _Retain("complex backquote substitution")
                _Shell(body).sequence(depth + 1)
                self.i = close + 1
                parts.append(_DYNAMIC)
            elif self.text.startswith("$(", self.i):
                if self.text.startswith("$((", self.i):
                    expression = _ARITHMETIC.match(self.text, self.i)
                    if not expression:
                        raise _Retain("dynamic arithmetic expansion")
                    self.i = expression.end()
                    parts.append(_DYNAMIC)
                    continue
                self.i += 2
                self.sequence(depth + 1, ")")
                parts.append(_DYNAMIC)
            elif c == "$":
                param = _PARAM.match(self.text, self.i)
                if param:
                    parts.append(_DYNAMIC)
                    self.i = param.end()
                elif self.text.startswith("$$", self.i):
                    parts.append("0")  # PID is one non-option numeric word.
                    self.i += 2
                elif self.text[self.i+1:self.i+2] in ("{", "[", "*", "@", "!", "#", "-") or (
                        not quoted and self.text[self.i+1:self.i+2] in ("'", '"')):
                    raise _Retain("complex shell expansion")
                else:
                    parts.append(c)
                    self.i += 1
            elif not quoted and c == "{":
                brace = _BRACE.match(self.text, self.i)
                if not brace:
                    raise _Retain("brace or compound shell syntax")
                parts.append(_DYNAMIC if parts and parts[0].startswith("-") else _GLOB)
                self.i = brace.end()
            elif not quoted and c in "*?[":
                # A pathname pattern with a literal prefix cannot introduce an option
                # unless its prefix itself starts with a dash.
                parts.append(_GLOB if parts and parts[0] and not parts[0].startswith("-") else _DYNAMIC)
                self.i += 1
            else:
                chunk = _PLAIN.match(self.text, self.i)
                parts.append(chunk[0] if chunk else c)
                self.i += len(parts[-1])
        if quoted or self.i == start:
            raise _Retain("missing word or unfinished quote")
        value = "".join(parts)
        if _LANG_ASSIGN.fullmatch(value) and not self.text.startswith(value.split("=", 1)[0] + "=", start):
            return _DYNAMIC + value  # Quotes before the equals sign make this a command word.
        return value

    def sequence(self, depth=0, end=""):
        if depth > 32:
            raise _Retain("shell nesting limit")
        words = []
        grouped = False
        while True:
            c = self.peek()
            if c and c in " \t":
                self.i += 1
                continue
            if c == "#":
                newline = self.text.find("\n", self.i)
                self.i = len(self.text) if newline < 0 else newline
                continue
            connector = _CONNECTOR.match(self.text, self.i)
            if not c or c == end or connector:
                _check_utility(words)
                if not c or c == end:
                    if c != end:
                        raise _Retain("unclosed command group")
                    self.i += bool(c)
                    return
                self.i = connector.end()
                words = []
                grouped = False
                continue
            redirect = _REDIRECT.match(self.text, self.i)
            if redirect and not self.text.startswith("<(", self.i):
                self.i = redirect.end()
                while self.peek() in (" ", "\t"):
                    self.i += 1
                op = redirect[2]
                if self.peek() in ("<", "(", "|", ">"):
                    raise _Retain("input or complex redirection")
                target = self.word(depth, end)
                if op == ">&" and target in ("1", "2"):
                    continue
                if target != "/dev/null":
                    raise _Retain("input redirection" if op == "<" else "output redirection")
                continue
            if c == "(" and not words and not grouped:
                self.i += 1
                self.sequence(depth + 1, ")")
                grouped = True
            elif c in "&()" or grouped:
                raise _Retain("background or compound shell syntax")
            else:
                if not words and c == "[" and self.text[self.i+1:self.i+2] in (" ", "\t"):
                    self.i += 1
                    words.append("[")
                else:
                    words.append(self.word(depth, end))


def retention_reason(command: str) -> str | None:
    if not isinstance(command, str) or not command.strip() or _DYNAMIC in command or _GLOB in command:
        return "empty or invalid shell input"
    try:
        _Shell(command).sequence()
    except _Retain as reason:
        return str(reason)
    return None


def bash_is_read_only(cmd: str) -> bool:
    return retention_reason(cmd) is None
