#!/usr/bin/env python3
"""LSP proxy for ltex-ls-plus that drops diagnostics located in Typst code instead of prose."""

import json, os, re, subprocess, sys, threading

IDENT = re.compile(r"[\w-]+")
REF = re.compile(r"@[\w:-]+")
LABEL = re.compile(r"<[\w:.-]+>")
STATEMENTS = {"set", "show", "let", "import", "include", "if", "for", "while"}
OPEN = {"(": ")", "[": "]", "{": "}"}


def code_mask(text):
    """Mark every character that is not prose: code, comments, raw text, math, labels and references."""
    mask, n = bytearray(len(text)), len(text)

    def mark(a, b):
        mask[a:b] = b"\1" * (min(b, n) - a)

    def comment(i):
        if text.startswith("//", i):
            end = text.find("\n", i)
            return n if end < 0 else end
        end = text.find("*/", i)
        return n if end < 0 else end + 2

    def string(i):
        j = i + 1
        while j < n and text[j] != '"':
            j += 2 if text[j] == "\\" else 1
        return j + 1

    def balanced(i):
        stack, j = [OPEN[text[i]]], i + 1
        while j < n and stack:
            c = text[j]
            if text.startswith(("//", "/*"), j):
                j = comment(j)
            elif c == '"':
                j = string(j)
            elif c == "\\":
                j += 2
            else:
                if c in OPEN:
                    stack.append(OPEN[c])
                elif c == stack[-1]:
                    stack.pop()
                j += 1
        return j

    def markup(i, stop=None):
        while i < n and text[i] != stop:
            c = text[i]
            if c == "\\":
                i += 2
            elif text.startswith(("//", "/*"), i):
                end = comment(i)
                mark(i, end)
                i = end
            elif c == "`":
                ticks = len(text[i:]) - len(text[i:].lstrip("`"))
                end = text.find("`" * ticks, i + ticks)
                end = n if end < 0 else end + ticks
                mark(i, end)
                i = end
            elif c == "$":
                end = text.find("$", i + 1)
                end = n if end < 0 else end + 1
                mark(i, end)
                i = end
            elif c == "@" and (found := REF.match(text, i)):
                mark(i, found.end())
                i = found.end()
            elif c == "<" and (found := LABEL.match(text, i)):
                mark(i, found.end())
                i = found.end()
            elif c == "#":
                i = code(i)
            elif c == "[":
                i = markup(i + 1, "]") + 1
            else:
                i += 1
        return i

    def arguments(i):
        end = balanced(i)
        j = i
        while j < end:
            if text[j] == "[" and text[:j].rstrip().endswith("caption:"):
                mark(i, j + 1)
                close = markup(j + 1, "]")
                i = close
                j = close + 1
            elif text[j] == '"':
                j = string(j)
            else:
                j += 1
        mark(i, end)
        return end

    def code(i):
        found = IDENT.match(text, i + 1)
        if not found:
            mark(i, i + 1)
            return i + 1
        name, j = found.group(), found.end()
        if name in STATEMENTS:
            start = i
            while j < n and text[j] != "\n":
                j = balanced(j) if text[j] in OPEN else string(j) if text[j] == '"' else j + 1
            mark(start, j)
            return j
        mark(i, j)
        while j < n:
            if text[j] == "(":
                j = arguments(j)
            elif text[j] == "[":
                if name == "todo":
                    end = balanced(j)
                    mark(j, end)
                    j = end
                else:
                    mark(j, j + 1)
                    j = markup(j + 1, "]")
                    mark(j, j + 1)
                    j += 1
            elif text[j] == "." and (member := IDENT.match(text, j + 1)):
                mark(j, member.end())
                j = member.end()
            else:
                break
        return j

    markup(0)
    return mask


def offsets(text):
    starts = [0]
    for found in re.finditer("\n", text):
        starts.append(found.end())
    return starts


def in_code(diagnostic, text, mask, starts):
    r = diagnostic["range"]
    a = starts[min(r["start"]["line"], len(starts) - 1)] + r["start"]["character"]
    b = starts[min(r["end"]["line"], len(starts) - 1)] + r["end"]["character"]
    return any(mask[a : max(b, a + 1)])


def frame(obj):
    data = json.dumps(obj).encode()
    return b"Content-Length: %d\r\n\r\n" % len(data) + data


def read(stream):
    size = 0
    while True:
        line = stream.readline()
        if not line:
            return None
        if line.lower().startswith(b"content-length:"):
            size = int(line.split(b":")[1])
        elif line in (b"\r\n", b"\n"):
            return json.loads(stream.read(size))


documents = {}
server = subprocess.Popen([os.environ.get("LTEX_SERVER", "ltex-ls-plus")] + sys.argv[1:], stdin=subprocess.PIPE, stdout=subprocess.PIPE)


def from_client():
    while (message := read(sys.stdin.buffer)) is not None:
        params = message.get("params") or {}
        if message.get("method") == "textDocument/didOpen":
            documents[params["textDocument"]["uri"]] = params["textDocument"]["text"]
        elif message.get("method") == "textDocument/didChange":
            documents[params["textDocument"]["uri"]] = params["contentChanges"][-1]["text"]
        server.stdin.write(frame(message))
        server.stdin.flush()


def from_server():
    while (message := read(server.stdout)) is not None:
        if message.get("method") == "textDocument/publishDiagnostics":
            params = message["params"]
            text = documents.get(params["uri"])
            if text is not None and params["uri"].endswith(".typ"):
                mask, starts = code_mask(text), offsets(text)
                params["diagnostics"] = [d for d in params["diagnostics"] if not in_code(d, text, mask, starts)]
        sys.stdout.buffer.write(frame(message))
        sys.stdout.buffer.flush()


threading.Thread(target=from_client, daemon=True).start()
from_server()
