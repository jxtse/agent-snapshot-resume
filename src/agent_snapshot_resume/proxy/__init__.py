"""In-sandbox model proxies.

passthrough_proxy.py  key-injecting Anthropic-messages passthrough (127.0.0.1:4013). The real key lives only in this
                      process's memory; the agent is given a dummy key.
resume_proxy.py       resume rewrite proxy (127.0.0.1:4031). Removes resume-only artifacts from every request so the
                      model sees the original context, optionally adds one hint block, pins the original date.
runtime.py            shell snippets that install a pinned Python runtime + hash-pinned aiohttp and start both proxies.
Both proxy scripts run INSIDE the sandbox under that runtime; they import nothing from this package.
"""
