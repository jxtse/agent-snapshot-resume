# Pi CLI fixture

Pi 1.4.0 generated the session with four built-in tools and the scripted Chat Completions
upstream in `examples/offline/pi_probe.py`. Each resumed capture contains the request body
and the bundle run ID used to generate its sentinel.

The public fixture uses portable working and installation directories. Both original HTTP
bytes and independent resume captures received the same path substitutions. The reference
hashes were recomputed over the relocated original bytes. Running the probe in a fresh
directory produces unmodified local captures and hashes.
