"""agent-snapshot-resume: resume an agent trajectory from step k on an E2B-compatible sandbox.

Two things are restored before the agent continues:
  1. the environment, by cloning the sandbox from a snapshot taken after replaying the side effects of steps < k;
  2. the conversation, by resuming the agent's native session truncated right before step k, behind a
     request-rewriting proxy that removes resume-only artifacts so the model sees the original context.
"""
__version__ = "0.1.0"
