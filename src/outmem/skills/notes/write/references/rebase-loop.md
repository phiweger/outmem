# The pull–rebase–push loop

When a remote is configured, every writeback must reach the remote or
escalate as a hard error. The wiki's "single source of truth" property
depends on the remote actually receiving your commits.

## The contract

```bash
# Conceptually:
outmem pull              # git pull --rebase origin main
# … decide / write / commit …
outmem push              # git push origin main
```

If `pull` rebases your local commits on top of someone else's edits
that touched the same page, you may want to re-read the affected file
and reconsider — the human's commit may have invalidated the change
you were about to make.

If `push` is rejected (the remote moved between your pull and your
push), outmem retries the pull-rebase-push cycle **once**. A second
rejection — or a rebase conflict it can't resolve — surfaces as a hard
error (`WritebackError`). Do not paper over this. Do not retry blindly
in a loop.

## Why hard-error instead of silent retry

Silent push failure is one of the FAIL.md anti-patterns the spec
explicitly defends against. If your writeback didn't reach the remote,
the agent's view of the wiki diverges from the remote's view, the next
session reads stale state, and the steering signal collapses. Better
to surface "writeback failed" to the user and let them investigate
than to ship inconsistent state.

## What to do when writeback fails

1. **Tell the user.** Don't respond as if the commit succeeded.
2. **Inspect with `git status` and `git log`** in the wiki directory.
3. **Common causes:**
   - SSH key not loaded → `ssh-add`.
   - Branch protection on the remote → push to a different branch and
     open a PR (the wiki is supposed to be the single line of
     development, but emergencies are emergencies).
   - The remote diverged in a way the rebase couldn't resolve → human
     conflict resolution is required.

Outmem does not automatically resolve conflicts. A merge conflict
means two people (or one person and the agent) disagreed about a
specific line; the resolution requires intent the runtime doesn't have.
