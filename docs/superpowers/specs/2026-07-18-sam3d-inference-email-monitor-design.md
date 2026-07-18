# SAM3D-Body Inference Email Monitor Design

## Goal

Monitor the active SAM3D-Body inference run for the newly added gymnastics
videos and send an email to `chenkaixusan@gmail.com` when the run completes or
when it encounters a material problem.

The monitor must run independently in tmux, survive terminal disconnects, avoid
duplicate notifications, and keep email credentials out of the repository.

## Scope

The monitored person IDs are:

```text
69-134, 136-138
```

There are 69 target persons. ID135 is intentionally excluded because its source
videos are absent.

The active inference writes to:

```text
/home/data/xchen/gymnastics/sam3d_body_results/person
```

The active stable-run log is:

```text
/home/data/xchen/gymnastics/sam3d_body_results/logs/sam3dbody_new_20260718_w2.stdout.log
```

## Selected Approach

Use a standalone Python monitor with the standard library only. It polls local
process, log, and output state, then sends mail through Gmail SMTP over SSL.

This is preferred over a local mail relay because the machine has no
`sendmail`, `mailx`, or `msmtp` installation. A third-party mail API would add an
unnecessary account, dependency, and token.

## Components

### Monitor Script

Add `scripts/monitor_sam3d_inference.py` with these responsibilities:

- Load runtime settings from command-line arguments.
- Load SMTP settings from a private key-value configuration file.
- Poll the active inference every 60 seconds.
- Summarize person completion and generated NPZ counts.
- Scan only the stable-run log for newly observed fatal errors.
- Detect a stopped inference process before all persons finish.
- Detect output inactivity lasting at least 30 minutes.
- Send deduplicated problem and completion notifications.
- Persist monitor state atomically between polls and restarts.

### Private SMTP Configuration

Store SMTP settings outside the repository at:

```text
/home/workspace/kaixu/.config/gymnastics/sam3d_monitor.env
```

Required keys:

```text
SMTP_HOST=smtp.gmail.com
SMTP_PORT=465
SMTP_USER=chenkaixusan@gmail.com
SMTP_APP_PASSWORD=YOUR_GMAIL_APPLICATION_PASSWORD
EMAIL_TO=chenkaixusan@gmail.com
```

The monitor requires file mode `0600`. It must never print the password or add
the configuration file to Git.

### Persistent Monitor State

Store non-secret state at:

```text
/home/data/xchen/gymnastics/sam3d_body_results/logs/sam3dbody_monitor_state.json
```

The state records:

- last observed NPZ count and latest NPZ modification time;
- time of the last observed progress;
- last scanned byte offset in the stable-run log;
- notification fingerprints already sent;
- whether a terminal completion notification was sent.

State writes use a temporary sibling file followed by an atomic replacement.

## Status Detection

### Person Completion

A target person is complete only when all of the following are true:

- its person log contains `==== Finished Person: {person_id} ====`;
- its `face` output directory contains at least one `*_sam3d_body.npz` file;
- its `side` output directory contains at least one `*_sam3d_body.npz` file.

This avoids treating an empty directory created during startup as completed.

### Run Completion

The run is complete when all 69 target persons satisfy the person-completion
rules. The monitor sends one completion email containing elapsed monitor time,
completed-person count, total NPZ count, and output path, then exits with status
zero.

### Fatal Log Errors

New stable-run log content is checked for material failure patterns including:

- `CUDA out of memory` or `OutOfMemoryError`;
- `Traceback`;
- worker errors matching `处理 ... 时出错`;
- `CUDA error`, `Killed`, or `No module named`.

Ordinary warnings and per-frame `No person detected` messages are not fatal.
Each distinct fatal-error fingerprint generates at most one email. The monitor
continues after sending so it can report later recovery or final failure.

### Premature Process Exit

The inference is active only when a process command line contains both
`python -m SAM3Dbody.main` and the configured `--process-match` value. The
default match value is `infer.workers_per_gpu=2`, which identifies the active
stable run. The tmux session alone is insufficient because its shell remains
after inference exits.

If no matching process exists while any target person is incomplete, the
monitor sends an incomplete-run email with the missing person IDs and exits with
nonzero status.

### Stalled Output

Progress means either the total target NPZ count increases or the newest target
NPZ modification time advances. If the inference process remains active but no
progress occurs for 30 minutes, the monitor sends one stall notification. It
keeps monitoring and clears the active stall condition when output resumes.

## Email Behavior

Use `smtplib.SMTP_SSL` with Gmail on port 465. Messages use a stable prefix:

```text
[Gymnastics][SAM3D-Body]
```

Notification types are:

- `COMPLETED`: all target persons completed;
- `ERROR`: a fatal log event was observed;
- `STOPPED`: inference exited with incomplete persons;
- `STALLED`: no output progress for the configured threshold;
- `RECOVERED`: output resumed after a reported stall.

Mail sending is retried up to three times per poll. A notification is marked as
sent only after SMTP confirms delivery, allowing a temporary network failure to
be retried on the next poll.

## Command-Line Interface

The script exposes explicit defaults for the current run and allows overrides:

```text
--person-ids
--result-root
--person-log-root
--run-log
--state-file
--smtp-config
--poll-seconds
--stall-seconds
--process-match
--once
```

`--once` performs one poll without sleeping and is used by tests and manual
diagnostics.

## Testing

Add focused tests using temporary directories and a mocked SMTP client. Tests
cover:

- target ID parsing, including ranges and the ID135 gap;
- person and whole-run completion detection;
- empty/partial output directories remaining incomplete;
- fatal log scanning from a persisted byte offset;
- active-process, stopped-process, and stall decisions;
- notification deduplication across state reloads;
- SMTP success, retry, and credential redaction;
- one-shot mode without real email or sleeping.

No test sends a real email or reads the private SMTP configuration.

## Deployment

After tests pass and the Gmail application password is configured, launch the
monitor in a separate tmux session named:

```text
sam3dbody_monitor_20260718
```

Write monitor stdout and stderr to:

```text
/home/data/xchen/gymnastics/sam3d_body_results/logs/sam3dbody_monitor_20260718.log
```

The monitor must perform an initial status poll before sleeping so that an
already-failed or already-completed run is reported immediately.
