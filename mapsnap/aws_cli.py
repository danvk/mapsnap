"""Calling the AWS CLI with retries, shared by the corpus jobs (#354).

Lived in ``loc_craft`` until the work queue needed it too, and importing it from
there would have made a cycle. ``loc_craft`` re-exports these names, so the
sibling jobs that import them from it keep working.

Retrying is not optional in this fleet: a freshly booted instance loses its
first call to an IMDS credential race, which reports no error text at all --
hence the exit status in the message.
"""

import subprocess
import sys
import time

AWS_ATTEMPTS = 4
AWS_BACKOFF_SECONDS = 3.0


def run_aws(
    command: list[str], *, capture: bool = False
) -> subprocess.CompletedProcess:
    """Run an aws CLI command, retrying transient failures with a growing delay.

    Raises OSError with whatever the CLI said (and its exit status, since a
    credential race reports nothing at all) once the attempts are spent.
    """
    last = ""
    status = 0
    for attempt in range(AWS_ATTEMPTS):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return result
        status = result.returncode
        last = (result.stderr or result.stdout or "").strip()
        if attempt + 1 < AWS_ATTEMPTS:
            delay = AWS_BACKOFF_SECONDS * 2**attempt
            print(
                f"  aws {command[1]} {command[2]} failed (exit {status}), "
                f"retrying in {delay:.0f}s: {last[:120]}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise OSError(
        f"{' '.join(command[:4])} failed after {AWS_ATTEMPTS} attempts "
        f"(exit {status}): {last or 'no output'}"
    )
