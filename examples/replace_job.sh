#!/usr/bin/env bash
# Explicit mutation: submit -> verify/read back -> stop designated old job.
# PLATFORM_ADAPTER is an executable adapter YOU implement for your platform.
# Do not run against production until this replacement is authorized.
set -Eeuo pipefail
: "${PLATFORM_ADAPTER:?Set an executable platform adapter}"
: "${REQUEST_FILE:?Set validated replacement request JSON}"
: "${OLD_JOB_ID:?Set the exact authorized old job ID}"
PYTHON=${CHECK_PYTHON:-python3}
[[ -x "$PLATFORM_ADAPTER" && -r "$REQUEST_FILE" ]] || { echo 'Missing adapter/request' >&2; exit 2; }
# Adapter contract: stdout contains ONLY normalized JSON, diagnostics on stderr.
# submit REQUEST -> {"job_id":"..."}
# detail ID -> {"job_id":"...","status":"Pending|Inqueue|Starting|Running|Stopped|Failed|Succeeded","configuration_verified":true}
# stop ID -> {"job_id":"...","stopped":true} (adapter must poll to confirm).
NEW_JOB_ID=$("$PLATFORM_ADAPTER" submit "$REQUEST_FILE" | "$PYTHON" -c '
import json,sys
v=json.load(sys.stdin).get("job_id")
if not isinstance(v,str) or not v.strip(): raise SystemExit("Create response lacks a valid job_id; old job preserved")
print(v)
')
[[ "$NEW_JOB_ID" != "$OLD_JOB_ID" ]] || { echo 'Replacement ID equals old ID; refusing stop' >&2; exit 2; }
printf 'Created replacement: %s\n' "$NEW_JOB_ID"
"$PLATFORM_ADAPTER" detail "$NEW_JOB_ID" | "$PYTHON" -c '
import json,sys
v=json.load(sys.stdin)
if v.get("job_id")!=sys.argv[1] or v.get("configuration_verified") is not True or v.get("status") not in {"Pending","Inqueue","Starting","Running"}:
 raise SystemExit("Replacement readback not verified; old job preserved. Inspect replacement before retrying")
' "$NEW_JOB_ID"
"$PLATFORM_ADAPTER" stop "$OLD_JOB_ID" | "$PYTHON" -c '
import json,sys
v=json.load(sys.stdin)
if v.get("job_id")!=sys.argv[1] or v.get("stopped") is not True:
 raise SystemExit("Old job stop not confirmed; inspect both job states")
' "$OLD_JOB_ID"
printf 'Replacement verified; old job stopped: %s\n' "$OLD_JOB_ID"
