#!/usr/bin/env bash
# launch_uniform_campaign.sh <family>...  -- the named batches of scripts/run_uniform_campaign.sh, one after the
# other, as one transient user service, which outlives the shell that starts it.
#   scripts/launch_uniform_campaign.sh b0 b1 b2 b3 parties     # the three smaller event logs and the party count
#   scripts/launch_uniform_campaign.sh b4                      # RTFM in full, activity (needs the swap file)
# The campaign directory is the newest output/campaign_uniform_*, else today's; CAMPAIGN_STAMP overrides, and a
# second launch into the same directory resumes it. After every family scripts/check_uniform_campaign.py runs and
# the remaining families do not start when it fails. The clock is tested first: no launch beside
# systemd-timesyncd, and none when the monotonic clock leaves an NTP server over 30 s (CLOCK_UNTESTED_OK=1 starts
# without an NTP answer).
set -euo pipefail
cd "$(dirname "$0")/.."
UNIT=uniform-campaign
[ $# -gt 0 ] || { echo "usage: $0 <family>...   (scripts/run_uniform_campaign.sh --list-families)" >&2; exit 2; }

known=$(scripts/run_uniform_campaign.sh --list-families | awk '{print $2}')
for f in "$@"; do
    grep -qx -- "$f" <<<"$known" || { echo "!!! unknown family '$f'; known: $(echo $known)" >&2; exit 2; }
    case "$f" in
        b4|b5|b6|b7|b8)
            # A full-log compile needs far more than this host's memory.
            swapon --show=NAME --noheadings | grep -qx /swapfile-campaign || {
                echo "!!! $f is a full-log RTFM batch: activate /swapfile-campaign (swapon) first" >&2
                exit 1; } ;;
    esac
done
if systemctl --user is-active --quiet "$UNIT"; then
    echo "!!! the unit $UNIT is already running (systemctl --user status $UNIT)" >&2; exit 1; fi
if pgrep -f -- '-party\.x' >/dev/null; then
    echo "!!! an MP-SPDZ party is running; the lane is busy" >&2; exit 1; fi
PY=python3; [ -x .venv/bin/python ] && PY=.venv/bin/python
if systemctl is-active --quiet systemd-timesyncd; then
    echo "!!! systemd-timesyncd is active: a second time source beside the one of WSL" >&2
    echo "!!!   sudo systemctl disable --now systemd-timesyncd" >&2; exit 1; fi
echo "testing the monotonic clock against NTP for 30 s ..."
clock_rc=0; "$PY" scripts/clock_probe.py selftest 30 || clock_rc=$?
case "$clock_rc" in
    0) ;;
    4) [ "${CLOCK_UNTESTED_OK:-0}" = "1" ] || {
           echo "!!! no NTP server answered, the clock is not tested (CLOCK_UNTESTED_OK=1 starts anyway)" >&2; exit 1; } ;;
    *) echo "!!! the monotonic clock does not run at the rate of real time; nothing is started" >&2; exit 1 ;;
esac

if [ -n "${CAMPAIGN_STAMP:-}" ]; then STAMP="$CAMPAIGN_STAMP"
else
    latest=$(ls -dt output/campaign_uniform_* 2>/dev/null | head -1 || true)
    if [ -n "$latest" ]; then STAMP="${latest#output/campaign_uniform_}"; else STAMP=$(date +%Y-%m-%d); fi
fi
ROOT="output/campaign_uniform_${STAMP}"
mkdir -p "$ROOT"

# The code state of this launch: the commit, what differs from it, and the untracked files the campaign reads.
now=$(date +%Y%m%dT%H%M%S)
{ echo "launch: $(date -Is)  families: $*"; echo "commit: $(git rev-parse HEAD)"; git status --short
  git ls-files --others --exclude-standard -- scripts config src mp-spdz bridge | xargs -r sha256sum; echo; } >> "$ROOT/code_state.txt"
git diff HEAD > "$ROOT/code_state_${now}.patch"

# The batch loop of this launch, kept in the campaign directory; b0 is checked strictly (no refusal expected).
LOOP="$ROOT/launch_${now}.sh"
{
    echo '#!/usr/bin/env bash'
    echo "for f in $*; do"
    echo '    echo "=== family $f  $(date -Is) ==="'
    echo '    /bin/bash scripts/run_uniform_campaign.sh --family "$f"'
    echo '    strict=""; [ "$f" = b0 ] && strict="--strict-family b0"'
    echo "    if ! $PY scripts/check_uniform_campaign.py $ROOT \$strict; then"
    echo '        echo "!!! the check failed after family $f: the remaining families do not start"; exit 1'
    echo '    fi'
    echo 'done'
    echo 'echo "=== all families of this launch are through  $(date -Is) ==="'
} > "$LOOP"

systemctl --user reset-failed "$UNIT" 2>/dev/null || true
systemd-run --user --unit="$UNIT" -p OOMPolicy=continue \
    -E HOME="$HOME" -E USER="${USER:-$(id -un)}" -E LANG=C.UTF-8 -E SHELL=/bin/bash \
    -E PATH="$HOME/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    -E CAMPAIGN_STAMP="$STAMP" --working-directory="$PWD" \
    /bin/bash -c "exec /bin/bash $LOOP >> $ROOT/driver.log 2>&1 </dev/null"
echo "started: user unit $UNIT, families: $*"
echo "  log:    tail -f $ROOT/driver.log"
echo "  result: column -t $ROOT/summary.tsv"
echo "  check:  $PY scripts/check_uniform_campaign.py $ROOT"
echo "  state:  systemctl --user status $UNIT        stop: systemctl --user stop $UNIT"
