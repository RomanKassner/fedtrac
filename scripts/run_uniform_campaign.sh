#!/usr/bin/env bash
# Uniform campaign driver: one design for the whole evaluation, run top-down on the single MP-SPDZ lane.
# THE DESIGN  four event logs, each split by role between two parties; a run clusters all shared cases; both
#   instantiations under five embeddings; Secret-Labels with and without the refinement; Revealed-Projection five
#   draws; auto-L, auto-k_proj, k = 6 for Secret-Labels, auto-k and the adaptive threshold for Revealed-Projection.
# SEVEN CELLS PER (event log, embedding), in slab order: ms_<log>_rp_<enc>_s42 (clean, its own time),
#   <log>-<enc>_sl_km (builds the slab), <log>-<enc>_sl_seq, ms_<log>_rp_<enc>_s{1337,7,101,202} (slab tails).
# FAMILIES  b0 Sepsis activity preflight, b1 Sepsis, b2 Helpdesk, b3 BPIC 2013, b4..b8 RTFM per embedding;
#   parties: every event log at two and three parties under bigram; c0..c3 the party count as a cost family.
# USAGE
#   scripts/run_uniform_campaign.sh --dry-run          # the inventory + hours
#   scripts/run_uniform_campaign.sh --list-families
#   scripts/run_uniform_campaign.sh --family b0        # one batch
#   scripts/run_uniform_campaign.sh --from <cell>      # resume
# ENV  CAMPAIGN_STAMP: campaign dir stamp (newest uniform dir, else today); the same dir resumes it.
# CLOCK  every executed cell gets a row in clock.tsv (scripts/clock_probe.py): its span on the monotonic clock,
#   which times every round, against real time and an NTP server, plus the CPU time outside the campaign.
# DONE  a cell is done when its last run exited 0, its clock row lies within the bound and no later entry of
#   void.tsv names it (scripts/check_uniform_campaign.py --is-done); the log of an earlier run moves to superseded/.
set -uo pipefail
cd "$(dirname "$0")/.."

# 0  Options
DRY_RUN=0
FAMILY=""
ONLY=""
FROM=""
LIST_FAMILIES=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)        DRY_RUN=1 ;;
        --list-families)  LIST_FAMILIES=1 ;;
        --family)         FAMILY="${2:?--family needs a name}"; shift ;;
        --family=*)       FAMILY="${1#--family=}" ;;
        --only)           ONLY="${2:?--only needs a regex}"; shift ;;
        --only=*)         ONLY="${1#--only=}" ;;
        --from)           FROM="${2:?--from needs a cell id}"; shift ;;
        --from=*)         FROM="${1#--from=}" ;;
        -h|--help)        sed -n '1,19p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

RING_ESCAPE="${RING_ESCAPE:-0}"
ARGMIN_C3="${ARGMIN_C3:-1}"
RING_FACTOR="${RING_FACTOR:-1.4}"
CAMPAIGN_Z="${CAMPAIGN_Z:-1}"
LANE_WAIT_MAX_S="${LANE_WAIT_MAX_S:-14400}"
PYTHON="${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}"
SPDZ_DIR="${MP_SPDZ_DIR:-$HOME/MP-SPDZ}"
# softspoken k = 6: the fewest Round-1 bytes at every measured pin at no cost in exec.
export CLUSTERING_MPC_PARTY_FLAGS="${CLUSTERING_MPC_PARTY_FLAGS:--o softspoken=6}"
PARTY_FLAGS="$CLUSTERING_MPC_PARTY_FLAGS"
# Compile flags on top of each cell's -Z/-R pair; they ride the bytecode dephash and the extract-slab key.
EXTRA_COMPILE_FLAGS="${EXTRA_COMPILE_FLAGS:--D --budget 4000}"
PROTOCOL="semi2k-party.x"
THREADS="${CLUSTERING_MPC_THREADS:-8}"
GENERATION=4


# 1  The cell table: add_cell <family> <cell> <est_s> [opts...] -- <args...>; n=/l= are checked against the run.
CELLS=()
add_cell() {
    local family=$1 cell=$2 est=$3; shift 3
    local u=intersection np=2 n=- l=- ring=64 reuse=- wipe=- trunc=- proto=- pf=- tmo=4h
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do
        case "$1" in
            u=*)     u=${1#u=} ;;
            np=*)    np=${1#np=} ;;
            n=*)     n=${1#n=} ;;
            l=*)     l=${1#l=} ;;
            ring=*)  ring=${1#ring=} ;;
            reuse)   reuse=reuse ;;
            wipe)    wipe=wipe ;;
            trunc)   trunc=1 ;;
            proto=*) proto=${1#proto=} ;;
            pf=*)    pf=${1#pf=} ;;
            tmo=*)   tmo=${1#tmo=} ;;
            *) echo "add_cell $cell: unknown option '$1'" >&2; exit 2 ;;
        esac
        shift
    done
    [ "${1:-}" = "--" ] && shift
    [ "$reuse" = "reuse" ] || wipe=wipe      # a clean cell always starts fresh
    CELLS+=("${family}"$'\t'"${cell}"$'\t'"${u}"$'\t'"${np}"$'\t'"${n}"$'\t'"${l}"$'\t'"${ring}"$'\t'"${reuse}"$'\t'"${wipe}"$'\t'"${trunc}"$'\t'"${proto}"$'\t'"${pf}"$'\t'"${tmo}"$'\t'"${est}"$'\t'"$*")
}

# ---- pins, populations, splits, modes -----------------------------------
PIN_SEPSIS="--case-ids config/case_ids/sepsis_2party_full.txt"
PIN_HELPDESK="--case-ids config/case_ids/helpdesk_full.txt"
PIN_BPIC="--case-ids config/case_ids/bpic2013_role_full.txt"
PIN_RTFM="--case-ids config/case_ids/rtfm_full.txt"
PIN_SEPSIS_3P="--case-ids config/case_ids/sepsis_3party_full.txt"
PIN_HELPDESK_3P="--case-ids config/case_ids/helpdesk_3party_full.txt"
PIN_BPIC_3P="--case-ids config/case_ids/bpic2013_3party_full.txt"
PIN_RTFM_3P="--case-ids config/case_ids/rtfm_rtseries_1184.txt"
N_SEPSIS=927; N_HELPDESK=4577; N_BPIC=7545; N_RTFM=129985
N_SEPSIS_3P=925; N_HELPDESK_3P=4221; N_BPIC_3P=4507; N_RTFM_3P=1184
K_SL=6
SEEDS_TAIL="1337 7 101 202"

SL="--mpc --secure-kmeans"
SEQ="--secure-clustering sequence"
RP="--mpc --reveal-secret-projection"

D_SEPSIS=data/splits/sepsis_2party
D_HELPDESK=data/splits/helpdesk_split
D_BPIC=data/splits/bpic2013_2party
D_RTFM=data/splits/road_traffic_split
D_SEPSIS_3P=data/splits/3party
D_HELPDESK_3P=data/splits/helpdesk_3party
D_BPIC_3P=data/splits/bpic2013_3party
D_RTFM_3P=data/splits/road_traffic_3party

# grid_combo <family> <log> <dir> <pin> <n> <encoding> <tag> <tmo> <est rp> <est sl> <est seq> <est tail>
grid_combo() {
    local fam=$1 log=$2 dir=$3 pin=$4 n=$5 enc=$6 tag=$7 tmo=$8 e_rp=$9 e_sl=${10} e_seq=${11} e_tail=${12}
    local base="--data-dir $dir $pin --encoding $enc" seed
    add_cell "$fam" "ms_${log}_rp_${tag}_s42" "$e_rp" n=$n l=auto tmo=$tmo -- \
        $base $RP --projection-seed 42
    add_cell "$fam" "${log}-${tag}_sl_km" "$e_sl" n=$n l=auto tmo=$tmo reuse -- \
        $base $SL --k $K_SL
    add_cell "$fam" "${log}-${tag}_sl_seq" "$e_seq" n=$n l=auto tmo=$tmo reuse -- \
        $base $SL $SEQ --k $K_SL
    for seed in $SEEDS_TAIL; do
        add_cell "$fam" "ms_${log}_rp_${tag}_s${seed}" "$e_tail" n=$n l=auto tmo=$tmo reuse -- \
            $base $RP --projection-seed "$seed"
    done
}

# 1A  b0 -- the preflight batch: Sepsis under activity, with a clean Secret-Labels twin.
add_cell b0 enc_sepsis_sl_activity 80 n=$N_SEPSIS l=auto -- \
    --data-dir $D_SEPSIS $PIN_SEPSIS --encoding activity $SL --k $K_SL
grid_combo b0 sepsis "$D_SEPSIS" "$PIN_SEPSIS" $N_SEPSIS activity activity 4h 100 80 140 30

# 1B  b1 -- Sepsis, the four other embeddings; both ef cells refuse by design (exit 1).
grid_combo b1 sepsis "$D_SEPSIS" "$PIN_SEPSIS" $N_SEPSIS bigram      bigram     4h  420  90 150 200
grid_combo b1 sepsis "$D_SEPSIS" "$PIN_SEPSIS" $N_SEPSIS ef          ef         4h    5   5   5   5
grid_combo b1 sepsis "$D_SEPSIS" "$PIN_SEPSIS" $N_SEPSIS skip-bigram skipbigram 4h  800  90 150 400
grid_combo b1 sepsis "$D_SEPSIS" "$PIN_SEPSIS" $N_SEPSIS trigram     trigram    4h 1700  90 150 850

# 1C  b2 -- Helpdesk.
grid_combo b2 helpdesk "$D_HELPDESK" "$PIN_HELPDESK" $N_HELPDESK activity    activity   4h  250 125 160 125
grid_combo b2 helpdesk "$D_HELPDESK" "$PIN_HELPDESK" $N_HELPDESK bigram      bigram     4h  300 130 160 150
grid_combo b2 helpdesk "$D_HELPDESK" "$PIN_HELPDESK" $N_HELPDESK ef          ef         4h  270 140 160 145
grid_combo b2 helpdesk "$D_HELPDESK" "$PIN_HELPDESK" $N_HELPDESK skip-bigram skipbigram 4h  880 470 300 440
grid_combo b2 helpdesk "$D_HELPDESK" "$PIN_HELPDESK" $N_HELPDESK trigram     trigram    4h 1140 480 300 570

# 1D  b3 -- BPIC 2013 by role (7,545 shared cases; the estimates are scaled, not measured).
grid_combo b3 bpic2013 "$D_BPIC" "$PIN_BPIC" $N_BPIC activity    activity   6h  410 360 360 205
grid_combo b3 bpic2013 "$D_BPIC" "$PIN_BPIC" $N_BPIC bigram      bigram     6h  840 360 360 420
grid_combo b3 bpic2013 "$D_BPIC" "$PIN_BPIC" $N_BPIC ef          ef         6h  910 480 480 455
grid_combo b3 bpic2013 "$D_BPIC" "$PIN_BPIC" $N_BPIC skip-bigram skipbigram 6h 1370 790 790 685
grid_combo b3 bpic2013 "$D_BPIC" "$PIN_BPIC" $N_BPIC trigram     trigram    6h  930 440 440 465

# 1E  b4..b8 -- RTFM in full, one embedding per batch, cheapest first; ef and skip-bigram need the big swap file.
grid_combo b4 rtfm "$D_RTFM" "$PIN_RTFM" $N_RTFM activity    activity   12h  3065  2006 2900 1000
grid_combo b5 rtfm "$D_RTFM" "$PIN_RTFM" $N_RTFM bigram      bigram     12h  6099  3700 3000 2000
grid_combo b6 rtfm "$D_RTFM" "$PIN_RTFM" $N_RTFM trigram     trigram    12h 16630 10000 5000 7300
grid_combo b7 rtfm "$D_RTFM" "$PIN_RTFM" $N_RTFM ef          ef         12h 15600  9400 4000 3400
grid_combo b8 rtfm "$D_RTFM" "$PIN_RTFM" $N_RTFM skip-bigram skipbigram 12h 20823 12500 5000 6300

# 1F  parties -- the party count as the only axis, under bigram, on the cases the three parties share.
# party_combo <log> <Log> <dir2> <dir3> <pin> <n> <est rp 2p> <est sl 2p> <est tail 2p>
party_combo() {
    local log=$1 Log=$2 dir2=$3 dir3=$4 pin=$5 n=$6 e_rp=$7 e_sl=$8 e_tail=$9
    local np dir npopt npflag seed f
    for np in 2 3; do
        if [ "$np" = "2" ]; then dir=$dir2; npopt=""; npflag=""; f=1
        else dir=$dir3; npopt="np=3"; npflag="--n-parties 3"; f=3; fi
        local base="--data-dir $dir $npflag $pin --encoding bigram"
        add_cell parties "ms_${log}-pc${np}p_rp_bigram_s42" $((e_rp * f)) $npopt n=$n l=auto -- \
            $base $RP --projection-seed 42
        add_cell parties "rt${Log}_sl_bigram_${np}p" $((e_sl * f)) $npopt n=$n l=auto reuse -- \
            $base $SL --k $K_SL
        for seed in $SEEDS_TAIL; do
            add_cell parties "ms_${log}-pc${np}p_rp_bigram_s${seed}" $((e_tail * f)) $npopt n=$n l=auto reuse -- \
                $base $RP --projection-seed "$seed"
        done
    done
}
party_combo sepsis   Sepsis   "$D_SEPSIS"   "$D_SEPSIS_3P"   "$PIN_SEPSIS_3P"   $N_SEPSIS_3P   420  90 200
party_combo rtfm     Rtfm     "$D_RTFM"     "$D_RTFM_3P"     "$PIN_RTFM_3P"     $N_RTFM_3P     200  35 100
party_combo helpdesk Helpdesk "$D_HELPDESK" "$D_HELPDESK_3P" "$PIN_HELPDESK_3P" $N_HELPDESK_3P 300 130 150
party_combo bpic2013 Bpic2013 "$D_BPIC"     "$D_BPIC_3P"     "$PIN_BPIC_3P"     $N_BPIC_3P     500 220 250

# 1G  c0..c3 -- the party count as a COST family: per event log, embedding and party count one Secret-Labels
#     k-means cell and one Revealed-Projection full draw (seed 42), no tail draws and no refinement cell;
#     4 event logs x 4 embeddings x 2 party counts x 2 cells = 64 cells, one batch per event log, the small
#     ones first. Bigram is not here: the recorded `parties` family above is its measurement. Sepsis under ef
#     refuses by design at both party counts (4 of the 64 cells). The estimates are scaled, not measured.
# party_cost_combo <family> <log> <Log> <dir2> <dir3> <pin> <n> <encoding> <tag> <f10> <est rp 2p> <est sl 2p>
party_cost_combo() {
    local fam=$1 log=$2 Log=$3 dir2=$4 dir3=$5 pin=$6 n=$7 enc=$8 tag=$9
    local f10=${10} e_rp=${11} e_sl=${12}
    local np dir npopt npflag f base
    for np in 2 3; do
        if [ "$np" = "2" ]; then dir=$dir2; npopt=""; npflag=""; f=10
        else dir=$dir3; npopt="np=3"; npflag="--n-parties 3"; f=$f10; fi
        base="--data-dir $dir $npflag $pin --encoding $enc"
        add_cell "$fam" "ms_${log}-pc${np}p_rp_${tag}_s42" $((e_rp * f / 10)) $npopt n=$n l=auto -- \
            $base $RP --projection-seed 42
        add_cell "$fam" "rt${Log}_sl_${tag}_${np}p" $((e_sl * f / 10)) $npopt n=$n l=auto reuse -- \
            $base $SL --k $K_SL
    done
}

# c0  Sepsis (925 shared cases; the ef pair refuses at both party counts)
party_cost_combo c0 sepsis Sepsis "$D_SEPSIS" "$D_SEPSIS_3P" "$PIN_SEPSIS_3P" $N_SEPSIS_3P \
    activity    activity   20    92  98
party_cost_combo c0 sepsis Sepsis "$D_SEPSIS" "$D_SEPSIS_3P" "$PIN_SEPSIS_3P" $N_SEPSIS_3P \
    ef          ef         20     5   5
party_cost_combo c0 sepsis Sepsis "$D_SEPSIS" "$D_SEPSIS_3P" "$PIN_SEPSIS_3P" $N_SEPSIS_3P \
    skip-bigram skipbigram 20   784 530
party_cost_combo c0 sepsis Sepsis "$D_SEPSIS" "$D_SEPSIS_3P" "$PIN_SEPSIS_3P" $N_SEPSIS_3P \
    trigram     trigram    20  1782 591

# c1  Helpdesk (4,221 shared cases)
party_cost_combo c1 helpdesk Helpdesk "$D_HELPDESK" "$D_HELPDESK_3P" "$PIN_HELPDESK_3P" $N_HELPDESK_3P \
    activity    activity   29   297 104
party_cost_combo c1 helpdesk Helpdesk "$D_HELPDESK" "$D_HELPDESK_3P" "$PIN_HELPDESK_3P" $N_HELPDESK_3P \
    ef          ef         29   418 160
party_cost_combo c1 helpdesk Helpdesk "$D_HELPDESK" "$D_HELPDESK_3P" "$PIN_HELPDESK_3P" $N_HELPDESK_3P \
    skip-bigram skipbigram 29   789 448
party_cost_combo c1 helpdesk Helpdesk "$D_HELPDESK" "$D_HELPDESK_3P" "$PIN_HELPDESK_3P" $N_HELPDESK_3P \
    trigram     trigram    29  1068 437

# c2  BPIC 2013 by role (4,507 shared cases)
party_cost_combo c2 bpic2013 Bpic2013 "$D_BPIC" "$D_BPIC_3P" "$PIN_BPIC_3P" $N_BPIC_3P \
    activity    activity   24   348 155
party_cost_combo c2 bpic2013 Bpic2013 "$D_BPIC" "$D_BPIC_3P" "$PIN_BPIC_3P" $N_BPIC_3P \
    ef          ef         24   435 197
party_cost_combo c2 bpic2013 Bpic2013 "$D_BPIC" "$D_BPIC_3P" "$PIN_BPIC_3P" $N_BPIC_3P \
    skip-bigram skipbigram 24   700 377
party_cost_combo c2 bpic2013 Bpic2013 "$D_BPIC" "$D_BPIC_3P" "$PIN_BPIC_3P" $N_BPIC_3P \
    trigram     trigram    24   437 165

# c3  RTFM on the 1,184 cases its three parties share (NOT the full log)
party_cost_combo c3 rtfm Rtfm "$D_RTFM" "$D_RTFM_3P" "$PIN_RTFM_3P" $N_RTFM_3P \
    activity    activity   21    47  21
party_cost_combo c3 rtfm Rtfm "$D_RTFM" "$D_RTFM_3P" "$PIN_RTFM_3P" $N_RTFM_3P \
    ef          ef         21   190  43
party_cost_combo c3 rtfm Rtfm "$D_RTFM" "$D_RTFM_3P" "$PIN_RTFM_3P" $N_RTFM_3P \
    skip-bigram skipbigram 21   254  79
party_cost_combo c3 rtfm Rtfm "$D_RTFM" "$D_RTFM_3P" "$PIN_RTFM_3P" $N_RTFM_3P \
    trigram     trigram    21   301  67

# 2  Filters, families, dry run
effective_ring() { # <ring field> -> 64 | 128
    case "$1" in
        128g) [ "$ARGMIN_C3" = "1" ] && echo 64 || echo 128 ;;
        128)  echo 128 ;;
        *)    echo 64 ;;
    esac
}

selected() { # <family> <cell> -> 0 if this cell is in scope
    local fam=$1 cell=$2
    [ -z "$FAMILY" ] || [ "$fam" = "$FAMILY" ] || return 1
    [ -z "$ONLY" ] || [[ "$cell" =~ $ONLY ]] || return 1
    return 0
}

if [ "$LIST_FAMILIES" = "1" ]; then
    for rec in "${CELLS[@]}"; do printf '%s\n' "${rec%%$'\t'*}"; done | sort | uniq -c
    exit 0
fi

# The cells this invocation is responsible for, computed once so every stage agrees on them.
SEL=()
seen_from=0
[ -n "$FROM" ] || seen_from=1
for rec in "${CELLS[@]}"; do
    IFS=$'\t' read -r fam cell _rest <<<"$rec"
    selected "$fam" "$cell" || continue
    if [ "$seen_from" = "0" ]; then
        [ "$cell" = "$FROM" ] && seen_from=1 || continue
    fi
    SEL+=("$rec")
done
if [ "$seen_from" = "0" ]; then
    echo "!!! --from '$FROM' matched no cell in scope (family='${FAMILY}', only='${ONLY}')" >&2
    exit 1
fi
if [ "${#SEL[@]}" -eq 0 ]; then
    echo "!!! no cell selected (family='${FAMILY}', only='${ONLY}'); try --list-families" >&2
    exit 1
fi

# Slab continuity: a reuse row without a same-key predecessor builds its own slab.
slab_key() { # <universe> <np> <ring> <proto> <args...>
    local u=$1 np=$2 ring=$3 proto=$4; shift 4
    local dd= ci= mc= mt= enc= prev=
    for a in "$@"; do
        case "$prev" in
            --data-dir)      dd=$a ;;
            --case-ids)      ci=$a ;;
            --max-trace-len) mt=$a ;;
            --encoding)      enc=$a ;;
        esac
        prev=$a
    done
    printf '%s|%s|%s|%s|%s|%s|%s|%s|%s' "$dd" "$ci" "$mc" "$mt" "${enc:-bigram}" "$u" "$np" "$ring" "$proto"
}
prev_key=""; prev_reuse=""
for i in "${!SEL[@]}"; do
    IFS=$'\t' read -r fam cell u np n l ring reuse wipe trunc proto pf tmo est args <<<"${SEL[$i]}"
    # shellcheck disable=SC2086
    key=$(slab_key "$u" "$np" "$ring" "$proto" $args)
    if [ "$reuse" = "reuse" ] && [ "$wipe" != "wipe" ] && \
       { [ "$prev_reuse" != "reuse" ] || [ "$prev_key" != "$key" ]; }; then
        wipe=wipe
        SEL[$i]="${fam}"$'\t'"${cell}"$'\t'"${u}"$'\t'"${np}"$'\t'"${n}"$'\t'"${l}"$'\t'"${ring}"$'\t'"${reuse}"$'\t'"${wipe}"$'\t'"${trunc}"$'\t'"${proto}"$'\t'"${pf}"$'\t'"${tmo}"$'\t'"${est}"$'\t'"${args}"
        echo "[schedule] ${cell}: no same-key slab precedes it in this launch -> builds its own (wipe)" >&2
    fi
    prev_key=$key; prev_reuse=$reuse
done

if [ "$DRY_RUN" = "1" ]; then
    printf 'UNIFORM CAMPAIGN INVENTORY (dry run; no cell is executed)\n'
    printf 'ring_escape=%s argmin_c3=%s ring_factor=%s party_flags="%s" compile_flags="%s" protocol=%s threads=%s\n\n' \
        "$RING_ESCAPE" "$ARGMIN_C3" "$RING_FACTOR" "$PARTY_FLAGS" "${EXTRA_COMPILE_FLAGS:-none}" "$PROTOCOL" "$THREADS"
    {
        printf 'family\tcell\tuniverse\tnp\tN\tL\tring\tround1\test_h\n'
        for rec in "${SEL[@]}"; do
            IFS=$'\t' read -r fam cell u np n l ring reuse wipe trunc proto pf tmo est args <<<"$rec"
            r=$(effective_ring "$ring")
            r1=measured; [ "$reuse" = "reuse" ] && r1=reuse
            [ "$reuse" = "reuse" ] && [ "$wipe" = "wipe" ] && r1=reuse-build
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$fam" "$cell" "$u" "$np" \
                "$n" "$l" "$r" "$r1" \
                "$(awk -v e="$est" -v r="$r" -v f="$RING_FACTOR" \
                     'BEGIN{printf "%.2f", e*(r==128?f:1)/3600}')"
        done
    } | column -t
    echo
    {
        for rec in "${SEL[@]}"; do
            IFS=$'\t' read -r fam cell u np n l ring reuse wipe trunc proto pf tmo est args <<<"$rec"
            printf '%s\t%s\t%s\t%s\n' "$fam" "$est" "$(effective_ring "$ring")" "$reuse"
        done
    } | RING_FACTOR="$RING_FACTOR" awk -F'\t' '
        { order[$1] = (order[$1] ? order[$1] : ++seq); n[$1]++;
          h = $2 * ($3 == 128 ? ENVIRON["RING_FACTOR"] : 1);
          s[$1] += h; tot += h; cells++;
          if ($4 == "reuse") reuse++; else clean++ }
        END {
          printf "%-14s %6s %10s\n", "family", "cells", "est. h";
          for (f in order) inv[order[f]] = f;
          for (i = 1; i <= seq; i++) { f = inv[i];
              printf "%-14s %6d %10.2f\n", f, n[f], s[f]/3600 }
          printf "%-14s %6d %10.2f\n", "TOTAL", cells, tot/3600;
          printf "\nRound 1 measured (clean): %d cells   reused: %d cells\n", clean, reuse }'
    echo
    echo "Estimates: a clean cell carries the recorded wall clock of its gen-4 twin where one"
    echo "exists (Sepsis, Helpdesk, the full-log RTFM Revealed-Projection cells); every slab"
    echo "builder, every tail and every BPIC 2013 and party-count cell is UNMEASURED, scaled"
    echo "by hand from the nearest recorded cell."
    echo "The c0..c3 estimates are scaled from the recorded two-party bigram cell of the same"
    echo "event log and kind by the embedding's ratio in that log's grid batch, the three-party"
    echo "ones by the measured 2p->3p factor of the recorded bigram family. None is measured."
    exit 0
fi

# 3  Campaign dirs, meta, resume guards
if [ -n "${CAMPAIGN_STAMP:-}" ]; then
    STAMP="$CAMPAIGN_STAMP"
else
    latest=$(ls -dt output/campaign_uniform_* 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        STAMP="${latest#output/campaign_uniform_}"
        echo "[resume] continuing ${latest} (set CAMPAIGN_STAMP to start a new campaign)"
    else
        STAMP=$(date +%Y-%m-%d)
    fi
fi
ROOT="output/campaign_uniform_${STAMP}"
mkdir -p "$ROOT" "$ROOT/union"
COMMIT=$(git rev-parse --short HEAD)

SUMMARY_HDR='cell\texit\tfamily\tparties\tn\tL\tk\tk_priv\tsuppressed\twall_s\tround1_s\tkmeans_s\tseq_s\tround2_s\tglobal_mb\tgeneration\tparty_flags\tring\tring_overflow\treuse\tcase_universe\tround1_restarts\n'
# The manifest columns aggregate_campaign.py reads come first, in its order.
MANIFEST_HDR='cell\trun_dir\tcommit\tprotocol\tparty_flags\tthreads\tz\tflags\tcase_universe\tring_overflow\tfamily\tring\treuse\tgeneration\n'
for d in "$ROOT" "$ROOT/union"; do
    [ -f "$d/summary.tsv" ]  || printf "$SUMMARY_HDR"  > "$d/summary.tsv"
    [ -f "$d/manifest.tsv" ] || printf "$MANIFEST_HDR" > "$d/manifest.tsv"
done

# A resume under other compile or runtime flags would mix two cost models in one timing column: refuse.
guard_meta() { # <dir> <case_universe>
    local meta="$1/campaign_meta.txt" cu="$2" prev
    prev=$(grep -oP '^campaign_z: \K.*' "$meta" 2>/dev/null | head -1)
    if [ -n "$prev" ] && [ "$prev" != "$CAMPAIGN_Z" ]; then
        echo "!!! campaign_z ${CAMPAIGN_Z} differs from the recorded launch (${prev}); refusing to resume" >&2; exit 1; fi
    prev=$(grep -oP '^party_flags: \K.*' "$meta" 2>/dev/null | head -1)
    if [ -n "$prev" ] && [ "$prev" != "$PARTY_FLAGS" ]; then
        echo "!!! party_flags '${PARTY_FLAGS}' differ from the recorded launch ('${prev}'); refusing to resume" >&2; exit 1; fi
    prev=$(grep -oP '^protocol: \K.*' "$meta" 2>/dev/null | head -1)
    if [ -n "$prev" ] && [ "$prev" != "$PROTOCOL" ]; then
        echo "!!! protocol '${PROTOCOL}' differs from the recorded launch ('${prev}'); refusing to resume" >&2; exit 1; fi
    prev=$(grep -oP '^generation: \K.*' "$meta" 2>/dev/null | head -1)
    if [ -n "$prev" ] && [ "$prev" != "$GENERATION" ]; then
        echo "!!! generation ${GENERATION} differs from the recorded launch (${prev}); refusing to resume" >&2; exit 1; fi
    prev=$(grep -oP '^case_universe: \K.*' "$meta" 2>/dev/null | head -1)
    if [ -n "$prev" ] && [ "$prev" != "$cu" ]; then
        echo "!!! case_universe '${cu}' differs from the recorded launch ('${prev}'); refusing to mix case universes in one table" >&2; exit 1; fi
    { echo "launch: $(date -Is)"; echo "commit: $COMMIT"; \
      echo "generation: ${GENERATION}"; \
      echo "campaign_z: ${CAMPAIGN_Z}"; echo "party_flags: ${PARTY_FLAGS}"; \
      echo "extra_compile_flags: ${EXTRA_COMPILE_FLAGS:-none}"; \
      echo "protocol: ${PROTOCOL}"; echo "threads: ${THREADS}"; \
      echo "ring: 64 (128 on the cells whose ring column says so)"; \
      echo "ring_escape: ${RING_ESCAPE}"; echo "argmin_c3: ${ARGMIN_C3}"; \
      echo "ring_overflow_bypass: never"; \
      echo "case_universe: ${cu}"; \
      echo "sepsis_dir: ${D_SEPSIS}"; echo "sepsis_pin: ${PIN_SEPSIS#--case-ids }"; \
      echo "driver: scripts/run_uniform_campaign.sh${FAMILY:+ --family $FAMILY}${ONLY:+ --only $ONLY}${FROM:+ --from $FROM}"; \
      echo "cells: $(cells_in_scope "$cu")"; \
      git status --short; echo; } >> "$meta"
}

cells_in_scope() { # <case_universe> -> the selected cell names of that universe
    local cu=$1 out=""
    for rec in "${SEL[@]}"; do
        IFS=$'\t' read -r fam cell u _rest <<<"$rec"
        [ "$u" = "$cu" ] || continue
        out="${out}${cell} "
    done
    printf '%s' "$out"
}

# 4  Pre-flight: the generation tag, the splits, the pins, the binary
wipe_slab() {
    rm -f "$SPDZ_DIR"/Persistence/Transactions-P*.data \
          "$SPDZ_DIR"/Player-Data/extract_slab.json 2>/dev/null || true
}

# Cleanup after a kill, which the pipeline's own Drop guard does not see: party streams, slab, memory dumps.
lane_cleanup() {
    local f
    for f in "$SPDZ_DIR"/Player-Data/Input-P*-0; do
        [ -f "$f" ] && : > "$f"
    done
    rm -f "$SPDZ_DIR"/Persistence/Transactions-P*.data \
          "$SPDZ_DIR"/Player-Data/extract_slab.json \
          "$SPDZ_DIR"/Player-Data/Memory-*-P* 2>/dev/null || true
    echo "    [cleanup] raw inputs truncated; slab, sidecar and Memory-*-P* removed"
}

# One lane, always: every cell shares one MP_SPDZ_DIR, Player-Data, Persistence and port base.
wait_for_lane() {
    local waited=0
    while ps -eo comm= 2>/dev/null | grep -q -- '-party\.x'; do
        if [ "$waited" -eq 0 ]; then
            echo "    [lane] an MP-SPDZ party is running; waiting for the lane"
        fi
        sleep 30
        waited=$((waited + 30))
        if [ "$waited" -ge "$LANE_WAIT_MAX_S" ]; then
            echo "!!! lane still busy after ${waited}s; aborting the campaign" >&2
            exit 1
        fi
    done
    [ "$waited" -eq 0 ] || echo "    [lane] free after ${waited}s"
}

ensure_split() { # <split-dir> <partition-config> <raw-xes>
    local dir=$1 cfg=$2 xes=$3
    [ -d "$dir" ] && return 0
    echo "=== regenerating split $dir from $cfg ==="
    "$PYTHON" scripts/split_log.py --xes "$xes" --partition-file "$cfg" \
        --output-dir "$dir" || { echo "!!! split $dir FAILED" >&2; return 1; }
}

require_split() { # <split-dir> <why>
    [ -d "$1" ] && return 0
    echo "!!! MISSING split $1 -- $2" >&2
    return 1
}

build_pin() { # <out-file> <expected-n> <dir:parties> ...
    local out=$1 expect=$2; shift 2
    [ -f "$out" ] && return 0
    echo "=== building series pin $out ==="
    local tmp; tmp=$(mktemp -d) || return 1
    local acc="" spec dir np i=0
    for spec in "$@"; do
        dir=${spec%%:*}; np=${spec##*:}; i=$((i + 1))
        "$PYTHON" scripts/build_full_case_ids.py "$dir" "$tmp/p$i.txt" --parties "$np" \
            || { echo "!!! pin build failed on $dir" >&2; rm -rf "$tmp"; return 1; }
        LC_ALL=C sort -o "$tmp/p$i.txt" "$tmp/p$i.txt"
        if [ -z "$acc" ]; then acc="$tmp/p$i.txt"; else
            LC_ALL=C comm -12 "$acc" "$tmp/p$i.txt" > "$tmp/acc$i.txt"
            acc="$tmp/acc$i.txt"
        fi
    done
    local n; n=$(wc -l < "$acc")
    if [ "$n" != "$expect" ]; then
        echo "!!! $out would hold $n ids, expected $expect; NOT writing it" >&2
        rm -rf "$tmp"; return 1
    fi
    cp "$acc" "$out"; rm -rf "$tmp"
    echo "    $out: $n shared cases"
}

preflight() {
    # timing.json's `_generation` comes from Rust: refuse rather than mistag every cell.
    if ! grep -q 'TIMING_GENERATION: u32 = 4' src/constants.rs; then
        echo "!!! src/constants.rs still reads $(grep -oP 'TIMING_GENERATION: u32 = \K\d+' src/constants.rs)" >&2
        echo "!!! Every timing.json this campaign writes would carry that tag, and gen-4" >&2
        echo "!!! timing columns would silently pool with the previous generation." >&2
        echo "!!! Bump TIMING_GENERATION to 4 before launching (S1b-2 owes this edit)." >&2
        exit 1
    fi
    ensure_split $D_SEPSIS      config/sepsis_2party.json       "data/raw/Sepsis Cases.xes" || exit 1
    ensure_split $D_SEPSIS_3P   config/sepsis_3party.json       "data/raw/Sepsis Cases.xes" || exit 1
    ensure_split $D_HELPDESK    config/helpdesk_2party.json     data/raw/helpdesk.xes || exit 1
    ensure_split $D_HELPDESK_3P config/helpdesk_3party.json     data/raw/helpdesk.xes || exit 1
    ensure_split $D_BPIC        config/bpic2013_2party.json     data/raw/bpic2013_incidents.xes || exit 1
    ensure_split $D_BPIC_3P     config/bpic2013_3party.json     data/raw/bpic2013_incidents.xes || exit 1
    ensure_split $D_RTFM        config/road_traffic_2party.json data/raw/road_traffic_fine_management.xes || exit 1
    ensure_split $D_RTFM_3P     config/road_traffic_3party.json data/raw/road_traffic_fine_management.xes || exit 1

    build_pin config/case_ids/sepsis_2party_full.txt   $N_SEPSIS      $D_SEPSIS:2 || exit 1
    build_pin config/case_ids/helpdesk_full.txt        $N_HELPDESK    $D_HELPDESK:2 || exit 1
    build_pin config/case_ids/bpic2013_role_full.txt   $N_BPIC        $D_BPIC:2 || exit 1
    build_pin config/case_ids/rtfm_full.txt            $N_RTFM        $D_RTFM:2 || exit 1
    build_pin config/case_ids/sepsis_3party_full.txt   $N_SEPSIS_3P   $D_SEPSIS_3P:3 || exit 1
    build_pin config/case_ids/helpdesk_3party_full.txt $N_HELPDESK_3P $D_HELPDESK_3P:3 || exit 1
    build_pin config/case_ids/bpic2013_3party_full.txt $N_BPIC_3P     $D_BPIC_3P:3 || exit 1
    build_pin config/case_ids/rtfm_rtseries_1184.txt   $N_RTFM_3P     $D_RTFM:2 $D_RTFM_3P:3 || exit 1
    "$PYTHON" - <<'PY' || { echo "!!! pre-flight FAILED (pins)"; exit 1; }
import pathlib, sys
expect = {"sepsis_2party_full": 927, "helpdesk_full": 4577, "bpic2013_role_full": 7545,
          "rtfm_full": 129985, "sepsis_3party_full": 925, "helpdesk_3party_full": 4221,
          "bpic2013_3party_full": 4507, "rtfm_rtseries_1184": 1184}
ok = True
sets = {}
for stem, n in expect.items():
    p = pathlib.Path(f"config/case_ids/{stem}.txt")
    if not p.exists():
        print(f"[preflight] MISSING {p}"); ok = False; continue
    ids = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    sets[stem] = set(ids)
    if len(ids) != n:
        print(f"[preflight] {p}: {len(ids)} ids, expected {n}"); ok = False
    elif len(set(ids)) != len(ids):
        print(f"[preflight] {p}: duplicate ids"); ok = False
    else:
        print(f"[preflight] {stem}: {len(ids)} unique ids OK")
# the party-count axis reads the two-party split on the three-party case set
for small, big in (("sepsis_3party_full", "sepsis_2party_full"), ("helpdesk_3party_full", "helpdesk_full"),
                   ("bpic2013_3party_full", "bpic2013_role_full"), ("rtfm_rtseries_1184", "rtfm_full")):
    if small in sets and big in sets and not sets[small] <= sets[big]:
        print(f"[preflight] {small} is not inside {big}"); ok = False
sys.exit(0 if ok else 1)
PY
    echo "=== pre-flight: building the release binary (kept out of every cell) ==="
    cargo build --release > "$ROOT/preflight_build.log" 2>&1 \
        || { echo "!!! cargo build --release FAILED; see $ROOT/preflight_build.log"; exit 1; }
    echo "[preflight] MP-SPDZ dir: $SPDZ_DIR ; protocol $PROTOCOL ; party_flags ${PARTY_FLAGS}"
    local wide=0 rec fam cell u ring _np _n _l _rest
    for rec in "${SEL[@]}"; do
        IFS=$'\t' read -r fam cell u _np _n _l ring _rest <<<"$rec"
        [ "$(effective_ring "$ring")" = "128" ] && wide=$((wide + 1))
    done
    if [ "$wide" -gt 0 ]; then
        echo "[preflight] ${wide} cell(s) compile at -R 128. TWO PRECONDITIONS, neither of which"
        echo "[preflight] this driver can create. (1) The Rust argmin guard is a HARD 63-bit"
        echo "[preflight] bound (parameters.rs::secure_kmeans_argmin_bits, mod.rs 'bits > 63'):"
        echo "[preflight] it does not read the ring width, so it refuses these cells at config"
        echo "[preflight] time regardless of -R. Make it ring-aware (part of C3) or set"
        echo "[preflight] ARGMIN_C3=1 once it admits them -- do NOT reach for the bypass."
        echo "[preflight] (2) The MP-SPDZ party binary must be"
        echo "[preflight] BUILT for that ring width (CONFIG.mine RING_SIZE / MOD, then rebuild"
        echo "[preflight] ${PROTOCOL}); a 64-bit binary fails these cells at run time, which the"
        echo "[preflight] driver records as an exit != 0 and carries on. Probe one cheap cell"
        echo "[preflight] (--only sepsisR128_sl_seq) before committing a long lane to them."
    fi
}

# 5  Per-cell timing scrape
read_timing() { # <run-dir> -> "<r1>\t<km>\t<seq>\t<r2>\t<global_mb>\t<gen>\t<party_flags>\t<round1_restarts>"
    "$PYTHON" - "$1" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "timing.json"
if not p.exists():
    print("-\t-\t-\t-\t-\t-\t-"); raise SystemExit
d = json.load(open(p))
# A Round 1 the restart rule discarded sits under `round1_rsecret_degenerate*` and leaves every column here.
rounds = {k: v for k, v in d.items()
          if isinstance(v, dict) and "_degenerate" not in k}
def exec_sum(pred):
    vals = [v.get("exec_s") for k, v in rounds.items()
            if pred(k) and v.get("exec_s") is not None]
    return f"{sum(vals):.1f}" if vals else "-"
r1 = exec_sum(lambda k: k.startswith("round1"))
km = exec_sum(lambda k: k.startswith("secure_kmeans"))
sq = exec_sum(lambda k: k == "secure_sequence_cluster")
r2 = exec_sum(lambda k: k.startswith("round2"))
mb = sum(v.get("global_data_sent_mb") or 0.0 for v in rounds.values())
pf = " ".join(d.get("_party_flags", [])) or "none"
rst = d.get("_round1_restarts", "-")
print(f"{r1}\t{km}\t{sq}\t{r2}\t{mb:.1f}\t{d.get('_generation', 1)}\t{pf}\t{rst}")
PY
}

# 6  Run one cell
run_one() {
    local rec=$1
    local fam cell u np expect_n expect_l ring reuse wipe trunc proto pf tmo est args
    IFS=$'\t' read -r fam cell u np expect_n expect_l ring reuse wipe trunc proto pf tmo est args <<<"$rec"

    local cdir="$ROOT"; [ "$u" = "intersection" ] || cdir="$ROOT/$u"
    local prefix=uni;   [ "$u" = "intersection" ] || prefix=uniu
    local summary="$cdir/summary.tsv" manifest="$cdir/manifest.tsv"
    local why done_rc=1
    if [ -f "$summary" ]; then
        why=$("$PYTHON" scripts/check_uniform_campaign.py "$cdir" --is-done "$cell"); done_rc=$?
    fi
    if [ "$done_rc" -eq 0 ]; then
        echo "=== $cell (done, skipping) ==="; return 0
    elif [ "$done_rc" -ne 1 ]; then
        echo "!!! ${cell}: scripts/check_uniform_campaign.py --is-done failed (rc=${done_rc}); not run" >&2
        FAILED=$((FAILED + 1)); return 0
    fi
    [ -n "${why:-}" ] && echo "=== $cell runs again: ${why} ==="

    local rring; rring=$(effective_ring "$ring")
    local zflags=""
    [ "$CAMPAIGN_Z" = "1" ] && [ "$proto" = "-" ] && zflags="-Z ${np}"
    [ "$rring" = "64" ] || zflags="${zflags:+$zflags }-R ${rring}"
    [ -z "$EXTRA_COMPILE_FLAGS" ] || zflags="${zflags:+$zflags }${EXTRA_COMPILE_FLAGS}"
    local cell_proto="$proto"; [ "$proto" = "-" ] && cell_proto="$PROTOCOL"
    local cell_pf="$pf";       [ "$pf" = "-" ] && cell_pf="$PARTY_FLAGS"
    local reuse_flag=""; [ "$reuse" = "reuse" ] && reuse_flag="--reuse-extraction"

    echo "=== [$fam] $cell (np=$np, N=$expect_n, L=$expect_l, ring=$rring, ${reuse_flag:-clean}, flags='${cell_pf:-none}', timeout=$tmo) ==="
    # The manifest row goes in before the run, so a killed campaign records what it was running.
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$cell" "-" "$COMMIT" "$cell_proto" "${cell_pf:-none}" "$THREADS" \
        "${zflags:--}" "${args}${reuse_flag:+ $reuse_flag}${univ_arg:+ $univ_arg}" \
        "$u" 0 "$fam" "$rring" "$reuse" "$GENERATION" >> "$manifest"

    [ "$wipe" = "wipe" ] && wipe_slab
    wait_for_lane

    # The clocks before the cell; its row of clock.tsv is written after it.
    local clk0; clk0=$("$PYTHON" scripts/clock_probe.py probe 2>/dev/null)
    local log="$cdir/${cell}.log" t0=$SECONDS
    if [ -f "$log" ]; then   # the log of an earlier run of this cell is kept
        mkdir -p "$cdir/superseded"
        mv "$log" "$cdir/superseded/${cell}.$(date -r "$log" +%Y%m%dT%H%M%S).log"
    fi
    # shellcheck disable=SC2086
    CLUSTERING_MPC_COMPILE_FLAGS="${zflags}" \
    CLUSTERING_MPC_PARTY_FLAGS="${cell_pf}" \
    timeout --kill-after=60s "$tmo" \
    cargo run --release -- $args $reuse_flag --evaluate-full --fast-eval \
        --run-name "${prefix}_${cell}" > "$log" 2>&1
    local rc=$? dt=$((SECONDS - t0))
    local clk="" clk_rc=0
    if [ -n "$clk0" ]; then
        clk=$("$PYTHON" scripts/clock_probe.py row "$cell" "$clk0" 2>/dev/null); clk_rc=$?
        [ -f "$cdir/clock.tsv" ] || "$PYTHON" scripts/clock_probe.py header > "$cdir/clock.tsv"
        "$PYTHON" scripts/clock_probe.py upgrade "$cdir/clock.tsv" >/dev/null
        [ -n "$clk" ] && printf '%s\n' "$clk" >> "$cdir/clock.tsv"
    fi
    [ "$rc" -eq 124 ] && echo "!!! ${cell}: TIMED OUT after ${tmo}" >&2

    # ---- enforcement -------------------------------------------------
    # "Phase 0: N common cases" is post-truncation, so it is the effective population of any cell.
    local eff
    eff=$(grep -oP 'Phase 0: \K\d+(?= common cases)' "$log" | head -1)
    if [ "$rc" -eq 0 ] && [ "$expect_n" != "-" ] && [ "${eff:-0}" != "$expect_n" ]; then
        echo "!!! ${cell}: effective population ${eff:-<none>} != expected ${expect_n}" >&2; rc=1
    fi
    local lgot=""
    if [ "$expect_l" = "auto" ]; then
        lgot=$(grep -oP '\[MPC\] Dynamic L = \K\d+' "$log" | head -1)
        if grep -q 'Using user-specified max_trace_len' "$log"; then
            echo "!!! ${cell}: expected auto-L but the run pinned --max-trace-len" >&2; rc=1
        fi
    elif [ "$expect_l" != "-" ]; then
        lgot=$(grep -oP 'Using user-specified max_trace_len = \K\d+' "$log" | head -1)
        if [ "$rc" -eq 0 ] && [ "${lgot:-0}" != "$expect_l" ]; then
            echo "!!! ${cell}: max_trace_len ${lgot:-<none>} != expected ${expect_l}" >&2; rc=1
        fi
    fi
    # Slab hygiene both ways: a measured cell that reused an extraction is not quotable.
    if [ "$reuse" != "reuse" ] && grep -q '\[MPC\] --reuse-extraction:' "$log"; then
        echo "!!! ${cell}: --reuse-extraction appears in a MEASURED cell" >&2; rc=1
    fi
    # The stable substring of src/pipeline/mpc/mod.rs's refusal.
    if grep -q 'can overflow the 64-bit ring' "$log"; then
        echo "!!! ${cell}: the argmin ring-width guard REFUSED this cell (ring=${rring})" >&2
    fi

    # ---- columns -----------------------------------------------------
    local k="" kpriv="" supp prev=""
    for a in $args; do [ "$prev" = "--k" ] && k=$a; prev=$a; done
    [[ "$k" =~ ^[0-9]+$ ]] || k=""
    [ -n "$k" ] || k=$(grep -oP 'Resolved k=\K\d+' "$log" | head -1)
    kpriv=$(grep -oP 'Budget-bounded k_priv = \K\d+(?= \(selected in-MPC)' "$log" | head -1)
    supp=$(grep -oP '\K\d+(?= suppressed-tail rows)' "$log" | head -1)

    local rundir tim r1 km sq r2 gmb gen tpf rst
    # `_20*` (the run timestamp), not `_*`, so a cell name that prefixes another does not match both.
    rundir=$(ls -dt output/runs/${prefix}_${cell}_20* 2>/dev/null | head -1)
    tim=$(read_timing "${rundir:-/nonexistent}")
    r1=$(printf '%s' "$tim" | cut -f1); km=$(printf '%s' "$tim" | cut -f2)
    sq=$(printf '%s' "$tim" | cut -f3); r2=$(printf '%s' "$tim" | cut -f4)
    gmb=$(printf '%s' "$tim" | cut -f5); gen=$(printf '%s' "$tim" | cut -f6)
    tpf=$(printf '%s' "$tim" | cut -f7); rst=$(printf '%s' "$tim" | cut -f8)
    if [ "$rc" -eq 0 ] && [ "$gen" != "-" ] && [ "$gen" != "$GENERATION" ]; then
        echo "!!! ${cell}: timing.json carries _generation=${gen}, not ${GENERATION}" >&2; rc=1
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$cell" "$rc" "$fam" "$np" "${eff:--}" "${lgot:--}" "${k:--}" "${kpriv:--}" \
        "${supp:--}" "$dt" "$r1" "$km" "$sq" "$r2" "$gmb" "$gen" "$tpf" \
        "$rring" 0 "$reuse" "$u" "${rst:--}" >> "$summary"
    # Re-record the manifest row with the resolved run dir; later rows win, superseding the placeholder.
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$cell" "${rundir:--}" "$COMMIT" "$cell_proto" "${cell_pf:-none}" "$THREADS" \
        "${zflags:--}" "${args}${reuse_flag:+ $reuse_flag}${univ_arg:+ $univ_arg}" \
        "$u" 0 "$fam" "$rring" "$reuse" "$GENERATION" >> "$manifest"
    echo "    exit=$rc N=${eff:--} L=${lgot:--} k=${k:--} k_priv=${kpriv:--} wall=${dt}s r1=${r1}s km=${km}s seq=${sq}s r2=${r2}s global=${gmb}MB gen=${gen} restarts=${rst:--}"
    if [ -n "$clk" ]; then
        local gain; gain=$(printf '%s' "$clk" | cut -f10)
        echo "    clock: monotonic ${gain} % against $(printf '%s' "$clk" | cut -f8) over $(printf '%s' "$clk" | cut -f9) s; CPU outside the campaign $(printf '%s' "$clk" | cut -f13) s"
        [ "$clk_rc" -eq 3 ] && echo "!!! ${cell}: the monotonic clock left real time by ${gain} % during this cell; its round times carry that gain" >&2
    else
        echo "!!! ${cell}: no clock record (scripts/clock_probe.py)" >&2
    fi

    # Clean the lane after a failed cell and carry on: one cell's failure must not cost the rest.
    if [ "$rc" -ne 0 ]; then
        lane_cleanup
        FAILED=$((FAILED + 1))
    fi
    # The Helpdesk slab is quadratic in N: record its size once, right after it is built.
    if [ "$reuse" = "reuse" ] && [ "$wipe" = "wipe" ] && \
       grep -q 'creating core slab' "$log" 2>/dev/null && \
       ls "$SPDZ_DIR"/Persistence/Transactions-P0.data >/dev/null 2>&1; then
        { echo "slab after builder ${cell}:"; \
          ls -l "$SPDZ_DIR"/Persistence/Transactions-P*.data; \
          df -h "$SPDZ_DIR" | tail -1; echo; } >> "$cdir/campaign_meta.txt"
    fi
    return 0
}

# 7  Run the table, top-down
guard_meta "$ROOT" intersection
[ -n "$(cells_in_scope union)" ] && guard_meta "$ROOT/union" union

preflight

FAILED=0
RAN=0
for rec in "${SEL[@]}"; do
    run_one "$rec"
    RAN=$((RAN + 1))
done

wipe_slab
echo
echo "Uniform campaign: ${RAN} cells attempted, ${FAILED} with exit != 0."
echo "Intersection summary:"; column -t "$ROOT/summary.tsv"
if [ -s "$ROOT/union/summary.tsv" ] && [ "$(wc -l < "$ROOT/union/summary.tsv")" -gt 1 ]; then
    echo; echo "Union summary:"; column -t "$ROOT/union/summary.tsv"
fi
cat <<EOF

Aggregate (union and intersection are separate tables):
  $PYTHON scripts/aggregate_campaign.py --campaign $ROOT --latex
  $PYTHON scripts/aggregate_campaign.py --campaign $ROOT/union --latex
EOF
[ "$FAILED" -eq 0 ]
