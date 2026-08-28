#!/usr/bin/env bash
#
# generate_montage.sh -- turn rollout camera videos into one reviewable movie.
#
# For every episode found under a rollout folder the six camera streams are
# stacked into a 3x2 grid:
#
#     head_left  | head_center   | head_right
#     wrist_left | wrist_depth   | wrist_right
#
# The wrist depth stream is stored as a 16-bit metric depth packed into the R/G
# channels, so it is decoded back to a greyscale depth map for the grid rather
# than shown as the green banding the raw file plays back as.
#
# The per-episode grids are then montaged together across every subfolder,
# either one after another (default) or side by side as a grid of grids.
#
# Usage:
#   ./generate_montage.sh rollouts/pick
#   ./generate_montage.sh -m tile -o /tmp/all.mp4 rollouts/molmobot/pick
#
# The rollout folder is taken as given if it exists, otherwise it is looked up
# under $DATA_ROOT (default: <repo>/data/stretch_pick).
#
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/stretch_pick}"

# Grid layout, row major. Keep CAMERAS and CELL_LABELS in the same order.
CAMERAS=(head_camera_left head_camera head_camera_right
         wrist_camera_left wrist_camera_stereo_depth wrist_camera_right)
CELL_LABELS=("head left" "head center" "head right"
             "wrist left" "wrist depth" "wrist right")

CELL_W=${CELL_W:-320}       # per-camera cell size; sources are letterboxed into it
CELL_H=${CELL_H:-320}
BANNER_H=${BANNER_H:-28}    # strip above each grid holding the episode name
MAX_WALL_W=${MAX_WALL_W:-3840}  # -m tile: shrink tiles only past this width
FPS=${FPS:-}                # empty -> probed from the first source video
CRF=${CRF:-23}
PRESET=${PRESET:-veryfast}
JOBS=${JOBS:-}              # empty -> half the cores

MODE=concat
OUT=
KEEP=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] <rollout-folder>

  <rollout-folder>  e.g. rollouts/pick, absolute or relative to
                    \$DATA_ROOT ($DATA_ROOT)

Options:
  -o FILE   output video (default: <rollout-folder>/montage.mp4)
  -m MODE   concat (episodes play one after another, default)
            tile   (all episode grids shown at once)
  -W PX     camera cell width  (default $CELL_W)
  -H PX     camera cell height (default $CELL_H)
  -r FPS    output frame rate (default: frame rate of the first source video)
  -j N      parallel ffmpeg jobs (default: half the cores)
  -k        keep the intermediate per-episode grid videos (a later run then
            reuses them instead of re-encoding)
  -h        this help
EOF
}

while getopts ":o:m:W:H:r:j:kh" opt; do
    case "$opt" in
        o) OUT=$OPTARG ;;
        m) MODE=$OPTARG ;;
        W) CELL_W=$OPTARG ;;
        H) CELL_H=$OPTARG ;;
        r) FPS=$OPTARG ;;
        j) JOBS=$OPTARG ;;
        k) KEEP=1 ;;
        h) usage; exit 0 ;;
        :) echo "error: -$OPTARG needs an argument" >&2; exit 2 ;;
        \?) echo "error: unknown option -$OPTARG" >&2; usage >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

[[ $# -eq 1 ]] || { usage >&2; exit 2; }
[[ $MODE == concat || $MODE == tile ]] || { echo "error: -m must be concat or tile" >&2; exit 2; }
command -v ffmpeg >/dev/null || { echo "error: ffmpeg not found" >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "error: ffprobe not found" >&2; exit 1; }

# ---------------------------------------------------------------- input paths
ARG=$1
if [[ -d $ARG ]]; then
    ROOT=$(cd "$ARG" && pwd)
elif [[ -d "$DATA_ROOT/$ARG" ]]; then
    ROOT=$(cd "$DATA_ROOT/$ARG" && pwd)
else
    echo "error: no such rollout folder: $ARG (also tried $DATA_ROOT/$ARG)" >&2
    exit 1
fi

OUT=${OUT:-$ROOT/montage.mp4}
mkdir -p "$(dirname "$OUT")"
WORK="${OUT%.*}_grids"
mkdir -p "$WORK"

if [[ -z $JOBS ]]; then
    JOBS=$(( $(nproc 2>/dev/null || echo 2) / 2 ))
    (( JOBS > 0 )) || JOBS=1
fi

# ------------------------------------------------------------------ text draw
FONT=${FONT:-}
if [[ -z $FONT ]]; then
    for f in /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
             /usr/share/fonts/TTF/DejaVuSans.ttf; do
        [[ -f $f ]] && { FONT=$f; break; }
    done
    [[ -z $FONT ]] && FONT=$(fc-match -f "%{file}" "DejaVu Sans" 2>/dev/null || true)
fi
[[ -f ${FONT:-} ]] || FONT=

# drawtext_filter <text> <x-expr> <y-expr> <fontsize>
# Emits an empty string when no usable font was found, so the filter graph
# stays valid on machines without fontconfig.
drawtext_filter() {
    local text=$1 x=$2 y=$3 size=$4
    [[ -n $FONT ]] || { printf ''; return; }
    text=${text//:/-}          # ':' and '\'' would break the filter syntax
    text=${text//\'/}
    text=${text//\\/}
    printf "drawtext=fontfile='%s':text='%s':fontsize=%s:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=4:x=%s:y=%s" \
        "$FONT" "$text" "$size" "$x" "$y"
}

# ------------------------------------------------------------ episode lookup
# find_camera <dir> <episode-id> <camera> -> path on stdout, empty if missing
find_camera() {
    local dir=$1 ep=$2 cam=$3 m
    local matches=("$dir/episode_${ep}_${cam}_batch_"*.mp4 "$dir/episode_${ep}_${cam}.mp4")
    for m in "${matches[@]}"; do
        [[ -f $m ]] && { printf '%s' "$m"; return 0; }
    done
    return 0    # a missing camera is reported as empty output, not as failure
}

# Subfolders of the rollout dir, plus the dir itself when videos sit directly
# in it (single-scene rollouts).
SEARCH_DIRS=()
while IFS= read -r d; do SEARCH_DIRS+=("$d"); done < <(
    find "$ROOT" -mindepth 1 -maxdepth 1 -type d ! -path "$WORK" | sort)
if compgen -G "$ROOT/episode_*.mp4" >/dev/null; then
    SEARCH_DIRS+=("$ROOT")
fi
[[ ${#SEARCH_DIRS[@]} -gt 0 ]] || { echo "error: no subfolders or videos under $ROOT" >&2; exit 1; }

# Collect (dir, episode) pairs and the first video seen, for fps probing.
EP_DIRS=()
EP_IDS=()
FIRST_VIDEO=
for d in "${SEARCH_DIRS[@]}"; do
    while IFS= read -r ep; do
        [[ -n $ep ]] || continue
        EP_DIRS+=("$d")
        EP_IDS+=("$ep")
        [[ -n $FIRST_VIDEO ]] || FIRST_VIDEO=$(find_camera "$d" "$ep" "${CAMERAS[0]}")
    done < <(find "$d" -maxdepth 1 -name 'episode_*.mp4' -printf '%f\n' |
             sed -nE 's/^episode_([0-9]+)_.*\.mp4$/\1/p' | sort -u)
done

[[ ${#EP_IDS[@]} -gt 0 ]] || { echo "error: no episode_*.mp4 files found under $ROOT" >&2; exit 1; }

if [[ -z $FPS ]]; then
    if [[ -n $FIRST_VIDEO ]]; then
        FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate \
                      -of csv=p=0 "$FIRST_VIDEO")
    fi
    FPS=${FPS:-15}
fi

echo "rollout : $ROOT"
echo "episodes: ${#EP_IDS[@]} across ${#SEARCH_DIRS[@]} folder(s)"
echo "grid    : 3x2 of ${CELL_W}x${CELL_H} @ ${FPS} fps, mode=$MODE, jobs=$JOBS"
echo

# ------------------------------------------------------------- per-episode grid
GRID_W=$((CELL_W * 3))
GRID_H=$((CELL_H * 2 + BANNER_H))

# build_grid <out.mp4> <label> <six input videos...>
build_grid() {
    local out=$1 label=$2
    shift 2
    local files=("$@")
    local args=() fc="" i cell

    for i in "${!files[@]}"; do args+=(-i "${files[$i]}"); done

    for i in "${!files[@]}"; do
        cell="[$i:v]fps=${FPS}"
        # Depth streams are not pictures: the saver packs a 16-bit metric depth
        # into R (high byte) and G (low byte) with B=0, which plays back as
        # green contour bands. The high byte on its own is the depth map at
        # 8-bit resolution -- ~2 mm over the encoder's 5-55 cm range -- so pull
        # it out and show that as greyscale. Black is either 5 cm or invalid.
        [[ ${CAMERAS[$i]} == *depth* ]] && cell+=",format=gbrp,extractplanes=r,format=gray"
        cell+=",scale=${CELL_W}:${CELL_H}:force_original_aspect_ratio=decrease"
        cell+=",pad=${CELL_W}:${CELL_H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format=yuv420p"
        local t
        t=$(drawtext_filter "${CELL_LABELS[$i]}" 8 "h-th-8" 16)
        [[ -n $t ]] && cell+=",$t"
        fc+="${cell}[c$i];"
    done

    fc+="[c0][c1][c2]hstack=inputs=3[top];"
    fc+="[c3][c4][c5]hstack=inputs=3[bottom];"
    fc+="[top][bottom]vstack=inputs=2[grid];"
    fc+="[grid]pad=iw:ih+${BANNER_H}:0:${BANNER_H}:color=black"
    local bt
    bt=$(drawtext_filter "$label" 8 "($BANNER_H-th)/2" 18)
    [[ -n $bt ]] && fc+=",$bt"
    fc+="[v]"

    ffmpeg -hide_banner -loglevel error -y "${args[@]}" \
        -filter_complex "$fc" -map '[v]' \
        -c:v libx264 -preset "$PRESET" -crf "$CRF" -pix_fmt yuv420p \
        -movflags +faststart "$out"
}

GRIDS=()
GRID_LABELS=()
pending=0
skipped=0
reused=0

for idx in "${!EP_IDS[@]}"; do
    dir=${EP_DIRS[$idx]}
    ep=${EP_IDS[$idx]}
    scene=$(basename "$dir")
    [[ $dir == "$ROOT" ]] && scene=$(basename "$ROOT")
    label="$scene  episode $ep"

    inputs=()
    missing=()
    for cam in "${CAMERAS[@]}"; do
        f=$(find_camera "$dir" "$ep" "$cam")
        if [[ -z $f ]]; then missing+=("$cam"); else inputs+=("$f"); fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "skip  $label -- missing: ${missing[*]}" >&2
        skipped=$((skipped + 1))
        continue
    fi

    grid="$WORK/${scene}_episode_${ep}.mp4"
    GRIDS+=("$grid")
    GRID_LABELS+=("$label")

    if [[ -s $grid ]]; then
        echo "have  $label -- reusing $(basename "$grid")"
        reused=$((reused + 1))
        continue
    fi

    echo "grid  $label"
    build_grid "$grid" "$label" "${inputs[@]}" &
    pending=$((pending + 1))
    if (( pending >= JOBS )); then
        wait -n
        pending=$((pending - 1))
    fi
done
wait

[[ ${#GRIDS[@]} -gt 0 ]] || { echo "error: no episode had all 6 cameras" >&2; exit 1; }
for g in "${GRIDS[@]}"; do
    [[ -s $g ]] || { echo "error: grid encode failed for $g" >&2; exit 1; }
done

# ------------------------------------------------------------------- montage
echo
echo "montaging ${#GRIDS[@]} episode grid(s) -> $OUT"

if [[ ${#GRIDS[@]} -eq 1 ]]; then
    cp -f "${GRIDS[0]}" "$OUT"

elif [[ $MODE == concat ]]; then
    list="$WORK/concat.txt"
    : >"$list"
    for g in "${GRIDS[@]}"; do
        printf "file '%s'\n" "$g" >>"$list"
    done
    ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i "$list" \
        -c copy -movflags +faststart "$OUT"

else
    n=${#GRIDS[@]}
    cols=1
    while (( cols * cols < n )); do cols=$((cols + 1)); done
    rows=$(( (n + cols - 1) / cols ))

    # Tiles keep their full resolution; the wall is only shrunk if it would
    # come out wider than MAX_WALL_W.
    tw=$GRID_W
    th=$GRID_H
    if (( cols * tw > MAX_WALL_W )); then
        tw=$(( MAX_WALL_W / cols ))
        th=$(( tw * GRID_H / GRID_W ))
    fi
    tw=$(( tw - tw % 2 ))
    th=$(( th - th % 2 ))
    (( tw >= 2 && th >= 2 )) || { echo "error: too many episodes to tile at this cell size" >&2; exit 1; }
    echo "wall    : ${cols}x${rows} tiles of ${tw}x${th} -> $((cols * tw))x$((rows * th))"

    # Episodes have different lengths; freeze the last frame of the short ones
    # so every tile stays visible for the full run.
    maxdur=0
    for g in "${GRIDS[@]}"; do
        d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$g")
        awk -v a="$d" -v b="$maxdur" 'BEGIN{exit !(a>b)}' && maxdur=$d
    done

    args=()
    fc=""
    for i in "${!GRIDS[@]}"; do args+=(-i "${GRIDS[$i]}"); done
    slots=$((cols * rows))
    for (( i = n; i < slots; i++ )); do
        args+=(-f lavfi -t "$maxdur" -i "color=c=black:s=${tw}x${th}:r=${FPS}")
    done

    layout=""
    for (( i = 0; i < slots; i++ )); do
        if (( i < n )); then
            fc+="[$i:v]scale=${tw}:${th}:force_original_aspect_ratio=decrease"
            fc+=",pad=${tw}:${th}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
            fc+=",tpad=stop_mode=clone:stop_duration=${maxdur}[t$i];"
        else
            fc+="[$i:v]setsar=1[t$i];"
        fi
        layout+="${layout:+|}$(( (i % cols) * tw ))_$(( (i / cols) * th ))"
    done
    fc+=""
    for (( i = 0; i < slots; i++ )); do fc+="[t$i]"; done
    fc+="xstack=inputs=${slots}:layout=${layout}:fill=black[v]"

    ffmpeg -hide_banner -loglevel error -y "${args[@]}" \
        -filter_complex "$fc" -map '[v]' -t "$maxdur" \
        -c:v libx264 -preset "$PRESET" -crf "$CRF" -pix_fmt yuv420p \
        -movflags +faststart "$OUT"
fi

if (( KEEP )); then
    echo "per-episode grids kept in $WORK"
else
    rm -rf "$WORK"
fi

echo "done: $OUT"
(( skipped == 0 )) || echo "note: skipped $skipped episode(s) with missing cameras" >&2
(( reused == 0 )) || echo "note: reused $reused existing episode grid(s) from $WORK" >&2
