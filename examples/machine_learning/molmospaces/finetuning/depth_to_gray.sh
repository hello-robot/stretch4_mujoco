#!/usr/bin/env bash
#
# RG-encoded depth videos -> 8-bit grayscale-in-RGB videos a pretrained ViT can read.
#
# MolmoSpaces writes `wrist_camera_stereo_depth` with molmo_spaces/utils/depth_utils.py's
# `encode_depth_to_rgb`: a 16-bit value split across R (high byte) and G (low byte),
# linearly mapping [DEPTH_MIN, DEPTH_MAX] = [0.05m, 0.55m] onto [1, 65535], with
# RGB(0,0,0) reserved for "outside that range". Fed to the vision tower as-is, that is
# two channels of byte arithmetic and a black third channel -- nothing the backbone's
# pretraining prepared it for.
#
# So: keep the R channel, copy it into G and B, and write a new file. R is *already* the
# normalized 0-255 depth -- R = d16 >> 8 = 255.99 * (depth - 0.05) / 0.50 -- so no
# arithmetic is needed, and the only information lost is the low byte, which CRF-23 H.264
# has already reduced to noise (255 distinct values per frame against R's 129).
#
# The fixed encoding range is deliberately kept rather than stretching each frame to its
# own min/max: per-frame normalization makes the same physical geometry render differently
# from frame to frame and discards the metric scale, which is the only reason to feed a
# policy depth at all.
#
# Nothing is overwritten. Every source video is opened read-only, output goes to a new
# `<camera>` name beside it, and an existing output is skipped unless FORCE=1.
#
#     bash examples/machine_learning/molmospaces/finetuning/depth_to_gray.sh \
#         data/stretch_potato/rollouts/potato
#
# Then train on it by name -- finetune.py passes an unrecognised --cameras token straight
# through, and ensure_sensor_data_paths writes the HDF5 entry from the filename:
#
#     python -m examples.machine_learning.molmospaces.finetuning.finetune \
#         --rollouts data/stretch_potato/rollouts/potato \
#         --cameras right,head,wrist_right,wrist_camera_stereo_gray
#
set -euo pipefail

ROLLOUTS="${1:-data/stretch_potato/rollouts/potato}"

# The camera to read and the camera to write. Both are used to build filenames in
# MolmoSpaces' `episode_%08d_<camera><batch suffix>.mp4` convention, which is also how
# hdf5_layout.video_filename() looks a video up -- so DST_CAMERA is the name to pass to
# `--cameras`, and changing it here means changing it there.
SRC_CAMERA="${SRC_CAMERA:-wrist_camera_stereo_depth}"
DST_CAMERA="${DST_CAMERA:-wrist_camera_stereo_gray}"

# Lossless by default. The R channel this reads has already been through one lossy pass,
# and re-quantizing what survived costs depth accuracy for disk space. CRF=18 is roughly
# half the size if that trade is worth making.
CRF="${CRF:-0}"

# libx264rgb, matching depth_utils.DEPTH_VIDEO_CODEC. RGB in, RGB out, no YUV round trip
# -- which matters more than usual here: a yuv420p intermediate would put this data
# through a limited-range (16-235) conversion whose clipping and 1.16x stretch decode as
# wrong distances, silently.
CODEC="${CODEC:-libx264rgb}"
PIXFMT="${PIXFMT:-gbrp}"

JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
FORCE="${FORCE:-0}"
VERIFY="${VERIFY:-on}"

command -v ffmpeg >/dev/null || { echo "ffmpeg not found on PATH." >&2; exit 1; }
[ -d "$ROLLOUTS" ] || { echo "No such rollout directory: $ROLLOUTS" >&2; exit 1; }

# --------------------------------------------------------------------------------------
# The conversion itself.
#
# format=gbrp,shuffleplanes=2:2:2 is a plane copy and nothing else: gbrp orders the planes
# G,B,R, so plane 2 is red, and every output plane is sourced from it. No pixel arithmetic
# and no colorspace conversion, which is what keeps the byte values exactly as encoded --
# `extractplanes=r` followed by a gray->rgb conversion would hand swscale a range decision
# to make, and it makes it differently depending on how the stream is tagged.
#
# -fps_mode passthrough keeps the frame count identical to the source. validate_trajectories.py
# compares video length against trajectory length and marks the trajectory invalid when they
# disagree, so a dropped or duplicated frame here quietly removes episodes from training.
#
# Written to `.partial` and moved into place, so an interrupted run leaves nothing that
# looks like a finished video. `-f mp4` is not optional with that name: ffmpeg picks the
# muxer from the output extension, and `.partial` is not one it knows.
# --------------------------------------------------------------------------------------
convert_one() {
    local src="$1"
    local dir="${src%/*}"
    local base="${src##*/}"
    local head="${base%%_"$SRC_CAMERA"*}"   # episode_00000000
    local rest="${base#*_"$SRC_CAMERA"}"    # _batch_1_of_1.mp4, or .mp4
    local dst="$dir/${head}_${DST_CAMERA}${rest}"

    if [ -e "$dst" ] && [ "$FORCE" != 1 ]; then
        echo "skip $dst"
        return 0
    fi

    if ! ffmpeg -nostdin -loglevel error -y -i "$src" \
        -vf "format=gbrp,shuffleplanes=2:2:2" \
        -c:v "$CODEC" -pix_fmt "$PIXFMT" -crf "$CRF" \
        -fps_mode passthrough -an \
        -f mp4 "$dst.partial"
    then
        rm -f "$dst.partial"
        echo "FAILED $src" >&2
        return 1
    fi
    mv "$dst.partial" "$dst"
    echo "wrote $dst"
}

export -f convert_one
export SRC_CAMERA DST_CAMERA CRF CODEC PIXFMT FORCE

mapfile -d '' SOURCES < <(
    find "$ROLLOUTS" -follow -type f -name "episode_*_${SRC_CAMERA}*.mp4" -print0 | sort -z
)

if [ "${#SOURCES[@]}" -eq 0 ]; then
    echo "No episode_*_${SRC_CAMERA}*.mp4 under $ROLLOUTS." >&2
    echo "Depth recording is per-camera (record_depth=True in stretch/config.py); a run" >&2
    echo "generated without it has no depth videos to convert." >&2
    exit 1
fi

echo "Converting ${#SOURCES[@]} videos: $SRC_CAMERA -> $DST_CAMERA (${JOBS} at a time)"

# `set -e` does not cross into the shells xargs starts, so it is set again there -- without
# it a failed ffmpeg is followed by a failed `mv` and a "wrote" line for a file that does
# not exist. xargs then exits 123 if any conversion failed, which is caught here rather
# than by `set -e` so the failure is reported rather than just ending the script.
status=0
printf '%s\0' "${SOURCES[@]}" \
    | xargs -0 -P "$JOBS" -I{} bash -c 'set -euo pipefail; convert_one "$@"' _ {} \
    || status=$?

if [ "$status" != 0 ]; then
    echo >&2
    echo "Some conversions failed (xargs exit $status); see the FAILED lines above." >&2
    echo "Nothing was overwritten and no partial files were left behind. Re-running skips" >&2
    echo "what already converted, so fix the cause and run it again." >&2
    exit 1
fi

# --------------------------------------------------------------------------------------
# Verify one pair rather than trusting the filter graph.
#
# PSNR against the same plane-copy applied to the source: `average:inf` means the written
# file decodes back to exactly the source's R channel in all three channels. Anything
# finite at CRF=0 means a conversion crept in somewhere and the numbers a policy would
# read are not the distances that were recorded.
# --------------------------------------------------------------------------------------
if [ "$VERIFY" = on ]; then
    src="${SOURCES[0]}"
    dir="${src%/*}"; base="${src##*/}"
    dst="$dir/${base%%_"$SRC_CAMERA"*}_${DST_CAMERA}${base#*_"$SRC_CAMERA"}"

    echo
    echo "Verifying $dst"
    frames_src=$(ffprobe -v error -count_frames -select_streams v:0 \
        -show_entries stream=nb_read_frames -of csv=p=0 "$src")
    frames_dst=$(ffprobe -v error -count_frames -select_streams v:0 \
        -show_entries stream=nb_read_frames -of csv=p=0 "$dst")
    echo "  frames: source=$frames_src output=$frames_dst"
    [ "$frames_src" = "$frames_dst" ] || echo "  WARNING: frame counts differ -- validate_trajectories.py will drop these episodes."

    ffmpeg -nostdin -hide_banner -i "$dst" -i "$src" \
        -lavfi "[1:v]format=gbrp,shuffleplanes=2:2:2[ref];[0:v][ref]psnr" \
        -f null - 2>&1 | grep -o 'average:[^ ]*' | sed 's/^/  psnr /' || true
fi

echo
echo "Done. Train on it with:"
echo
echo "  python -m examples.machine_learning.molmospaces.finetuning.finetune \\"
echo "      --rollouts $ROLLOUTS \\"
echo "      --cameras right,head,wrist_right,$DST_CAMERA"
echo
echo "Four cameras, not three: MolmoBot's preprocessor requires an even image count and"
echo "pairs them as (exo[i], ego[i]), so exo views come first. It also caps max_images at"
echo "2, which the generated script has to override -- see the note printed by finetune.py."
