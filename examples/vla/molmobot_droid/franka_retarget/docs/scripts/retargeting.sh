#!/usr/bin/env bash
# docs/retargeting.png, rendered from docs/retargeting.mmd with mermaid-cli (needs Node and npm;
# the first run downloads mermaid-cli and a Chromium for it). Runs from any directory.
#
#     examples/vla/molmobot_droid/franka_retarget/docs/scripts/retargeting.sh
#
# If you change docs/retargeting.mmd, copy the change into the chart in franka_retarget/README.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
npx -p @mermaid-js/mermaid-cli mmdc -i docs/retargeting.mmd -o docs/retargeting.png -b white --size 3200 -s 2
