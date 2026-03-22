#!/bin/bash
set -eu -o nounset -o pipefail
exec lint-roller check -vv
