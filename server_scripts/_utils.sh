#!/bin/bash

source <(sed -E 's/^([A-Z_]+)[[:space:]]*=[[:space:]]*([0-9]+)[[:space:]]*$/:\ "${\1:=\2}"/' "$(dirname "${BASH_SOURCE[0]}")/../vllm_defaults.cfg")
