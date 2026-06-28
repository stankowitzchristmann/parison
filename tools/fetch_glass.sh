#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-data}"
mkdir -p "${DEST}"

cat <<'NOTE'
Physiological corpora are credentialed or license-gated and cannot be pulled
non-interactively. Follow glass-sources.txt:

  charisdb 1.0.0        wget -r -N -c -np https://physionet.org/files/charisdb/1.0.0/
  mimic4wdb 0.1.0       wget -r -N -c -np https://physionet.org/files/mimic4wdb/0.1.0/
  eicu-crd 2.0          requires a signed DUA; use the PhysioNet download token
  mimiciv 3.1           requires a signed DUA; use the PhysioNet download token

Place each corpus under the destination passed as the first argument.
NOTE
