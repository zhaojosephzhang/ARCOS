#!/usr/bin/env bash
set -u
ROOT=/home/zhaozhang/local/AGORA_work/AGORA_Data/z0
PY=/home/zhaozhang/anaconda3/bin/python
PIPE=${ROOT}/AGORA_parallel_DM_EM_projection_pipeline.py
mkdir -p ${ROOT}/single_logs

run_one() {
  label="$1"
  snapshot="$2"
  code="$3"
  outdir="${ROOT}/parallel_outputs/${label}"
  log="${ROOT}/single_logs/${label}_20k_pipeline_gasdisk.log"
  mkdir -p "$outdir"
  echo "===== $(date): starting ${label} with exact gasdisk parameters =====" > "$log"
  extra=()
  if [[ "$label" == "CHANGA" ]]; then
    extra+=(--tipsy-length-unit-kpc 1 --tipsy-mass-unit-msun 1)
  fi
  "$PY" "$PIPE" \
    --snapshot "$snapshot" \
    --code "$code" \
    --outdir "$outdir" \
    --ionization-mode auto \
    --n-los 20000 \
    --n-jobs 30 \
    --random-observers 128 \
    --s-max-kpc 100 \
    --ds-kpc 0.25 \
    --R-sun-kpc 8.2 \
    --velocity-reference-source gas \
    --velocity-reference-radius-kpc 30 \
    --ism-R-kpc 20 \
    --ism-abs-z-kpc 5 \
    --hot-Tmin-K 1e6 \
    --make-projections \
    --projection-box-kpc 40 \
    --projection-npix 512 \
    --projection-max-elements 100000000 \
    --projection-kernel-max-particles 500000 \
    --projection-quiver-step 12 \
    --projection-quiver-scale 2200 \
    --projection-quiver-width 0.0022 \
    --projection-quiver-alpha 0.75 \
    --projection-los-half-thickness-kpc 5 \
    --make-mollweide \
    --make-hot-em-diagnostics \
    "${extra[@]}" >> "$log" 2>&1
  status=$?
  echo "===== $(date): ${label} finished status=${status} =====" >> "$log"
  return $status
}

run_one CHANGA "${ROOT}/CHANGA/ncal-IV.003524" CHANGA
run_one AREPO "${ROOT}/AREPO/snap_336.hdf5" AREPO
