#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/home/zhaozhang/local/AGORA_work/AGORA_Data/z0}"
PIPELINE="${PIPELINE:-${ROOT}/AGORA_parallel_DM_EM_projection_pipeline.py}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT}/parallel_outputs}"
LOGDIR="${LOGDIR:-${ROOT}/batch_logs_20k}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhaozhang/anaconda3/bin/python}"

# Parameters synced with AGORA_z0_adaptive_DM_EM_analysis.ipynb.
N_LOS="${N_LOS:-20000}"
N_JOBS="${N_JOBS:-15}"
RANDOM_OBSERVERS="${RANDOM_OBSERVERS:-128}"
S_MAX_KPC="${S_MAX_KPC:-100}"
DS_KPC="${DS_KPC:-0.25}"
R_SUN_KPC="${R_SUN_KPC:-8.2}"
VELOCITY_REFERENCE_SOURCE="${VELOCITY_REFERENCE_SOURCE:-stars}"
VELOCITY_REFERENCE_RADIUS_KPC="${VELOCITY_REFERENCE_RADIUS_KPC:-30.0}"
ISM_R_KPC="${ISM_R_KPC:-20.0}"
ISM_ABS_Z_KPC="${ISM_ABS_Z_KPC:-5.0}"
HOT_TMIN_K="${HOT_TMIN_K:-1.0e6}"
PROJECTION_BOX_KPC="${PROJECTION_BOX_KPC:-40}"
PROJECTION_NPIX="${PROJECTION_NPIX:-512}"
PROJECTION_MAX_ELEMENTS="${PROJECTION_MAX_ELEMENTS:-100000000}"
PROJECTION_KERNEL_MAX_PARTICLES="${PROJECTION_KERNEL_MAX_PARTICLES:-100000000}"
PROJECTION_QUIVER_STEP="${PROJECTION_QUIVER_STEP:-12}"
# Projection LOS half-thickness in kpc. Empty string restores full-column projection.
PROJECTION_LOS_HALF_THICKNESS_KPC="${PROJECTION_LOS_HALF_THICKNESS_KPC-}"
UNIT_BASE="${UNIT_BASE:-auto}"
INTEGRATION_BACKEND="${INTEGRATION_BACKEND:-auto}"
PARTICLE_INTERPOLATION="${PARTICLE_INTERPOLATION:-auto}"
CENTER_MODE="${CENTER_MODE:-stellar_com}"
DISK_NORMAL_SOURCE="${DISK_NORMAL_SOURCE:-stars}"
CHUNK_LOS="${CHUNK_LOS:-64}"

# Notebook default: IONIZATION_MODE=None means G4Cal uses temperature_weighted,
# other datasets use fully_ionized. Set IONIZATION_MODE=auto to use pipeline auto.
IONIZATION_MODE="${IONIZATION_MODE:-auto}"

mkdir -p "${OUTPUT_BASE}" "${LOGDIR}"
cd "${ROOT}" || exit 1

ALL_CODES=(ARTI Enzo AREPO GADGET3 GEAR CHANGA G4Cal_Pablo)
if [[ "$#" -gt 0 ]]; then
  REQUESTED_CODES=("$@")
else
  REQUESTED_CODES=("${ALL_CODES[@]}")
fi

canonical_code() {
  case "$1" in
    ARTI|ART-I|ART|arti|art-i|art) echo "ARTI" ;;
    Enzo|ENZO|enzo) echo "Enzo" ;;
    AREPO|arepo) echo "AREPO" ;;
    GADGET3|GADGET-3|gadget3|gadget-3) echo "GADGET3" ;;
    GEAR|gear) echo "GEAR" ;;
    CHANGA|changa) echo "CHANGA" ;;
    G4Cal_Pablo|G4CAL_PABLO|G4Cal|G4CAL|g4cal_pablo|g4cal) echo "G4Cal_Pablo" ;;
    *) echo "$1" ;;
  esac
}

snapshot_for() {
  case "$1" in
    ARTI) echo "${ROOT}/ARTI/10MpcBox_csf512_04078.d" ;;
    Enzo) echo "${ROOT}/Enzo/RD0347/RD0347" ;;
    AREPO) echo "${ROOT}/AREPO/snap_336.hdf5" ;;
    GADGET3) echo "${ROOT}/GADGET3/snapshot_304/snapshot_304.0.hdf5" ;;
    GEAR) echo "${ROOT}/GEAR/snapshot_0845.hdf5" ;;
    CHANGA) echo "${ROOT}/CHANGA/ncal-IV.003524" ;;
    G4Cal_Pablo) echo "${ROOT}/G4Cal_Pablo/snapshot_034.hdf5" ;;
    *) return 1 ;;
  esac
}

pipeline_code_for() {
  case "$1" in
    ARTI) echo "ARTI" ;;
    Enzo) echo "ENZO" ;;
    AREPO) echo "AREPO" ;;
    GADGET3) echo "GADGET-3" ;;
    GEAR) echo "GEAR" ;;
    CHANGA) echo "CHANGA" ;;
    G4Cal_Pablo) echo "G4Cal_Pablo" ;;
    *) return 1 ;;
  esac
}

ion_mode_for() {
  local label="$1"
  if [[ -n "${IONIZATION_MODE}" ]]; then
    echo "${IONIZATION_MODE}"
  elif [[ "${label}" == "G4Cal_Pablo" ]]; then
    echo "temperature_weighted"
  else
    echo "fully_ionized"
  fi
}

COMMON_ARGS=(
  --integration-backend "${INTEGRATION_BACKEND}"
  --unit-base "${UNIT_BASE}"
  --n-los "${N_LOS}"
  --n-jobs "${N_JOBS}"
  --chunk-los "${CHUNK_LOS}"
  --particle-interpolation "${PARTICLE_INTERPOLATION}"
  --random-observers "${RANDOM_OBSERVERS}"
  --s-max-kpc "${S_MAX_KPC}"
  --ds-kpc "${DS_KPC}"
  --R-sun-kpc "${R_SUN_KPC}"
  --center-mode "${CENTER_MODE}"
  --disk-normal-source "${DISK_NORMAL_SOURCE}"
  --velocity-reference-source "${VELOCITY_REFERENCE_SOURCE}"
  --velocity-reference-radius-kpc "${VELOCITY_REFERENCE_RADIUS_KPC}"
  --ism-R-kpc "${ISM_R_KPC}"
  --ism-abs-z-kpc "${ISM_ABS_Z_KPC}"
  --hot-Tmin-K "${HOT_TMIN_K}"
  --make-projections
  --projection-box-kpc "${PROJECTION_BOX_KPC}"
  --projection-npix "${PROJECTION_NPIX}"
  --projection-max-elements "${PROJECTION_MAX_ELEMENTS}"
  --projection-kernel-max-particles "${PROJECTION_KERNEL_MAX_PARTICLES}"
  --projection-quiver-step "${PROJECTION_QUIVER_STEP}"
  --make-mollweide
  --make-hot-em-diagnostics
)

if [[ -n "${PROJECTION_LOS_HALF_THICKNESS_KPC}" ]]; then
  COMMON_ARGS+=(--projection-los-half-thickness-kpc "${PROJECTION_LOS_HALF_THICKNESS_KPC}")
fi

run_one() {
  local requested="$1"
  local label
  label="$(canonical_code "${requested}")"
  local snapshot code ion_mode outdir logfile
  snapshot="$(snapshot_for "${label}")" || { echo "Unknown code: ${requested}"; return 2; }
  code="$(pipeline_code_for "${label}")" || { echo "Unknown code: ${requested}"; return 2; }
  ion_mode="$(ion_mode_for "${label}")"
  outdir="${OUTPUT_BASE}/${label}"
  logfile="${LOGDIR}/${label}.log"

  local extra_args=()
  if [[ "${label}" == "CHANGA" ]]; then
    if [[ -z "${CHANGA_TIPSY_LENGTH_UNIT_KPC:-}" || -z "${CHANGA_TIPSY_MASS_UNIT_MSUN:-}" ]]; then
      echo "===== $(date): skipping CHANGA: set CHANGA_TIPSY_LENGTH_UNIT_KPC and CHANGA_TIPSY_MASS_UNIT_MSUN first =====" | tee -a "${LOGDIR}/batch_status.log"
      return 0
    fi
    extra_args+=(--tipsy-length-unit-kpc "${CHANGA_TIPSY_LENGTH_UNIT_KPC}")
    extra_args+=(--tipsy-mass-unit-msun "${CHANGA_TIPSY_MASS_UNIT_MSUN}")
    if [[ -n "${CHANGA_TIPSY_TIME_UNIT_S:-}" ]]; then
      extra_args+=(--tipsy-time-unit-s "${CHANGA_TIPSY_TIME_UNIT_S}")
    fi
  fi

  mkdir -p "${outdir}"
  {
    echo "===== $(date): starting ${label} ====="
    echo "code=${code}"
    echo "snapshot=${snapshot}"
    echo "outdir=${outdir}"
    echo "ionization_mode=${ion_mode}"
    echo "N_LOS=${N_LOS} N_JOBS=${N_JOBS} S_MAX_KPC=${S_MAX_KPC} DS_KPC=${DS_KPC}"
  } | tee -a "${LOGDIR}/batch_status.log"

  "${PYTHON_BIN}" "${PIPELINE}" \
    --snapshot "${snapshot}" \
    --code "${code}" \
    --outdir "${outdir}" \
    --ionization-mode "${ion_mode}" \
    "${COMMON_ARGS[@]}" \
    "${extra_args[@]}" >"${logfile}" 2>&1

  local status=$?
  if [[ ${status} -eq 0 ]]; then
    echo "===== $(date): finished ${label} OK =====" | tee -a "${LOGDIR}/batch_status.log"
  else
    echo "===== $(date): ${label} FAILED with status ${status}; see ${logfile} =====" | tee -a "${LOGDIR}/batch_status.log"
  fi
  return ${status}
}

echo "Batch started at $(date)" | tee -a "${LOGDIR}/batch_status.log"
echo "ROOT=${ROOT}" | tee -a "${LOGDIR}/batch_status.log"
echo "PYTHON_BIN=${PYTHON_BIN}" | tee -a "${LOGDIR}/batch_status.log"
echo "Requested codes: ${REQUESTED_CODES[*]}" | tee -a "${LOGDIR}/batch_status.log"

failed=0
for requested in "${REQUESTED_CODES[@]}"; do
  run_one "${requested}" || failed=1
done

echo "Batch finished at $(date), failed=${failed}" | tee -a "${LOGDIR}/batch_status.log"
exit ${failed}
