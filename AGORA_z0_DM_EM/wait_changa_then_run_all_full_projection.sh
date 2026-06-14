#!/usr/bin/env bash
set -u
ROOT="/home/zhaozhang/local/AGORA_work/AGORA_Data/z0"
WAIT_FOR_PID="${WAIT_FOR_PID:-758113}"
LOGDIR="${ROOT}/batch_logs_20k"
mkdir -p "${LOGDIR}"
cd "${ROOT}" || exit 1

echo "[$(date)] Waiting for CHANGA/current PID ${WAIT_FOR_PID} before full-projection rerun" | tee -a "${LOGDIR}/wait_changa_then_full_projection.log"
while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
  sleep 120
done

echo "[$(date)] PID ${WAIT_FOR_PID} is done; starting full-projection all-code rerun" | tee -a "${LOGDIR}/wait_changa_then_full_projection.log"
CHANGA_TIPSY_LENGTH_UNIT_KPC=1 CHANGA_TIPSY_MASS_UNIT_MSUN=1 PROJECTION_LOS_HALF_THICKNESS_KPC="" VELOCITY_REFERENCE_SOURCE=gas N_LOS=20000 N_JOBS=30 S_MAX_KPC=100 DS_KPC=0.25 RANDOM_OBSERVERS=128 PROJECTION_BOX_KPC=40 PROJECTION_NPIX=512 PROJECTION_MAX_ELEMENTS=100000000 PROJECTION_KERNEL_MAX_PARTICLES=500000 IONIZATION_MODE=auto /bin/bash run_all_agora_codes_20k.sh   > "${LOGDIR}/nohup_all_codes_20k_full_projection.log" 2>&1
status=$?
echo "[$(date)] full-projection rerun finished with status ${status}" | tee -a "${LOGDIR}/wait_changa_then_full_projection.log"
exit ${status}
