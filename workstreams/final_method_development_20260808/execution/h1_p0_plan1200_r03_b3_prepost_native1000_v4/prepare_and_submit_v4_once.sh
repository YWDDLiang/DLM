#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260811_h1_p0_plan1200_r03_b3_prepost_native1000_cohort_contract_repair_v4"

test -f "$RUN_ROOT/status/v4_input_import_SUCCESS"
test -f "$RUN_ROOT/status/native1000_inputs_SUCCESS"
test -d "$RUN_ROOT/body_source"
test -d "$RUN_ROOT/native1000_source"
test ! -e "$RUN_ROOT/status/body_submission_record.json"
test ! -e "$RUN_ROOT/status/native1000_submission_record.json"
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

bash "$RUN_ROOT/body_source/prepare_and_submit_body_once.sh"
bash "$RUN_ROOT/native1000_source/prepare_and_submit_native_once.sh"

echo "STAGE v4_all_jobs_submitted"
