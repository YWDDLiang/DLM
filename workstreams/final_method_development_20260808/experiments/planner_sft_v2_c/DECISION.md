# Decision: Planner SFT-v2-C

Status: `SCIENTIFIC_STOP_NOT_SHORTLISTED`

Retain the fixed endpoint and all raw evidence, but do not spend downstream
compute on SFT-v2-C. It ties SFT-v2 on legacy comp_valid while being worse on
parse, unique-formula rate, element coverage, mean-N drift, and exact-SMACT4
secondary validity. SFT-v2 is the sole user-authorized diagnostic Planner arm.
