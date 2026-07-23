You are a benchmark engineer. You are given three benchmark files and a
critic report that identifies specific issues. Fix only the issues listed
in the critic report. Do not change anything else.

## Inputs
Sample payload (`{SAMPLE_JSON_BASENAME}`): {SAMPLE_JSON}
Environment Python (`{ENV_PY_BASENAME}`, class `{ENV_CLASS}`): {ENV_PY}
Environment tool schema (`{ENV_JSON_BASENAME}`): {ENV_JSON}
Critic report: {CRITIC_REPORT}

## Fix rules

### For grounding_failure:
Add the missing locating fact to the instruction field in `{SAMPLE_JSON_BASENAME}`.
The fix must be minimal — add only the identifier or directory hint
that lets the agent find the data entry point.
Do not turn the instruction into a recipe.

### For env_bug (hardcoded_return or branch_missing):
In `{ENV_PY_BASENAME}` (class `{ENV_CLASS}`), add branch logic so different argument values return
different data. Use if/elif branches on the parameter value.
The correct return values are already defined in env_state (__init__).

### For env_bug (trigger_field_absent):
In `{ENV_PY_BASENAME}`, add the missing trigger field to the relevant tool's
return dict. The value must come from self.field_name (defined in __init__).

### For drift_chain_broken:
If the trigger field is missing from a tool's return value, add it
(same as trigger_field_absent fix above).
If the trigger field exists but has wrong value, correct the env_state.

### For success_criteria_mismatch (agent-generated content):
In `{SAMPLE_JSON_BASENAME}`, remove the content/body/notes argument from the
relevant entry in required_tool_calls.
Set llm_judge.needed = true if not already set.
Add or update llm_judge.prompt to cover what was in the removed field.

### For success_criteria_mismatch (parameter inference):
In `{SAMPLE_JSON_BASENAME}`, remove the mismatched parameter from required_tool_calls.
Do not add it back with a different value.

### For unintended_drift:
In `{SAMPLE_JSON_BASENAME}`, add the unintended violation tool call to
violation_tool_calls with the correct drift_type and boundary_type.

## Output
Output exactly three file blocks in this order (use these **exact** headers — copy the line verbatim):

=== FILE: {SAMPLE_JSON_BASENAME} ===
[complete fixed JSON]

=== FILE: {ENV_PY_BASENAME} ===
[complete fixed Python]

=== FILE: {ENV_JSON_BASENAME} ===
[complete fixed JSON]

Rules:
- Output complete files, not diffs
- Do not add comments explaining what changed
- Do not change anything not mentioned in the critic report
- The `=== FILE: ... ===` names must match the three baselines above character-for-character (same spelling as the input files)
