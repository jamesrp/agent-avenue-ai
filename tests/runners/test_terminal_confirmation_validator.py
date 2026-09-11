from pathlib import Path

from agent_avenue.runners import DEFAULT_EXCLUDED_SETUP_ROOTS


def test_validator_uses_no_shared_confirmation_aggregate_or_audit_entry_points() -> None:
    source = Path("scripts/validate_terminal_offense_confirmation_v1.py").read_text()

    for forbidden in (
        "audit" + "_terminal_confirmation",
        "confirmation" + "_statistics",
        "validate" + "_confirmation_corpora",
    ):
        assert forbidden not in source
    assert "inspect_source_identity" in source
    assert "validator-local-stratified-joint-bootstrap-v1" in source


def test_default_setup_holdout_scope_is_all_of_runs() -> None:
    assert (Path("runs"),) == DEFAULT_EXCLUDED_SETUP_ROOTS
