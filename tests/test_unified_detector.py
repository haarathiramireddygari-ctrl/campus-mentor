
from pathlib import Path

import pandas as pd
import pytest

from core.unified_detector import (
    analyze_unified,
    choose_unified_decision
)


BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def legacy():

    return {

        "recognized_features": 100,

        "risk_score": 20.0,

        "indicators": [],

        "safety_context": False,

        "mixed_context": False

    }


# Test 1: Low-risk decision
def test_low_risk(legacy):

    decision, _ = choose_unified_decision(
        20.0, legacy
    )

    assert decision == "Low Detected Risk"


# Test 2: Intermediate score
def test_intermediate_score(legacy):

    decision, _ = choose_unified_decision(
        45.0, legacy
    )

    assert decision == (
        "Manual Verification Required"
    )


# Test 3: Fraud alert
def test_binary_alert(legacy):

    decision, _ = choose_unified_decision(
        80.0, legacy
    )

    assert decision == "Potential Fraud"


# Test 4: Model disagreement
def test_model_disagreement(legacy):

    legacy["risk_score"] = 70.0

    decision, _ = choose_unified_decision(
        25.0, legacy
    )

    assert decision == (
        "Manual Verification Required"
    )


# Test 5: Strong security indicator
def test_strong_indicator(legacy):

    legacy["indicators"] = [
        {"indicator": "credential_request"}
    ]

    decision, _ = choose_unified_decision(
        10.0, legacy
    )

    assert decision == "Potential Fraud"


# Test 6: Insufficient recognized features
def test_insufficient_features(legacy):

    legacy["recognized_features"] = 2

    decision, _ = choose_unified_decision(
        10.0, legacy
    )

    assert decision == (
        "Manual Verification Required"
    )


# Test 7: Protective context
def test_protective_context(legacy):

    legacy["safety_context"] = True

    decision, _ = choose_unified_decision(
        85.0, legacy
    )

    assert decision == (
        "Manual Verification Required"
    )


# Test 8: Mixed context
def test_mixed_context(legacy):

    legacy["mixed_context"] = True

    decision, _ = choose_unified_decision(
        80.0, legacy
    )

    assert decision == (
        "Manual Verification Required"
    )


# Test 9: Empty input
def test_empty_input():

    result = analyze_unified("", "SMS")

    assert result["success"] is False


# Test 10: Invalid source
def test_invalid_source():

    result = analyze_unified(
        "Test message",
        "InvalidSource"
    )

    assert result["success"] is False


# Tests 11 and 12:
# Verify all 200 previously evaluated decisions
@pytest.mark.parametrize(
    "dataset_name",
    [
        "hybrid_development_100",
        "independent_holdout_100"
    ]
)
def test_evaluation_parity(dataset_name):

    dataset = pd.read_csv(

        BASE_DIR
        / "data"
        / f"{dataset_name}.csv"

    )

    previous = pd.read_csv(

        BASE_DIR
        / "reports"
        / "unified_v4"
        / dataset_name
        / "unified_predictions.csv"

    )

    assert len(dataset) == 100
    assert len(dataset) == len(previous)

    assert dataset["message"].equals(
        previous["message"]
    )

    assert dataset["source"].equals(
        previous["source"]
    )

    assert dataset["category"].equals(
        previous["actual_category"]
    )

    for index, row in dataset.iterrows():

        result = analyze_unified(

            row["message"],

            source=row["source"]

        )

        assert result["success"] is True

        assert result["decision"] == previous.loc[
            index, "unified_decision"
        ], f"Decision mismatch at row {index}"

        assert result["decision_reason"] == previous.loc[
            index, "decision_reason"
        ], f"Reason mismatch at row {index}"
