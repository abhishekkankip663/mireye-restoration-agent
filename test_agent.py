"""
Test suite for agent.py. Every external call (the risk-app API, Mireye,
Groq) is mocked -- these tests verify the agent's own logic (parsing,
error handling, the tool-call loop), not the third parties' actual
responses.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

import agent


def mock_response(json_data=None, status_ok=True, status_code=200):
    resp = MagicMock()
    resp.json.return_value = json_data if json_data is not None else {}
    resp.status_code = status_code
    if status_ok:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = Exception("boom")
    return resp


class TestGetErosionContext:
    @patch("agent.requests.get")
    def test_parses_real_response_shape(self, mock_get):
        mock_get.return_value = mock_response({
            "risk": {
                "score": 5, "level": "moderate", "confidence": "high",
                "rusle_lite": {"relative_index": 0.5, "annual_soil_loss_tons_per_acre": 12.3},
                "rusle_unavailable_reason": None,
                "factors": [
                    {"label": "Landslide susceptibility", "detail": "72", "severity": "high"},
                    {"label": "Soil ponding frequency", "detail": "None", "severity": "low"},
                ],
                "data_completeness": {
                    "factors_evaluated": 7, "factors_total": 7,
                    "unavailable_factors": [], "imputed_points": {},
                },
            },
            "tree_cover_loss_by_year": [
                {"umd_tree_cover_loss__year": 2020, "area_ha": 3.0},
                {"umd_tree_cover_loss__year": 2010, "area_ha": 5.0},  # before 2018, excluded
            ],
        })
        result = agent.get_erosion_context(40.0, -100.0)
        assert result["composite_score"] == 5
        assert result["recent_deforestation_ha_since_2018"] == 3.0
        assert result["factors"] == [
            {"label": "Landslide susceptibility", "detail": "72", "severity": "high"},
            {"label": "Soil ponding frequency", "detail": "None", "severity": "low"},
        ]
        assert result["data_completeness"]["unavailable_factors"] == []
        assert result["data_completeness"]["factors_evaluated"] == 7

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_timeout_is_retried_not_given_up_on_immediately(self, mock_get, mock_sleep):
        # a cold-starting host can take longer than the per-request timeout
        # to respond -- that's a Timeout, a different exception type than
        # ConnectionError, and has to be retried the same way a 502 is
        mock_get.side_effect = [
            requests.exceptions.Timeout("timed out"),
            mock_response({"risk": {"score": 2, "level": "low"}}, status_code=200),
        ]
        result = agent.get_erosion_context(40.0, -100.0)
        assert result["composite_score"] == 2
        assert mock_get.call_count == 2
        assert mock_sleep.called

    @patch("agent.requests.get", side_effect=Exception("network down"))
    def test_network_failure_returns_error_dict_not_raise(self, _mock):
        result = agent.get_erosion_context(40.0, -100.0)
        assert "error" in result

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_retries_once_on_502_then_succeeds(self, mock_get, mock_sleep):
        # a cold-starting host returns a transient 502 first, then a real result
        mock_get.side_effect = [
            mock_response(status_code=502),
            mock_response({"risk": {"score": 3, "level": "moderate"}}, status_code=200),
        ]
        result = agent.get_erosion_context(40.0, -100.0)
        assert result["composite_score"] == 3
        assert mock_get.call_count == 2
        assert mock_sleep.called

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_gives_up_after_persistent_502(self, mock_get, mock_sleep):
        # get_erosion_context uses attempts=4 (RISK_APP_URL's cold starts
        # can run 40-60s+, so it gets a wider retry budget than the default)
        mock_get.side_effect = [
            mock_response(status_code=502),
            mock_response(status_code=502),
            mock_response(status_code=502),
            mock_response(status_ok=False, status_code=502),
        ]
        result = agent.get_erosion_context(40.0, -100.0)
        assert "error" in result
        assert mock_get.call_count == 4


class TestGetEconomicContext:
    @patch("agent.get_mireye_token", return_value="fake-token")
    @patch("agent.requests.post")
    def test_parses_real_response_shape(self, mock_post, _mock_token):
        mock_post.return_value = mock_response({
            "disposition": "resolved",
            "county": "Test County",
            "state": "Test State",
            "county_market": {"population": 1000, "median_household_income_usd": 50000},
            "in_opportunity_zone": True,
        })
        result = agent.get_economic_context(40.0, -100.0)
        assert result["county"] == "Test County"
        assert result["population"] == 1000
        assert result["in_opportunity_zone"] is True

    @patch("agent.get_mireye_token", return_value="fake-token")
    @patch("agent.requests.post")
    def test_unresolved_disposition_returns_error_not_partial_data(self, mock_post, _mock_token):
        mock_post.return_value = mock_response({"disposition": "clarify", "candidates": []})
        result = agent.get_economic_context(40.0, -100.0)
        assert "error" in result

    def test_missing_token_returns_error_dict_not_raise(self):
        with patch.dict("os.environ", {}, clear=True):
            result = agent.get_economic_context(40.0, -100.0)
        assert "error" in result


class TestRunPrioritizationAgent:
    def test_raises_without_groq_key(self):
        with patch.object(agent, "GROQ_API_KEY", None):
            with pytest.raises(RuntimeError):
                agent.run_prioritization_agent([{"name": "X", "lat": 1, "lng": 2}])

    @patch.object(agent, "GROQ_API_KEY", "fake-key")
    @patch("agent._groq_chat")
    def test_calls_tools_then_returns_final_recommendation(self, mock_chat):
        # first turn: model calls a tool; second turn: model answers
        tool_call_turn = {
            "choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "get_erosion_context", "args": None,
                                 "arguments": json.dumps({"lat": 1.0, "lng": 2.0})},
                }],
            }}]
        }
        final_turn = {"choices": [{"message": {"role": "assistant", "content": "Prioritize X."}}]}
        mock_chat.side_effect = [tool_call_turn, final_turn]

        with patch.object(agent, "TOOL_IMPLS", {"get_erosion_context": lambda a: {"composite_score": 1}}):
            result = agent.run_prioritization_agent([{"name": "X", "lat": 1.0, "lng": 2.0}])

        assert result["recommendation"] == "Prioritize X."
        assert len(result["trace"]) == 1
        assert result["trace"][0]["tool"] == "get_erosion_context"

    @patch.object(agent, "GROQ_API_KEY", "fake-key")
    @patch("agent._groq_chat")
    def test_step_cap_prevents_infinite_loop(self, mock_chat):
        # model keeps calling tools forever -- must stop, not hang
        never_stops = {
            "choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": "call_x",
                    "function": {"name": "get_erosion_context", "arguments": json.dumps({"lat": 1, "lng": 2})},
                }],
            }}]
        }
        mock_chat.return_value = never_stops

        with patch.object(agent, "TOOL_IMPLS", {"get_erosion_context": lambda a: {"ok": True}}):
            result = agent.run_prioritization_agent([{"name": "X", "lat": 1.0, "lng": 2.0}])

        assert result["recommendation"] is None
        assert "error" in result
        assert len(result["trace"]) == 12

    @patch.object(agent, "GROQ_API_KEY", "fake-key")
    @patch("agent._groq_chat")
    def test_step_cap_scales_with_candidate_count(self, mock_chat):
        # a fixed 12-round cap would cut off a legitimate 10-parcel batch
        # (2 tool calls each = 20+ rounds needed) before it could finish
        never_stops = {
            "choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": "call_x",
                    "function": {"name": "get_erosion_context", "arguments": json.dumps({"lat": 1, "lng": 2})},
                }],
            }}]
        }
        mock_chat.return_value = never_stops
        candidates = [{"name": f"P{i}", "lat": float(i), "lng": float(i)} for i in range(10)]

        with patch.object(agent, "TOOL_IMPLS", {"get_erosion_context": lambda a: {"ok": True}}):
            result = agent.run_prioritization_agent(candidates)

        assert len(result["trace"]) == 30  # max(12, 10 * 3)
