"""
Test suite for agent.py. Every external call (the risk-app API, Mireye,
Groq) is mocked -- these tests verify the agent's own logic (parsing,
error handling, the tool-call loop), not the third parties' actual
responses.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

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
                "factors": [
                    {"label": "Landslide susceptibility", "severity": "high"},
                    {"label": "Soil ponding frequency", "severity": "low"},
                ],
            },
            "tree_cover_loss_by_year": [
                {"umd_tree_cover_loss__year": 2020, "area_ha": 3.0},
                {"umd_tree_cover_loss__year": 2010, "area_ha": 5.0},  # before 2018, excluded
            ],
        })
        result = agent.get_erosion_context(40.0, -100.0)
        assert result["composite_score"] == 5
        assert result["recent_deforestation_ha_since_2018"] == 3.0
        assert result["top_factors"] == ["Landslide susceptibility"]

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
