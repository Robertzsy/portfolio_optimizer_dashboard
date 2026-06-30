from portfolio.ai_analysis import AIProviderConfig, build_portfolio_prompt, default_system_prompt


def test_build_portfolio_prompt_includes_snapshot():
    snapshot = {"assets": ["SPY", "TLT"], "max_sharpe": {"sharpe": 1.1}}
    messages = build_portfolio_prompt(snapshot, "English")

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "SPY" in messages[1]["content"]
    assert "not investment advice" in messages[0]["content"]
    assert "senior professional market analyst" in messages[0]["content"]
    assert "beginner" not in messages[1]["content"].lower()


def test_custom_system_prompt_overrides_default():
    custom = "You are a custom institutional macro strategist."
    messages = build_portfolio_prompt({"assets": ["SPY"]}, "English", system_prompt=custom)

    assert messages[0]["content"] == custom
    assert default_system_prompt("English") != custom


def test_ai_provider_config_keeps_custom_provider_values():
    config = AIProviderConfig(
        provider_name="Custom",
        sdk_type="OpenAI-compatible",
        base_url="https://api.example.com/v1",
        api_key="sk-test",
        model="custom-model",
    )

    assert config.sdk_type == "OpenAI-compatible"
    assert config.model == "custom-model"
