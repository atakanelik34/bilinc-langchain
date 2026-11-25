from __future__ import annotations

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_bilinc import BilincRecall, BilincRemember

ENV = ({"BILINC_API_KEY": "env-key"}, {}, {"bilinc_api_key": "env-key"})


class TestBilincRememberStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BilincRemember]:
        return BilincRemember

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"key": "user.timezone", "value": "Europe/Istanbul"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return ENV


class TestBilincRecallStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BilincRecall]:
        return BilincRecall

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "what timezone is the user in?"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return ENV
