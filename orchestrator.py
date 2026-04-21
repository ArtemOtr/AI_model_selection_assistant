import json
from typing import Any, Dict
from agents.parser_agent import ParserCacheAgent
from agents.selection_agents import SelectionTools, SelectionRequirements, build_requirements_adk_agent, build_ranking_adk_agent
import json
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps import App
from google.adk.agents import LlmAgent

import json
from typing import Any, Dict


class Orchestrator:
    def __init__(self, agent1, agent2):
        self.agent1 = agent1
        self.agent2 = agent2

    async def run_pipeline(self, user_prompt: str) -> Dict[str, Any]:

        # =====================
        # 1. FIRST AGENT
        # =====================
        resp1 = await self.agent1.call(user_prompt)

        raw_requirements = resp1.output_text if hasattr(resp1, "output_text") else str(resp1)

        try:
            requirements = json.loads(raw_requirements)
        except Exception:
            requirements = raw_requirements

        # =====================
        # 2. SECOND AGENT
        # =====================
        second_input = {
            "user_prompt": user_prompt,
            "requirements": requirements
        }

        resp2 = await self.agent2.call(json.dumps(second_input, ensure_ascii=False))

        ranking_result = resp2.output_text if hasattr(resp2, "output_text") else str(resp2)

        # =====================
        # FINAL OUTPUT
        # =====================
        return {
            "requirements": requirements,
            "ranking_result": ranking_result
        }


if __name__ == "__main__":
    import asyncio

    async def _run():
        parser = ParserCacheAgent()
        await parser.start()

        agent1 = build_requirements_adk_agent(parser)
        agent2 = build_ranking_adk_agent(parser)

        orchestrator = Orchestrator(agent1, agent2)

        while True:
            prompt = input("> ")

            if prompt.lower() in ("exit", "quit"):
                break

            result = await orchestrator.run_pipeline(prompt)
            print(result)

    asyncio.run(_run())