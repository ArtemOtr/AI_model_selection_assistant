
from agents.parser_agent import ParserCacheAgent
from agents.selection_agent import build_selection_agent


parser_agent1 = ParserCacheAgent()
root_agent = build_selection_agent(parser_agent=parser_agent1)

