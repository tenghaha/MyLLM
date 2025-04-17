import os
import json
from typing import Sequence, Type
from typing_extensions import Annotated, TypedDict

from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults



class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    system_prompt: str

class ConfigSchema(TypedDict):
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float

# 工具节点：搜索功能
search_tool = TavilySearchResults(name="search_tool", max_results=5)

# 主agent
def agent(state: State, config: RunnableConfig):
    model = ChatDeepSeek(
        # base_url=BASE_URL[st.session_state.model_config["api"]],
        model_name=config["configurable"].get("model"),
        max_tokens=config["configurable"].get("max_tokens"),
        temperature=config["configurable"].get("temperature"),
        top_p=config["configurable"].get("top_p"),
        frequency_penalty=config["configurable"].get("frequency_penalty"),
        presence_penalty=config["configurable"].get("presence_penalty")
    )
    # 决定是否call search_tool
    if config["configurable"].get("use_web_context"):
        tool_choice = "auto"
    else:
        tool_choice = "none"
    model = model.bind_tools(
        [search_tool],
        tool_choice=tool_choice
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    if method := config["configurable"].get("response_type") == "json_mode":
        chain = prompt_template | model.with_structured_output(method)
        response = {"role": "ai", "content": json.dumps(chain.invoke(state), ensure_ascii=False)}
    else:
        chain = prompt_template | model
        response = chain.invoke(state)
    return {"messages": [response]}


class WorkFlow(StateGraph):
    def __init__(self):
        super().__init__(state_schema=State, config_schema=ConfigSchema)
        self.add_edge(START, "agent")
        self.add_node("agent", agent)
        self.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "search",
                END: END
            }
        )
        self.add_node("search", ToolNode([search_tool]))
        self.add_edge("search", "agent")
