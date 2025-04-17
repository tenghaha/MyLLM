import os
import json
from typing import Sequence, Type
from typing_extensions import Annotated, TypedDict

from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults



class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    system_prompt: str
    retriever: VectorStoreRetriever

class ConfigSchema(TypedDict):
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float


class WorkFlow(StateGraph):
    def __init__(self, retriever: VectorStoreRetriever | None = None):
        super().__init__(state_schema=State, config_schema=ConfigSchema)
        self.search_tool = TavilySearchResults(
            name="search_tool",
            description="Searching webpages to provide user with an informed response. ",
            max_results=5
        )    # 工具节点：搜索功能
        self.retriever_tool = None
        if retriever:
            self.retriever_tool = create_retriever_tool(
                retriever,
                name="retriever_tool",
                description="Analyze the contents of the user's uploaded documents to extract pertinent information."
            )
        self._build()

    
    # 主agent
    def _agent(self, state: State, config: RunnableConfig):
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
            [self.search_tool],
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
    
    # 构建计算图
    def _build(self):
        self.add_edge(START, "agent")
        self.add_node("agent", self._agent)
        self.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "search",
                END: END
            }
        )
        self.add_node("search", ToolNode([self.search_tool]))
        self.add_edge("search", "agent")
