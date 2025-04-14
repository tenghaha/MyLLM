
import json
from typing import Sequence
from typing_extensions import Annotated, TypedDict

import streamlit as st

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder



# TODO: 待实现功能：Prompt储存、对话管理、流式传输、RAG、Agent Tools、OpenAI兼容
# 页面开始
st.title("Prompts")

# 1. 侧边栏：确定模型及参数
with st.sidebar:
    if "model_config" not in st.session_state:
        st.session_state["model_config"] = {"thread_id": "abc456"} # TODO: 对话管理：用config控制session_id
    
    st.session_state.model_config["api"] = st.selectbox(
        "API",
        options=["DeepSeek"],
        placeholder="选择API提供者"
    )
    #TODO: 自动选择模型
    st.session_state.model_config["model"] = st.selectbox(
        "模型",
        options=["deepseek-chat", "deepseek-reasoner"],
        placeholder="选择模型"
    )
    st.session_state.model_config["response_type"] = st.radio(
        "输出格式",
        options=["text","json_mode"],
        captions=["以纯文本格式输出","以JSON格式输出"]
    )
    st.session_state.model_config["max_tokens"] = st.slider(
        "max_tokens",
        1, 8192,
        value=4096,
        step=32,
        help="限制一次请求中模型生成 completion 的最大 token 数。输入 token 和输出 token 的总长度受模型的上下文长度的限制。"
    )
    st.session_state.model_config["temperature"] = st.slider(
        "temperature",
        0., 2.,
        value=1.,
        step=0.01,
        help="采样温度。更高的值会使输出更随机，而更低的值会使其更加集中和确定。"
    )
    st.session_state.model_config["top_p"] = st.slider(
        "top_p",
        0., 1.,
        value=1.,
        step=0.01,
        help="作为调节采样温度的替代方案，模型会考虑前 top_p 概率的 token 的结果。不建议与 temperature 同时修改。"
    )
    st.session_state.model_config["presence_penalty"] = st.slider(
        "presence_penalty",
        -2., 2.,
        value=0.,
        step=0.1,
        help="如果该值为正，那么新 token 会根据其是否已在已有文本中出现受到相应的惩罚，从而增加模型谈论新主题的可能性。"
    )
    st.session_state.model_config["frequency_penalty"] = st.slider(
        "frequency_penalty",
        -2., 2.,
        value=0.,
        step=0.1,
        help="介于 -2.0 和 2.0 之间的数字。如果该值为正，那么新 token 会根据其在已有文本中的出现频率受到相应的惩罚，降低模型重复相同内容的可能性。"
    )


# 2. 预处理：编译LangGraph

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

def call_model(state: State, config: RunnableConfig):
    model = ChatDeepSeek(
        # base_url=BASE_URL[st.session_state.model_config["api"]],
        model_name=config["configurable"].get("model"),
        max_tokens=config["configurable"].get("max_tokens"),
        temperature=config["configurable"].get("temperature"),
        top_p=config["configurable"].get("top_p"),
        frequency_penalty=config["configurable"].get("frequency_penalty"),
        presence_penalty=config["configurable"].get("presence_penalty")
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt_template.invoke(state)
    if config["configurable"].get("response_type") == "json_mode":
        response = model.with_structured_output(
            method='json_mode',
            include_raw=True
            ).invoke(prompt)["raw"] # 取原始JSON字符串message
    else:
        response = model.invoke(prompt)
    return {"messages": [response]}

workflow = StateGraph(state_schema=State, config_schema=ConfigSchema)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
workflow.add_edge("model", END)

memory = MemorySaver()
if "app" not in st.session_state:
    st.session_state["app"] = workflow.compile(checkpointer=memory)


# 3. 主页面：处理prompt和输入
class OutputWrapper():
    def __init__(self):
        self._cot_end = False
        self.values = {}
        self.sp = "------------------------------------"

    def __call__(self, output):
        for stream_mode, chunk in output:
            if stream_mode == "messages":
                msg_chunk, _ = chunk
                if "reasoning_content" not in msg_chunk.additional_kwargs and not msg_chunk.content:
                    yield ""
                elif "reasoning_content" in msg_chunk.additional_kwargs:
                    yield msg_chunk.additional_kwargs["reasoning_content"]
                else:
                    if not self._cot_end:
                        self._cot_end = True
                        yield f"\n\n{self.sp}\n\n"
                    yield msg_chunk.content
            else:
                self.values = chunk


@st.fragment()
def run_chat(output):
    output_wrapper = OutputWrapper()
    msg_placeholder = st.empty()
    with msg_placeholder:
        with st.status("思考中...", expanded=True):
            msg = st.write_stream(output_wrapper(output))
    msg_placeholder.empty()
    cot_msg, msg = msg.split(output_wrapper.sp)
    cot_msg, msg = cot_msg.strip(), msg.strip()
    if cot_msg:
        with st.expander("思维链"):
            st.caption(cot_msg)
    st.markdown(msg)
    st.session_state["messages"] = output_wrapper.values["messages"]


if "messages" not in st.session_state:
    st.session_state["messages"] = []

col1, col2 = st.columns(2)

with col1:
    with st.container(height=500):
        system_prompt = st.text_area(
            "Prompt模板"
        )

with col2:
    chat_box = st.container(height=500)
    chatbox_placeholder = chat_box.empty()
    if not st.session_state.messages:
        chatbox_placeholder.caption("输入消息以开始聊天")
    for msg in st.session_state.messages:
        chat_box.chat_message(msg.type).write(msg.content)

    if user_input := st.chat_input(placeholder="使用Prompt模板发送消息"):
        chatbox_placeholder.empty()
        chat_box.chat_message("user").write(user_input)

        input_messages = [HumanMessage(user_input)]
        output = st.session_state.app.stream(
            {"messages": input_messages, "system_prompt": system_prompt},
            config = {"configurable": st.session_state.model_config},
            stream_mode=["messages", "values"]
        )

        with chat_box.chat_message("ai"):
            run_chat(output)

        # st.session_state.messages = output["messages"]

        # output = st.session_state.app.invoke(
        #     {"messages": input_messages, "system_prompt": system_prompt},
        #     config = {"configurable": st.session_state.model_config}
        # )
        # # output["messages"][-1].pretty_print()

        # with chat_box.chat_message("ai"):
        #     if "reasoning_content" in output["messages"][-1].additional_kwargs:
        #         st.caption(output["messages"][-1].additional_kwargs["reasoning_content"])
        #     st.write(output["messages"][-1].content)
        # st.session_state.messages = output["messages"]


# 页面结束
