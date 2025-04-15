import hashlib
import json
import time
from typing import Sequence
from typing_extensions import Annotated, TypedDict

import streamlit as st

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder



# TODO: 待实现功能：Prompt储存、用户认证、RAG、Agent Tools、OpenAI兼容
# 页面开始
st.title("Prompts")

# 1. 侧边栏：确定模型及参数
def hash_from_time():
    time_bytes = str(time.time()).encode("utf-8")
    return hashlib.md5(time_bytes).hexdigest()

def restart_session():
    st.session_state["model_config"]["thread_id"] = hash_from_time()
    st.session_state["messages"] = []
    st.session_state["confirmed_restart"] = True
    st.rerun()

@st.dialog("注意")
def if_restart():
    st.write("开始一段新的对话并清空所有对话上下文，确认吗？")
    c1, c2 = st.columns(2, vertical_alignment="bottom")
    with c1:
        if st.button("确认", use_container_width=True):
            restart_session()
    with c2:
        if st.button("取消", use_container_width=True):
            st.rerun()       


with st.sidebar:
    if "model_config" not in st.session_state:
        st.session_state["model_config"] = {"thread_id": hash_from_time()}
    
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

    # st.info(st.session_state.model_config["model"])

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


def parse_messages_history(_messages: list):
    parsed_messages = []
    for msg in _messages:
        if isinstance(msg, HumanMessage):
            parsed_messages.append({
                "role": "human",
                "content": msg.content
                })
        elif isinstance(msg, AIMessage):
            parsed_messages.append({
                "role": "ai",
                "content": msg.content,
                "additional_kwargs": msg.additional_kwargs  # TODO:如何处理tool call
                })
    return parsed_messages


if "messages" not in st.session_state:
    st.session_state["messages"] = []

prompt_col, main_col, widget_col = st.columns([0.35, 0.6, 0.05])

with prompt_col:
    with st.container(height=500):
        prompt_pattern = st.selectbox(
            "选择模板",
            options=["(新模板)","__"]
        )
        if prompt_pattern == "(新模板)":
            st.button("保存模板", use_container_width=True) # 如果选择新模板，可以保存模板至prompts.json
        else:
            p_col1, p_col2 = st.columns(2)  # 选择已有模板的逻辑
            p_col1.button("更新模板", use_container_width=True)
            p_col2.button("删除模板", use_container_width=True)
        system_prompt = st.text_area(
            "Prompt模板",
            height=300,
            max_chars=1000,
            placeholder="提示需要模型做的事情\n例如：“帮我将输入的中文翻译为英文”"
        )

with main_col:
    chat_box = st.container(height=500)
    chatbox_placeholder = chat_box.empty()
    if not st.session_state.messages:
        chatbox_placeholder.caption("发送消息以开始聊天")
    for msg in st.session_state.messages:
        chat_box.chat_message(msg.type).write(msg.content)

with widget_col:
    if st.button("", icon=":material/restart_alt:", help="开始新对话"):
        if_restart()
    parsed_messages = parse_messages_history(st.session_state.messages)
    parsed_messages = [{"role": "system", "content": system_prompt}] + parsed_messages
    st.download_button(
        "",
        data=json.dumps(parsed_messages, ensure_ascii=False),
        file_name=f"对话记录_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}.json",
        icon=":material/download:",
        help="导出对话记录")
        

if user_input := st.chat_input(placeholder="发送消息"):
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

# with st.expander("debug"):
#     st.write(st.session_state.messages)
# 页面结束
