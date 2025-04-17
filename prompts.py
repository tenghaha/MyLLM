import hashlib
import json
import time

import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from llm.state_graph import WorkFlow


# TODO: 待实现功能：用户认证、对话长度控制、RAG、Agent Tools、OpenAI兼容
# TODO: RAG功能：网络检索、本地文档上传（PDF或TXT）检索，知识库检索。网络检索和本地文档上传逻辑通用（临时RAG库），知识库检索需要包含知识库管理+检索功能
# TODO: RAG开发顺序：将网络检索工具加入graph - 设计整体检索功能，接口为本地上传文档+自建知识库 - 设计自建知识库
# 页面开始
st.header("Prompts")
st.caption("💬直接开始或使用预定的prompt模板进行会话")
st.caption("⚙️在左侧边栏调整模型参数，右侧主窗口进行会话/保存模板/导出会话记录等")

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

    st.session_state.model_config["use_web_context"] = st.checkbox(
        "联网搜索",
        help="模型是否检索互联网信息（消耗更多token）"
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
workflow = WorkFlow()
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
            # print(stream_mode, chunk)
            if stream_mode == "messages":
                msg_chunk, _ = chunk
                if isinstance(msg_chunk, ToolMessage):
                    num_web_page = len(json.loads(msg_chunk.content))
                    yield f"*引用 {num_web_page} 篇网页作为参考*"
                else:
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


@st.dialog("保存模板")
def save_system_prompt(system_prompt):
    if not system_prompt:
        st.write("要保存的prompt模板不可为空")
        if st.button("确定", use_container_width=True):
            st.rerun()
    else:
        prompt_name = st.text_input("模板名称", max_chars=20)
        prompt_desc = st.text_input("模板描述（可选）", max_chars=100)
        confirm_col, cancel_col = st.columns(2)
        if confirm_col.button("保存", use_container_width=True):
            with open("./prompts/prompts.json", encoding="utf-8") as fp:
                prompt_dict = json.loads(fp.read())
            with open("./prompts/prompts.json", "w", encoding="utf-8") as fp:
                if prompt_name in prompt_dict:
                    prompt_name = prompt_name + "_1"    # 模板重名处理
                prompt_dict[prompt_name] = {
                    "content": system_prompt,
                    "description": prompt_desc
                }
                fp.write(json.dumps(prompt_dict, ensure_ascii=False))
            st.rerun()
        if cancel_col.button("取消", use_container_width=True):
            st.rerun()


@st.dialog("更新模板")
def update_system_prompt(prompt_name ,system_prompt):
    if not system_prompt:
        st.write("要保存的prompt模板不可为空")
        if st.button("确定"):
            st.rerun()
    else:
        st.write("是否保存新的模板？")
        prompt_desc = st.text_input("模板描述（留空则不变）", max_chars=100)
        confirm_col, cancel_col = st.columns(2)
        if confirm_col.button("保存", use_container_width=True):
            with open("./prompts/prompts.json", encoding="utf-8") as fp:
                prompt_dict = json.loads(fp.read())
            with open("./prompts/prompts.json", "w", encoding="utf-8") as fp:
                prompt_dict[prompt_name]["content"] = system_prompt
                prompt_dict[prompt_name]["description"] = prompt_desc
                fp.write(json.dumps(prompt_dict, ensure_ascii=False))
            st.rerun()
        if cancel_col.button("取消", use_container_width=True):
            st.rerun()


@st.dialog("删除模板")
def delete_system_prompt(prompt_name):
    st.write("是否确认删除？ :red[本操作不可撤销！]")
    confirm_col, cancel_col = st.columns(2)
    if confirm_col.button("确认", use_container_width=True):
        with open("./prompts/prompts.json", encoding="utf-8") as fp:
            prompt_dict = json.loads(fp.read())
        with open("./prompts/prompts.json", "w", encoding="utf-8") as fp:
            prompt_dict.pop(prompt_name)
            fp.write(json.dumps(prompt_dict, ensure_ascii=False))
        st.rerun()
    if cancel_col.button("取消", use_container_width=True):
        st.rerun()


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
                "additional_kwargs": msg.additional_kwargs
                })
    return parsed_messages


if "messages" not in st.session_state:
    st.session_state["messages"] = []

prompt_col, main_col, widget_col = st.columns([0.35, 0.6, 0.05])

with prompt_col:    # TODO: 为每个用户设置不同的prompt库
    with st.container(height=550):
        prompt_save_placeholder = st.container()
        prompt_content_placeholder = st.empty()
        prompt_desc_placeholder = st.empty()

        system_prompt = prompt_content_placeholder.text_area(
            "Prompt模板",
            height=200,
            max_chars=1000,
            placeholder="提示需要模型做的事情\n例如：“帮我将输入的中文翻译为英文”"
        )
        with prompt_desc_placeholder.container(height=130):
            st.write()

        with open("./prompts/prompts.json", encoding="utf-8") as fp:
            saved_prompts = json.loads(fp.read())
        prompt_name = prompt_save_placeholder.selectbox(
            "选择模板",
            options=["(新模板)"] + list(saved_prompts.keys())
        )
        if prompt_name == "(新模板)":
            if prompt_save_placeholder.button("保存当前模板", icon=":material/save:", use_container_width=True):
                save_system_prompt(system_prompt)   # 如果选择新模板，可以保存模板至./prompts/prompts.json
        else:
            system_prompt = prompt_content_placeholder.text_area(
                "Prompt模板",
                value=saved_prompts[prompt_name]["content"],
                height=200,
                max_chars=1000,
                placeholder="提示需要模型做的事情\n例如：“帮我将输入的中文翻译为英文”"
            )

            with prompt_desc_placeholder.container(height=130):
                st.write(saved_prompts[prompt_name]["description"])

            p_col1, p_col2 = prompt_save_placeholder.columns(2)  # 选择已有模板的逻辑
            if p_col1.button("更新", icon=":material/sync:", use_container_width=True):
                update_system_prompt(prompt_name, system_prompt)
            if p_col2.button("删除", icon=":material/delete:", use_container_width=True):
                delete_system_prompt(prompt_name)


with main_col:
    chat_box = st.container(height=550)
    chatbox_placeholder = chat_box.empty()
    if not st.session_state.messages:
        chatbox_placeholder.caption("发送消息以开始聊天")
    for msg in st.session_state.messages:
        if msg.type in ["human", "ai"] and msg.content:
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
        

if user_input := st.chat_input(placeholder="发送消息", accept_file="multiple", file_type=["pdf", "txt"]):
    chatbox_placeholder.empty()
    chat_box.chat_message("user").write(user_input.text)

    input_messages = [HumanMessage(user_input.text)]
    output = st.session_state.app.stream(
        {"messages": input_messages, "system_prompt": system_prompt},
        config = {"configurable": st.session_state.model_config},
        stream_mode=["messages", "values"]
    )

    with chat_box.chat_message("ai"):
        run_chat(output)
    
    st.rerun()

# with st.expander("debug"):
    # st.write(st.session_state.app.get_state({"configurable": st.session_state.model_config}))
# 页面结束
