import streamlit as st
import openai
import re

# --- 页面配置 ---
st.set_page_config(
    page_title="AI 学习笔记生成器",
    page_icon="🚀",
    layout="wide"
)


# --- AI 函数封装 ---

def create_openai_client(api_key, base_url):
    """根据用户输入创建 OpenAI 客户端"""
    return openai.OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else "https://api.openai.com/v1"
    )


def generate_outline(client, model, temperature, chat_log):
    """第一步：调用 AI 生成 Markdown 大纲"""
    system_prompt = """
    你是一个非常优秀的学习笔记总结助理。
    你的任务是：通读我提供的聊天记录全文，分析出对话的内在逻辑和知识体系。
    然后，请以 Markdown 标题的形式（例如 `##` 或 `###`），生成一个简洁的、层级分明的学习笔记大纲。
    你生成的回复中，必须只包含这个 Markdown 标题大纲，不要有任何其他解释或说明文字。
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chat_log}
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"生成大纲失败：{e}")
        return None


def generate_details_streaming(client, model, temperature, chat_log, heading):
    """第二步：根据单个标题，流式生成详细内容"""
    system_prompt = f"""
    你是一个顶级的学习笔记专家。
    你的任务是：根据我提供的聊天记录全文，以及一个特定的标题，详细总结和阐述与「{heading}」这个标题相关的所有内容。
    请你只输出和这个标题直接相关的内容，确保总结的专业、详尽、易于理解。
    请使用 Markdown 格式进行输出，可以包含要点、代码块、引用等。
    你的输出将直接作为笔记内容，所以不要包含任何额外的、与笔记内容无关的解释，比如“好的，这是关于xxx的总结：”。
    """
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chat_log}  # 再次提供全文作为上下文
            ],
            temperature=temperature,
            stream=True,  # ！！！这是实现流式输出的关键！！！
        )
        for chunk in stream:
            # 检查块中是否有内容
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        st.error(f"生成「{heading}」的详细内容失败：{e}")


# --- 侧边栏 ---
with st.sidebar:
    st.header("🔑 API 设置")
    base_url = st.text_input("API Base URL (可选)")
    api_key = st.text_input("请输入你的 API Key", type="password")

    st.header("⚙️ 模型配置")
    selected_model = st.selectbox("选择模型", ["pplx-claude-sonnet-4-20250514", "gemini-2.5-pro", "gpt-4.1"])
    temperature = st.slider("温度 (Temperature)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# --- 主界面 ---
st.title("AI 学习笔记生成器 🚀")

chat_log = st.text_area(
    "请在此处粘贴你的聊天记录...",
    height=400,
    placeholder="例如：\n\n用户: 你好，请帮我解释一下什么是“机器学习”？\nAI: 机器学习是人工智能的一个分支..."
)

if st.button("✨ 开始生成笔记", type="primary"):
    if not api_key:
        st.error("请输入你的 API Key！")
    elif not chat_log:
        st.error("请输入聊天记录！")
    else:
        # 创建客户端
        client = create_openai_client(api_key, base_url)

        # --- 第一步：生成大纲 ---
        with st.spinner("🧠 AI 正在为你生成大纲..."):
            outline = generate_outline(client, selected_model, temperature, chat_log)

        if outline:
            st.success("大纲生成完毕！开始填充详细内容...")

            # 使用正则表达式解析出所有 Markdown 标题
            headings = re.findall(r"^#+\s+(.*)", outline, re.MULTILINE)

            st.markdown("---")
            st.subheader("📖 最终学习笔记")

            # --- 第二步：遍历大纲，流式填充细节 ---
            final_note = ""  # 用于累加最终结果
            for heading in headings:
                # 重新组合成 Markdown 标题格式并显示
                full_heading = f"## {heading}"  # 为了统一格式，我们都用二级标题
                st.markdown(full_heading)
                final_note += full_heading + "\n\n"

                # 使用 st.write_stream 来显示流式响应
                response_stream = generate_details_streaming(client, selected_model, temperature, chat_log, heading)

                # st.write_stream 会自动处理生成器，并将内容逐块显示在页面上
                full_response = st.write_stream(response_stream)
                final_note += full_response + "\n\n"

            st.success("🎉 笔记生成完毕！")

            # 提供一个下载按钮
            st.download_button(
                label="📥 下载笔记 (Markdown)",
                data=final_note,
                file_name="学习笔记.md",
                mime="text/markdown",
            )