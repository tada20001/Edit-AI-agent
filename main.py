import operator
from typing import List, Annotated, Sequence
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.graph import END, StateGraph

from chains import generate_chain, reflect_chain
import streamlit as st


# --- 1. 상태에 '다음 경로'를 저장할 필드 추가 ---
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# --- 2. 노드 및 조건부 엣지 함수 정의 ---
def generate_node(state: GraphState):
    res = generate_chain.invoke({"messages": state["messages"]})
    return {"messages": [res]}

def reflect_node(state: GraphState):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}

# ❗수정됨: 함수의 이름과 로직을 "검토 결과에 따라" 분기하도록 변경했습니다.
def grade_generation(state: GraphState):
    """
    Reflector의 피드백을 확인하여 다음 행동을 결정합니다.
    - "성공"이 포함되어 있으면 그래프를 종료합니다.
    - 그렇지 않으면, 루프를 계속합니다.
    - 최대 3번의 수정-검토 사이클(총 메시지 6개)을 초과하면 안전하게 종료합니다.
    """
    # 가장 마지막 메시지가 Reflector의 피드백입니다.
    last_message = state["messages"][-1]

    # 최대 반복 횟수 도달 시 종료 (무한 루프 방지)
    if len(state["messages"]) > 6:
        print("--- 최대 반복 횟수에 도달하여 종료합니다. ---")
        return "end"

    # 피드백 내용에 "성공"이 포함되어 있는지 확인
    if "성공" in last_message.content:
        print("--- 검토 결과 '성공'이므로 종료합니다. ---")
        return "end"
    else:
        print("--- 검토 결과 '실패'이므로 수정을 계속합니다. ---")
        return "continue"


# --- 3. 그래프 생성 및 컴파일 ---
@st.cache_resource
def build_graph():
    # 상수 정의
    GENERATE = "generate"
    REFLECT = "reflect"

    builder = StateGraph(GraphState)

    builder.add_node(GENERATE, generate_node)
    builder.add_node(REFLECT, reflect_node)
    builder.set_entry_point(GENERATE)

    # ❗수정됨: 그래프 흐름을 재구성했습니다.
    # 1. 생성(GENERATE) 후에는 항상 검토(REFLECT)로 갑니다.
    builder.add_edge(GENERATE, REFLECT)

    # 2. 검토(REFLECT) 후에 조건부로 분기합니다.
    builder.add_conditional_edges(
        REFLECT,
        grade_generation, # 새로운 조건 함수 사용
        {
            "continue": GENERATE, # 'continue'를 반환하면 다시 GENERATE로
            "end": END            # 'end'를 반환하면 종료
        }
    )

    graph = builder.compile()
    return graph


graph = build_graph()
load_dotenv()


# --- 4. Streamlit UI 구성 ---
st.title("🤖 문장 수정을 위한 AI Agent")
st.markdown("### 발표자료, 보고서 문구 등을 입력하면 AI가 스스로 검토하고 수정합니다.")

# ❗수정됨: 가독성을 위해 노드 이름을 그래프 빌더 안으로 옮겼습니다.
NODE_NAME_MAP = {
    "generate": "🤖 AI 초안 생성/수정",
    "reflect": "🧐 AI 자체 검토 및 피드백",
}

# 사용자 입력을 받을 텍스트 영역
user_input = st.text_area("수정하고 싶은 전체 문장을 여기에 붙여넣으세요:", height=150,
                          placeholder="여기에 원본 텍스트를 입력하세요...")


if st.button("AI 실행하기"):
    if not user_input:
        st.warning("텍스트를 입력해주세요.")
    else:
        initial_prompt = f"""아래 텍스트를 비평하고 더 나은 버전으로 수정해주세요.
            ---
            원본 텍스트:
            {user_input}
            """
        # 초기 입력값 설정
        initial_message = HumanMessage(content=initial_prompt)
        inputs = {"messages": [initial_message]}
        st.markdown("---")

        final_answer = ""
        # AI가 작업하는 동안 스피터 표시
        with st.spinner("AI 에이전트가 생각중입니다..."):
            # stream을 사용해 각 단계의 결과를 실시간으로 받음
            for step in graph.stream(inputs, {"recursion_limit": 10}):
                # step은 {"node_name": state} 형태의 디셔너리임
                node_name = list(step.keys())[0]
                state = list(step.values())[0]
                display_name = NODE_NAME_MAP.get(node_name)

                if display_name:
                    st.subheader(display_name)
                    last_message = state["messages"][-1]
                    if isinstance(last_message, HumanMessage):
                        st.info(f"👤 **[AI의 자체 피드백]**\n\n{last_message.content}")
                    elif isinstance(last_message, AIMessage):
                        st.success(f"🤖 **[AI의 생성/수정안]**\n\n{last_message.content}")
                        # ❗수정됨: AI가 답변을 생성할 때마다 그 내용을 final_answer에 업데이트합니다.
                        final_answer = last_message.content

                #final_state = state

        st.markdown("---")
        st.header("✅ 최종 결과")
       # ❗수정됨: 루프가 끝난 후, 저장해 둔 마지막 AI 답변을 출력합니다.
        if final_answer:
            st.markdown(final_answer)
        else:
            st.error("최종 결과물을 찾을 수 없습니다.")







