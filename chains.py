from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

google_api_key = os.environ.get("GOOGLE_API_KEY")
# ❗수정됨: 각 LLM에 temperature 값을 설정합니다.
llm_generate = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=google_api_key,
    temperature=0.2 # 약간의 창의성을 허용
)

llm_reflect = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=google_api_key,
    temperature=0.0 # 매우 엄격하고 일관된 평가를 위해
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 컨설팅 펌의 품질 관리(QA) 파트너입니다. 제출된 수정안이 아래 '품질 평가 가이드라인'을 완벽하게 준수했는지 냉철하게 평가하세요.

        ❗**품질 평가 가이드라인 (QA Checklist):**
        1.  **❗수정됨: 문장 구조 및 길이 (매우 중요):**
            - **(Best Case):** 핵심 정보를 모두 포함한 **완벽한 한 문장**이며, 길이는 **35~45단어** 내외인가? -> **"성공"**
            - **(Acceptable Case):** 내용이 복잡하여 예외적으로 **두 문장**으로 작성되었으며, 각 문장이 명료하고 합쳐서 60단어 이내인가? -> **"성공"**
            - **(Failure Case):** 한 문장이 50단어를 초과하여 너무 길거나, 세 문장 이상이거나, 의미가 불분명한가? -> **"실패"**
        2.  **C-level 적합성:** C-level 임원이 보고받기에 충분히 명료하고 전문적인가?
        3.  **주장의 명확성:** 핵심 주장이나 결론이 즉시 파악되는가? (두괄식 구조)
        4.  **원본 충실도:** 원본의 핵심 의미나 데이터가 훼손되지는 않았는가?

        **피드백은 '성공' 또는 '실패'로 시작하세요.** 실패 시, 위 가이드라인 중 어떤 항목을 위반했는지(예: 문장이 너무 김, 세 문장 이상임) 구체적인 이유를 한 문장으로 설명하세요.
        """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)



generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 최고 수준의 전략 컨설턴트입니다. 당신의 임무는 주어진 초안을 C-level 임원들에게 보고하기에 적합한, 명확하고 논리적이며 설득력 있는 비즈니스 문장으로 재작성하는 것입니다.

    ❗**재작성 원칙 (Rules of Engagement):**
    1.  **❗수정됨: 한 문장 우선 원칙:**
        - **최우선 목표는 모든 핵심 정보를 '단 한 문장'에 응축하는 것입니다.**
        - 문장은 35~45단어 내외로 간결해야 합니다.
    2.  **❗추가됨: 두 문장 예외 조항:**
        - 만약 한 문장으로 요약할 경우, 문장이 50단어를 초과하여 너무 길어지거나, 중요한 의미가 누락된다고 판단될 때만 **예외적으로 '두 문장'까지 허용합니다.**
        - 두 문장으로 작성할 경우, 각 문장은 독립적으로 완결되어야 하며, 합쳐서 60단어를 넘지 않도록 간결하게 작성하세요.
    3.  **두괄식 구성:** 핵심 결론이나 가장 중요한 문장을 가장 앞에 제시하세요.
    4.  **객관적 톤:** 모호한 추측성 표현을 제거하고, 사실 위주로 서술하세요.
    5.  **의미 보존:** 원본의 핵심 메시지나 데이터는 절대 왜곡하거나 누락하지 마세요.
    """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generate_chain = generation_prompt | llm_generate
reflect_chain = reflection_prompt | llm_reflect











