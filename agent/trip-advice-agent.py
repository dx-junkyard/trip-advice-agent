import vertexai
from vertexai.preview import reasoning_engines
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from dotenv import load_dotenv
import os

load_dotenv()

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

model_kwargs = {
    # temperature (float): The sampling temperature controls the degree of
    # randomness in token selection.
    "temperature": 0.28,
    # max_output_tokens (int): The token limit determines the maximum amount of
    # text output from one prompt.
    "max_output_tokens": 1000,
    # top_p (float): Tokens are selected from most probable to least until
    # the sum of their probabilities equals the top-p value.
    "top_p": 0.95,
    # top_k (int): The next token is selected from among the top-k most
    # probable tokens. This is not supported by all model versions. See
    # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding#valid_parameter_values
    # for details.
    "top_k": None,
    # safety_settings (Dict[HarmCategory, HarmBlockThreshold]): The safety
    # settings to use for generating content.
    # (you must create your safety settings using the previous step first).
    "safety_settings": safety_settings,
}

vertexai.init(
    project=os.getenv("PROJECT_ID"),
    location="us-central1",
    staging_bucket=f"gs://{os.getenv('BACKET_NAME')}",
)


def get_place(
    query: str,
):
    """
    Get place information from the backend.

    Args:
        query (str): The query to search for.
    """
    response = requests.get(
        os.getenv("PLACE_API"),
        params={"query": query},
    )
    return response.json()


def get_session_history(session_id: str):
    from langchain_google_firestore import FirestoreChatMessageHistory
    from google.cloud import firestore

    client = firestore.Client(project=os.getenv("PROJECT_ID"))
    return FirestoreChatMessageHistory(
        client=client,
        session_id=session_id,
        collection="chathistory",
        encode_message=False,
    )


custom_prompt_template = {
    "user_input": lambda x: x["input"],
    "history": lambda x: x["history"],
    "agent_scratchpad": lambda x: format_to_tool_messages(x["intermediate_steps"]),
} | ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたは観光案内AIとして振る舞います。ユーザーからの旅行プランに関する要望を受け付け、以下の点に注意しながら回答してください。"
            "1.ユーザーの興味を優先:旅行先や興味のカテゴリをもとに、適切な観光スポットや体験を提案してください。"
            "2.口コミ情報を活用:口コミや評価を確認したい場合、get_place 関数を使用してください。得られた口コミ情報や評価をユーザーに共有し、スポット選定の参考になるようにしてください。"
            "3.旅程を提案する際の形式:スポット名、そこで行うこと（アクティビティ）、おすすめ理由、訪問日などをわかりやすく提示してください。"
            "4.してはいけないこと:個人情報（住所や電話番号など）を直接聞き出す行為や、法的・倫理的に問題がある行為を助長する提案は行わないでください。公序良俗に反する内容の回答は避けてください。確認できない情報を推測で回答しないでください。"
            "5.トーン・スタイル:丁寧でフレンドリーな言葉遣いを心がけ、必要に応じて専門用語は噛み砕いて説明してください。もし答えが不明な場合は「現時点では不明」と伝え、追加情報を求めてください。"
            "6.目的:ユーザーが旅行計画をスムーズに立てられるようサポートすることが最優先です。必要に応じてget_placeなどの関数を活用し、情報を正確に提示してください。",
        ),
        ("placeholder", "{history}"),
        ("user", "{user_input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


agent = reasoning_engines.LangchainAgent(
    model="gemini-1.5-pro-002",  # Required.
    prompt=custom_prompt_template,  # Optional.
    tools=[get_place],  # Optional.
    model_kwargs=model_kwargs,  # Optional.
    chat_history=get_session_history,
)

response = agent.query(
    input="""
旅行プランを考えてください。
---
旅行エリア
- 仙台

興味・目的
- グルメ、観光スポットめぐり

日程や滞在時間
- 1/28-1/31

移動手段
- 新幹線、旅行中は電車・バス・徒歩

他に希望条件や特記事項があれば
- できるだけ混雑を避けたい
- 主要な観光スポットは抑えたい
- すぐ疲れるのであまり多くのスポットは回れない
- 美味しいものは食べたい
""",
    config={"configurable": {"session_id": "1"}},
)

print(response)
