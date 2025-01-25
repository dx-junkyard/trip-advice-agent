import vertexai
from vertexai.preview import reasoning_engines
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
import requests
from dotenv import load_dotenv
import os
from langchain_core.tools import tool

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
    "max_output_tokens": 8192,
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

@tool
def get_place(
    query: str,
):
    """
    Get place information from the backend.
    Returns detailed information, including its name, address, types, reviews, average rating, and photo URL.

    Args:
        query (str): The query to search. e.g. restaurant name, spot name, etc.

    Returns:
        A list of dictionaries, where each dictionary represents a station with the following keys:
        - 'name' (dict): Contains 'text' (station name) and 'languageCode' (language code).
        - 'address' (str): The address of the station.
        - 'types' (list of str): A list of station types (e.g., 'transit_station', 'train_station').
        - 'reviews' (list of dict): A list of top 5 reviews, each containing:
            - 'rating' (int): Rating out of 5.
            - 'text' (dict): A dictionary with 'text' containing the review content.
        - 'userReviewCount' (int): The number of user reviews.
        - 'rating' (float): The average rating of the station.
        - 'photoUri' (str): URL of a photo representing the station.
        - 'businessHour' (str): The business status of the station.
        - 'website' (str): URL of the station's website.
    """
    response = requests.get(
        os.getenv("PLACE_API"),
        params={"query": query},
    )
    return response.json()



def langgraph_builder(*, model, **kwargs):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langgraph.graph import END, MessageGraph
    from langgraph.prebuilt import ToolNode
    from langchain_core.messages import BaseMessage, HumanMessage
    from typing import Literal


    def router(state: list[BaseMessage]) -> Literal["get_place", "summary"]:
        """Initiates product details retrieval if the user asks for a product."""
        # Get the tool_calls from the last message in the conversation history.
        tool_calls = state[-1].tool_calls
        # If there are any tool_calls
        if len(tool_calls):
            # Return the name of the tool to be called
            return "get_place"
        else:
            # End the conversation flow.
            return "summary"

    output_parser = StrOutputParser()

    model_with_tools = model.bind_tools(tools=[get_place])

    planner = ChatPromptTemplate.from_template(
        "あなたは観光案内AIとして振る舞います。ユーザーからの旅行プランに関する要望を受け付け、以下の点に注意しながら回答してください。"
        "- ユーザーの興味を優先:旅行先や興味のカテゴリをもとに、適切な観光スポットや体験を提案してください。"
        "- 一日ごとにテーマを設定:1日の観光プランを提案する際には、その日のテーマ（例:グルメ、観光スポット巡り、アクティビティ）を設定してください。"
        "- 長期旅行の場合:3日以上の長期旅行の場合は、数日ごとにまとめて提案文を記載してください。"
        "- 旅程を提案する際の形式:スポット名、そこで行うこと（アクティビティ）、おすすめ理由、訪問日などをわかりやすく提示してください。"
        "- してはいけないこと:個人情報（住所や電話番号など）を直接聞き出す行為や、法的・倫理的に問題がある行為を助長する提案は行わないでください。公序良俗に反する内容の回答は避けてください。確認できない情報を推測で回答しないでください。Googleの検索URLは貼らないでください"
        "- トーン・スタイル:丁寧でフレンドリーな言葉遣いを心がけ、必要に応じて専門用語は噛み砕いて説明してください。もし答えが不明な場合は「現時点では不明」と伝え、追加情報を求めてください。"
        "- 目的:ユーザーが旅行計画をスムーズに立てられるようサポートすることが最優先です。"
        "ユーザからの入力:{input}",
    ) | model | output_parser

    tools = ChatPromptTemplate.from_template(
        "get_place関数を使用して、プラン内のすべての観光スポットに関する情報やレビューを取得してください。"
        "もし観光スポットが特定の施設や店舗でない場合は、get_place関数を使用して周りの観光スポットを取得し、その情報から適切なスポットを提案してください。"
        "ユーザからの入力:{input}",
    ) | model_with_tools

    summary = ChatPromptTemplate.from_template(
        "これまでの情報をもとに、旅行プランの提案文をまとめてください。提案文には各スポットのレビューはスポットごとに1つ記載してください"
        "提案文は各日付に対して以下のフォーマットで記載してください。ただし3日以上の長期旅行の場合は、数日ごとにまとめて提案文を記載してください。"
        "営業時間は日付に合わせて記載してください。もし定休日だったり休業中だったりする場合はそのスポットは提案しないでください。"
        "末尾にはこの旅行の全体の説明や注意事項を記載してください。"
        "クチコミの文が長い場合は、適宜省略してください。"
        "この提案は一度しか行わないため、提案文をよく検討してから提出してください。"
        """
    旅行プランの先頭には以下の情報を記載してください。
    # {{planName}}
    - 全体のテーマ　：{{theme}}
    - エリア名　　　：{{area}}

    各日程には以下の情報を記載し、最終日まで続けてください。
    ### 日付: {{date}}(短期旅行の場合) {{startDate}} - {{endDate}}(長期旅行の場合)
    - テーマ　：{{theme}}
    - 旅行エリア　　：{{area}}
    ────────────────────────────────

    #### スポット名　　：{{spotName}}
    - アクティビティ：{{activity}}
    - おすすめ理由　：{{reason}}
    - クチコミ評価　：{{rating}} ({{ratingCount}}件)
    - クチコミの声　： 
        - {{review1}}
        - {{review2}}
    - 営業時間　　　：{{businessHours}}
    - 所要時間(目安)：{{duration}}
    - Webサイト    ：[spotName]({{websiteUrl}})


    ![spotNameの写真](photoUri)

    """
    "ユーザからの入力:{input}",
    ) | model | output_parser

    builder = MessageGraph()
    builder.add_node("planner", planner)
    builder.add_node("tools", tools)
    builder.add_node("get_place", ToolNode([get_place]))
    builder.add_node("summary", summary)

    builder.add_edge("planner", "tools")
    # builder.add_conditional_edges("tools", router)
    builder.add_edge("tools", "get_place")
    # builder.add_edge("get_place", "planner")
    builder.add_edge("get_place", "summary")
    builder.add_edge("summary", END)
    builder.set_entry_point("planner")
    graph = builder.compile()
    print(graph.get_graph().draw_mermaid())
    return graph

agent = reasoning_engines.LangchainAgent(
    model="gemini-1.5-pro-002",
    # model="gemini-1.5-flash",
    runnable_builder=langgraph_builder,
    model_kwargs=model_kwargs,
)

# 短期旅行の場合
response = agent.query(
    input={
        "role": "user", 
        "content": """
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
        """},
)

# 長期旅行の場合
# response = agent.query(
#     input={
#         "role": "user", 
#         "content": """
# 旅行プランを考えてください。
# ---
# 旅行エリア
# - イタリア(フィレンツェ・ローマ・その他)

# 興味・目的
# - グルメ、観光スポットめぐり

# 日程や滞在時間
# - 4/25-5/4

# 移動手段
# - 飛行機、旅行中は電車・バス・徒歩

# 他に希望条件や特記事項があれば
# - できるだけ混雑を避けたい
# - 主要な観光スポットは抑えたい
# - すぐ疲れるのであまり多くのスポットは回れない
# - 美味しいものは食べたい
# """},
# )

print(response)

