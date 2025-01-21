import vertexai
from vertexai.preview import reasoning_engines
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
import os
from dotenv import load_dotenv

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

def langgraph_builder(*, model, **kwargs):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langgraph.graph import END, MessageGraph

    output_parser = StrOutputParser()

    planner = ChatPromptTemplate.from_template(
        "Generate an argument about: {input}"
    ) | model | output_parser

    pros = ChatPromptTemplate.from_template(
        "List the positive aspects of {input}"
    ) | model | output_parser

    cons = ChatPromptTemplate.from_template(
        "List the negative aspects of {input}"
    ) | model | output_parser

    summary = ChatPromptTemplate.from_template(
        "Input:{input}\nGenerate a final response given the critique",
    ) | model | output_parser

    builder = MessageGraph()
    builder.add_node("planner", planner)
    builder.add_node("pros", pros)
    builder.add_node("cons", cons)
    builder.add_node("summary", summary)

    builder.add_edge("planner", "pros")
    builder.add_edge("planner", "cons")
    builder.add_edge("pros", "summary")
    builder.add_edge("cons", "summary")
    builder.add_edge("summary", END)
    builder.set_entry_point("planner")
    return builder.compile()

agent = reasoning_engines.LangchainAgent(
    model="gemini-1.5-pro-002",
    runnable_builder=langgraph_builder,
    model_kwargs=model_kwargs,
)

# Example query
response = agent.query(
    input={"role": "user", "content": "scrum methodology"},
)

print(response)
