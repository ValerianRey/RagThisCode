from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat(body: ChatRequest) -> dict[str, Any]:

    server_ip = "51.77.212.235"  # localhost for local testing
    client = MultiServerMCPClient(
        {
            "similarity_search": {
                "url": f"http://{server_ip}:9000/mcp",
                "transport": "streamable_http",
            },
            "add_repo_to_vector_store": {
                "url": f"http://{server_ip}:9000/mcp",
                "transport": "streamable_http",
            },
            "delete_repo_from_vector_store": {
                "url": f"http://{server_ip}:9000/mcp",
                "transport": "streamable_http",
            },
        }
    )

    tools = await client.get_tools()

    SYSTEM_PROMPT = """
    You are a helpful assistant that is specifically designed to answer questions about a codebase, the content is stored in a vector store.
    Please only answer questions about the codebase by running the similarity_search tool.
    If necessary, you can call the tool multiple times to get the information you need.
    Also use markdown to format your responses.
    """

    agent = create_react_agent(
        model="openai:gpt-4.1",
        prompt=SYSTEM_PROMPT,
        tools=tools,
    )

    response = await agent.ainvoke({"messages": body.message})

    messages = response["messages"]
    final_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            final_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    return {"final": final_text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=app, host="127.0.0.1", port=7070)
