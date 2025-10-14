from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


app = FastAPI()

allowed_origins = [
    "http://localhost:7070",
    "http://127.0.0.1:7070",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files to serve the frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# Serve the main HTML file at the root
@app.get("/")
async def read_index():
    return FileResponse("frontend/index.html")


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


@app.post("/chat_stream")
async def chat_stream(body: ChatRequest) -> StreamingResponse:

    async def event_generator():
        server_ip = "51.77.212.235"
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

        saw_chunks = False

        async for event in agent.astream_events({"messages": body.message}, version="v1"):
            event_name = event.get("event")
            data = event.get("data")

            if event_name == "on_chat_model_stream" and data is not None:
                chunk = data.get("chunk")
                if chunk is not None:
                    content = getattr(chunk, "content", None)
                    if isinstance(content, list):
                        text_parts = []
                        for part in content:
                            text = getattr(part, "text", None)
                            if isinstance(text, str):
                                text_parts.append(text)
                        if text_parts:
                            saw_chunks = True
                            yield "".join(text_parts)
                    elif isinstance(content, str):
                        saw_chunks = True
                        yield content

            elif event_name == "on_llm_stream" and data is not None:
                chunk_text = data.get("chunk", "")
                if isinstance(chunk_text, str) and chunk_text:
                    saw_chunks = True
                    yield chunk_text

            elif event_name == "on_chain_end" and data is not None:
                if not saw_chunks:
                    output = data.get("output")
                    messages = (
                        output["messages"]
                        if isinstance(output, dict) and "messages" in output
                        else None
                    )
                    final_text = ""
                    if isinstance(messages, list):
                        for msg in reversed(messages):
                            if isinstance(msg, AIMessage):
                                final_text = (
                                    msg.content
                                    if isinstance(msg.content, str)
                                    else str(msg.content)
                                )
                                break
                    if final_text:
                        yield final_text

        yield "\n[DONE]"

    return StreamingResponse(event_generator(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=app, host="0.0.0.0", port=7070)
