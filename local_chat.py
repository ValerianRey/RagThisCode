import asyncio
import json
import threading

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel

from common import get_vector_store
from server import run_mcp_server
from store_code_in_vector_db import add_repo_to_vector_store


class ChatRequest(BaseModel):
    message: str


app = FastAPI()

server_ip = "localhost"  # "0.0.0.0" when exposed to the internet, localhost when not
proxy_ip = "localhost"  # "0.0.0.0" when exposed to the internet, localhost when not
proxy_port = 7070
server_port = 9000
allowed_origins = [
    f"http://localhost:{proxy_port}",
    f"http://127.0.0.1:{proxy_port}",
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
# Mount assets directory
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


# Serve the main HTML file at the root
@app.get("/")
async def read_index():
    return FileResponse("frontend/index.html")


@app.get("/{repo_name:path}")
async def add_repo_page(repo_name: str):
    """Display loading page for repository ingestion"""
    with open("frontend/repo_loading.html", "r") as f:
        html_content = f.read()

    html_content = html_content.replace("{repo_name}", repo_name)
    html_content = html_content.replace("{repo_name_json}", json.dumps(repo_name))

    return HTMLResponse(content=html_content)


@app.post("/api/add_repo/{repo_name:path}")
async def add_repo_api(repo_name: str):
    """API endpoint to add a repository to the vector store"""

    try:
        server_status = requests.get(f"http://{server_ip}:{server_port}/health")
        if server_status.status_code != 404:
            raise HTTPException(
                status_code=500,
                detail="MCP server is already running, make sure you do not have multiple tabs open.",
            )
    except requests.exceptions.ConnectionError:
        # If the server is not running, we can continue
        pass

    try:
        vector_store = get_vector_store()
        await asyncio.to_thread(
            add_repo_to_vector_store,
            repo_name=repo_name,
            vector_store=vector_store,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding repo to vector store: {str(e)}")

    try:
        server_thread = threading.Thread(
            target=run_mcp_server,
            daemon=True,
            args=(repo_name, vector_store, server_ip, server_port),
        )
        server_thread.start()
        # Give the server a moment to start up
        await asyncio.sleep(1.0)

        health_response = requests.get(f"http://{server_ip}:{server_port}/health")
        if health_response.status_code != 200:
            raise HTTPException(
                status_code=500, detail=f"MCP server is not healthy: {health_response.text}"
            )
        if health_response.json()["repo_name"] != repo_name:
            raise HTTPException(
                status_code=500,
                detail="MCP server is not serving the correct repo, make sure you do not have multiple tabs open.",
            )

        return JSONResponse(
            content={
                "status": "success",
                "message": f"Successfully added {repo_name} to vector store and started MCP server.",
            },
            status_code=200,
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting MCP server: {str(e)}")


@app.post("/chat_stream")
async def chat_stream(body: ChatRequest) -> StreamingResponse:

    async def event_generator():
        client = MultiServerMCPClient(
            {
                "similarity_search": {
                    "url": f"http://{server_ip}:{server_port}/mcp",
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

        agent = create_agent(
            model="openai:gpt-4.1",
            system_prompt=SYSTEM_PROMPT,
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

    uvicorn.run(app=app, host=proxy_ip, port=proxy_port)
