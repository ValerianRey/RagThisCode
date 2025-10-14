import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


def print_messages(response):
    messages = response["messages"]

    for message in messages:

        if isinstance(message, HumanMessage):
            emoji = "👤"
            content = message.content
        elif isinstance(message, AIMessage):
            emoji = "🤖"
            if message.tool_calls:
                content = message.tool_calls[0]["args"]
            else:
                content = message.content
        elif isinstance(message, ToolMessage):
            emoji = "🔧"
            content = json.loads(message.content)
            content = f"\n{'=' * 100}\n".join(content)
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        print(f"{emoji} {content}")


async def main():

    client = MultiServerMCPClient(
        {
            "similarity_search": {
                "url": "http://51.77.212.235:9000/mcp",
                "transport": "streamable_http",
            },
            "add_repo_to_vector_store": {
                "url": "http://51.77.212.235:9000/mcp",
                "transport": "streamable_http",
            },
            "delete_repo_from_vector_store": {
                "url": "http://51.77.212.235:9000/mcp",
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
    response = await agent.ainvoke(
        {
            # "messages": "is there any java code in torchjd?"
            # "messages": "can you remove TorchJD/torchjd from the vector store?"
            "messages": "can you query the vector store for 'class UPGrad('"
        }
    )
    print_messages(response)


if __name__ == "__main__":
    asyncio.run(main())
