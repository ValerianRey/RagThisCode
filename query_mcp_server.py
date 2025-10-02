import asyncio

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent


def print_messages(response):
    messages = response["messages"]

    for message in messages:


        if isinstance(message, HumanMessage):
            emoji = "👤"
        elif isinstance(message, AIMessage):
            emoji = "🤖"
        elif isinstance(message, ToolMessage):
            emoji = "🔧"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        print(f"{emoji} {message.content}")


async def main():

    client = MultiServerMCPClient(
        {
            "similarity_search": {
                "url": "http://localhost:9000/mcp",
                "transport": "streamable_http",
            }
        }
    )

    tools = await client.get_tools()

    agent = create_react_agent(model="openai:gpt-4.1", tools=tools)
    response = await agent.ainvoke(
        {"messages": "can you find the implementation of UPGrad?"}
    )
    print_messages(response)


if __name__ == "__main__":
    asyncio.run(main())
