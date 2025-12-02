# RagThisCode
Setup an MCP server to do RAG over any public GitHub repo or any of your private repos

# Demos powered by RagThisCode

* https://chatwithtorchjd.org Here, the multi-objective optimization library [TorchJD](https://github.com/TorchJD/torchjd) is pre-fetched. You can directly talk with its code.

# Installation
Install uv: https://docs.astral.sh/uv/getting-started/installation/

Create a github access token at: https://github.com/settings/personal-access-tokens. You can make it
never expire and give it access to only public repositories (unless you want to do RAG on your
private repos as well). It seems that it only works if you use Tokens (classic) and select the scope public_repo.

Add:
```
export GITHUB_ACCESS_TOKEN="ghp_....."
```
to your `.bashrc` or `.zshrc`.

Create an OpenAI API key: https://platform.openai.com/docs/overview

Add:
```
export OPENAI_API_KEY="sk-proj-....."
```
to your `.bashrc` or `.zshrc`.

Restart your terminal for your exports to be effective.

# Running

To fetch code from GitHub, run:
```
uv run store_code_in_vector_db.py
```

To start the MCP server, run:
```
uv run server.py
```

To start the chat client that will connect to the MCP server that you just started, run:
```
uv run chat.py
```

To start the proxy
```
uv run proxy.py
```

To start the client
```
python3 -m http.server 5173 --bind 127.0.0.1 --directory "./frontend"
```

# Contributing

Run:
```
uv run pre-commit install
```


# Running in Docker

```
docker build -t ragthiscode:latest .
```

```
docker run -p 7070:7070 -p 9000:9000  -e OPENAI_API_KEY=$OPENAI_API_KEY   -e GITHUB_ACCESS_TOKEN=$GITHUB_ACCESS_TOKEN -v $PWD/data/chroma_langchain_db:/app/data/chroma_langchain_db  --cpus="1.0" --memory="2g" -d ragthiscode
```
