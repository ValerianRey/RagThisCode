# RagThisCode
Setup an MCP server to do RAG over any public GitHub repo or any of your private repos

<a href="#" onclick="(function(){const m=window.location.href.match(/github\.com\/([^\/]+)\/([^\/]+)/);if(m){window.location.href='http://127.0.0.1:7070/'+m[1]+'/'+m[2].split('/')[0].split('?')[0].split('#')[0];}else{alert('Could not extract repository information.');}return false;})();" style="display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px; background-color: #238636; color: white; text-decoration: none; border-radius: 6px; font-weight: 500; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif; font-size: 14px; border: 1px solid #2ea043; transition: background-color 0.2s;">
  <img src="assets/logo.png" alt="RagThisCode" style="height: 20px; width: auto; vertical-align: middle;">
  💬 Chat with this code
</a>

## Embed Chat Button in Your README

Add this button to your GitHub repository README to allow users to quickly access the chat interface:

```html
<a href="#" onclick="(function(){const m=window.location.href.match(/github\.com\/([^\/]+)\/([^\/]+)/);if(m){window.location.href='http://127.0.0.1:7070/'+m[1]+'/'+m[2].split('/')[0].split('?')[0].split('#')[0];}else{alert('Could not extract repository information.');}return false;})();" style="display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px; background-color: #238636; color: white; text-decoration: none; border-radius: 6px; font-weight: 500; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif; font-size: 14px; border: 1px solid #2ea043; transition: background-color 0.2s;">
  <img src="assets/logo.png" alt="RagThisCode" style="height: 20px; width: auto; vertical-align: middle;">
  💬 Chat with this code
</a>
```

The button will automatically detect the repository owner and name from the GitHub URL and redirect to `http://127.0.0.1:7070/{owner}/{repo}`.

# Installation
Install uv: https://docs.astral.sh/uv/getting-started/installation/

Create a github access token at: https://github.com/settings/personal-access-tokens. You can make it
never expire and give it access to only public repositories (unless you want to do RAG on your
private repos as well).

Add:
```
export GITHUB_ACCESS_TOKEN="github_pat_....."
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

Behind a proxy
```
docker run -p 127.0.0.1:7070:7070 -p 127.0.0.1:9000:9000  -e OPENAI_API_KEY=$OPENAI_API_KEY   -e GITHUB_ACCESS_TOKEN=$GITHUB_ACCESS_TOKEN -v $PWD/data/chroma_langchain_db:/app/data/chroma_langchain_db  --cpus="1.0" --memory="2g" -d ragthiscode
```

Or otherwise (exposes port without proxy)
```
docker run -p 7070:7070 -p 9000:9000  -e OPENAI_API_KEY=$OPENAI_API_KEY   -e GITHUB_ACCESS_TOKEN=$GITHUB_ACCESS_TOKEN -v $PWD/data/chroma_langchain_db:/app/data/chroma_langchain_db  --cpus="1.0" --memory="2g" -d ragthiscode
```
