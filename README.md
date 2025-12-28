# mcp-server-qdrant — Notes-focused fork for storing and retrieving notes

This is a forked version of the original repository, adapted specifically for storing, managing and retrieving
structured notes in Qdrant. It exposes a small set of MCP tools optimized for note workflows (add/update/delete/search).

The Model Context Protocol (MCP) enables LLMs to call external tools to fetch or store contextual information. This
server uses Qdrant as the backing vector store.

## Overview

This fork provides a compact MCP server that focuses on structured note storage and retrieval backed by Qdrant.

## Components

### Tools (registered)

The server registers the following MCP tools (note-focused):

- `qdrant-search-notes`
  - Purpose: Hybrid search (semantic + keyword) over notes.
  - Inputs:
    - `query` (string): Query to search for.
    - `collection_name` (string): Collection to search in (optional if default collection is configured).
    - `fusion_method` (string, default: `rrf`): `rrf` or `dbsf` fusion method.
    - `dense_limit` (int, default: 20): Max results from dense (semantic) search.
    - `sparse_limit` (int, default: 20): Max results from sparse (keyword) search.
    - `final_limit` (int, default: 10): Final number of results after fusion.
    - `query_filter` (optional): Qdrant filter object for restricting results.
  - Returns: List of matched entries formatted as strings.

- `qdrant-add-note`
  - Purpose: Add a structured note to the default collection.
  - Inputs:
    - `text` (string): Primary note content.
    - `context` (string): When/why/how the note is useful.
    - `type` (string): One of `cli`, `api`, `learning`, `snippet`, `pattern`.
    - `created_at` (string): ISO-8601 timestamp for the note.
    - `tool` (string, optional): Tool or command name.
    - `tags` (list[string], optional): Tags for categorization.
    - `language` (string, optional): Programming language if applicable.
    - `source` (string, optional): Source or reference URL.
  - Returns: Confirmation message.

- `qdrant-update-note`
  - Purpose: Update an existing note by `note_id` in the default collection.
  - Inputs: `note_id` (string) plus the same metadata fields as `qdrant-add-note`.
  - Returns: Confirmation message.

- `qdrant-delete-note`
  - Purpose: Delete a note by `note_id` from the default collection.
  - Inputs: `note_id` (string).
  - Returns: Confirmation message.

Note: The original `qdrant-store` / `qdrant-find` tools are not registered in this fork; the above tools are the
actual registered interfaces. Read-only mode and collection defaults affect which parameters are required.

## Environment Variables

The server is configured using environment variables. Important variables:

| Name                | Description                                                    | Default / Notes |
|---------------------|----------------------------------------------------------------|-----------------|
| `QDRANT_URL`        | URL of the Qdrant server                                       | None            |
| `QDRANT_API_KEY`    | API key for the Qdrant server                                  | None            |
| `COLLECTION_NAME`   | Default collection name to use                                 | None            |
| `QDRANT_LOCAL_PATH` | Path to local Qdrant database (alternative to `QDRANT_URL`)    | None            |
| `EMBEDDING_PROVIDER`| Embedding provider (e.g. `fastembed`)                          | `fastembed`     |
| `EMBEDDING_MODEL`   | Embedding model name                                            | `sentence-transformers/all-MiniLM-L6-v2` |
| `TOOL_HYBRID_FIND_DESCRIPTION` | Custom description for `qdrant-search-notes` tool   | See `settings.py` |
| `TOOL_ADD_NOTE_DESCRIPTION`    | Custom description for `qdrant-add-note` tool             | See `settings.py` |

Do not supply both `QDRANT_URL` and `QDRANT_LOCAL_PATH` at the same time.

> IMPORTANT: Command-line arguments are not supported — use environment variables for configuration.

### FastMCP environment variables

The server uses FastMCP; common FastMCP environment variables are supported (e.g. `FASTMCP_HOST`, `FASTMCP_PORT`,
`FASTMCP_LOG_LEVEL`, etc.).

## Installation and running

Using `uvx` (recommended for running as a tool):

```bash
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="my-collection" \
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2" \
uvx mcp-server-qdrant
```

Transport protocols supported: `stdio` (default), `sse`, and `streamable-http` — choose via the `--transport` flag.

Run in development mode with the MCP inspector:

```bash
COLLECTION_NAME=mcp-dev fastmcp dev src/mcp_server_qdrant/server.py
```

## Development notes

- The server registers `qdrant-search-notes`, and — when not in read-only mode — `qdrant-add-note`,
  `qdrant-update-note`, and `qdrant-delete-note`.
- Tool descriptions and defaults can be adjusted via environment variables and `settings.py`.

## License

Apache License 2.0 — see the `LICENSE` file for details.


### Using with Cursor/Windsurf

You can configure this MCP server to work as a code search tool for Cursor or Windsurf by customizing the tool
descriptions:

```bash
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="code-snippets" \
TOOL_STORE_DESCRIPTION="Store reusable code snippets for later retrieval. \
The 'information' parameter should contain a natural language description of what the code does, \
while the actual code should be included in the 'metadata' parameter as a 'code' property. \
The value of 'metadata' is a Python dictionary with strings as keys. \
Use this whenever you generate some code snippet." \
TOOL_FIND_DESCRIPTION="Search for relevant code snippets based on natural language descriptions. \
The 'query' parameter should describe what you're looking for, \
and the tool will return the most relevant code snippets. \
Use this when you need to find existing code snippets for reuse or reference." \
uvx mcp-server-qdrant --transport sse # Enable SSE transport
```

In Cursor/Windsurf, you can then configure the MCP server in your settings by pointing to this running server using
SSE transport protocol. The description on how to add an MCP server to Cursor can be found in the [Cursor
documentation](https://docs.cursor.com/context/model-context-protocol#adding-an-mcp-server-to-cursor). If you are
running Cursor/Windsurf locally, you can use the following URL:

```
http://localhost:8000/sse
```

> [!TIP]
> We suggest SSE transport as a preferred way to connect Cursor/Windsurf to the MCP server, as it can support remote
> connections. That makes it easy to share the server with your team or use it in a cloud environment.

This configuration transforms the Qdrant MCP server into a specialized code search tool that can:

1. Store code snippets, documentation, and implementation details
2. Retrieve relevant code examples based on semantic search
3. Help developers find specific implementations or usage patterns

You can populate the database by storing natural language descriptions of code snippets (in the `information` parameter)
along with the actual code (in the `metadata.code` property), and then search for them using natural language queries
that describe what you're looking for.

> [!NOTE]
> The tool descriptions provided above are examples and may need to be customized for your specific use case. Consider
> adjusting the descriptions to better match your team's workflow and the specific types of code snippets you want to
> store and retrieve.

**If you have successfully installed the `mcp-server-qdrant`, but still can't get it to work with Cursor, please
consider creating the [Cursor rules](https://docs.cursor.com/context/rules-for-ai) so the MCP tools are always used when
the agent produces a new code snippet.** You can restrict the rules to only work for certain file types, to avoid using
the MCP server for the documentation or other types of content.

### Using with Claude Code

You can enhance Claude Code's capabilities by connecting it to this MCP server, enabling semantic search over your
existing codebase.

#### Setting up mcp-server-qdrant

1. Add the MCP server to Claude Code:

    ```shell
    # Add mcp-server-qdrant configured for code search
    claude mcp add code-search \
    -e QDRANT_URL="http://localhost:6333" \
    -e COLLECTION_NAME="code-repository" \
    -e EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2" \
    -e TOOL_STORE_DESCRIPTION="Store code snippets with descriptions. The 'information' parameter should contain a natural language description of what the code does, while the actual code should be included in the 'metadata' parameter as a 'code' property." \
    -e TOOL_FIND_DESCRIPTION="Search for relevant code snippets using natural language. The 'query' parameter should describe the functionality you're looking for." \
    -- uvx mcp-server-qdrant
    ```

2. Verify the server was added:

    ```shell
    claude mcp list
    ```

#### Using Semantic Code Search in Claude Code

Tool descriptions, specified in `TOOL_STORE_DESCRIPTION` and `TOOL_FIND_DESCRIPTION`, guide Claude Code on how to use
the MCP server. The ones provided above are examples and may need to be customized for your specific use case. However,
Claude Code should be already able to:

1. Use the `qdrant-store` tool to store code snippets with descriptions.
2. Use the `qdrant-find` tool to search for relevant code snippets using natural language.

### Run MCP server in Development Mode

The MCP server can be run in development mode using the `mcp dev` command. This will start the server and open the MCP
inspector in your browser.

```shell
COLLECTION_NAME=mcp-dev fastmcp dev src/mcp_server_qdrant/server.py
```

### Using with VS Code

For one-click installation, click one of the install buttons below:

[![Install with UVX in VS Code](https://img.shields.io/badge/VS_Code-UVX-0098FF?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=qdrant&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22mcp-server-qdrant%22%5D%2C%22env%22%3A%7B%22QDRANT_URL%22%3A%22%24%7Binput%3AqdrantUrl%7D%22%2C%22QDRANT_API_KEY%22%3A%22%24%7Binput%3AqdrantApiKey%7D%22%2C%22COLLECTION_NAME%22%3A%22%24%7Binput%3AcollectionName%7D%22%7D%7D&inputs=%5B%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantUrl%22%2C%22description%22%3A%22Qdrant+URL%22%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantApiKey%22%2C%22description%22%3A%22Qdrant+API+Key%22%2C%22password%22%3Atrue%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22collectionName%22%2C%22description%22%3A%22Collection+Name%22%7D%5D) [![Install with UVX in VS Code Insiders](https://img.shields.io/badge/VS_Code_Insiders-UVX-24bfa5?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=qdrant&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22mcp-server-qdrant%22%5D%2C%22env%22%3A%7B%22QDRANT_URL%22%3A%22%24%7Binput%3AqdrantUrl%7D%22%2C%22QDRANT_API_KEY%22%3A%22%24%7Binput%3AqdrantApiKey%7D%22%2C%22COLLECTION_NAME%22%3A%22%24%7Binput%3AcollectionName%7D%22%7D%7D&inputs=%5B%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantUrl%22%2C%22description%22%3A%22Qdrant+URL%22%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantApiKey%22%2C%22description%22%3A%22Qdrant+API+Key%22%2C%22password%22%3Atrue%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22collectionName%22%2C%22description%22%3A%22Collection+Name%22%7D%5D&quality=insiders)

[![Install with Docker in VS Code](https://img.shields.io/badge/VS_Code-Docker-0098FF?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=qdrant&config=%7B%22command%22%3A%22docker%22%2C%22args%22%3A%5B%22run%22%2C%22-p%22%2C%228000%3A8000%22%2C%22-i%22%2C%22--rm%22%2C%22-e%22%2C%22QDRANT_URL%22%2C%22-e%22%2C%22QDRANT_API_KEY%22%2C%22-e%22%2C%22COLLECTION_NAME%22%2C%22mcp-server-qdrant%22%5D%2C%22env%22%3A%7B%22QDRANT_URL%22%3A%22%24%7Binput%3AqdrantUrl%7D%22%2C%22QDRANT_API_KEY%22%3A%22%24%7Binput%3AqdrantApiKey%7D%22%2C%22COLLECTION_NAME%22%3A%22%24%7Binput%3AcollectionName%7D%22%7D%7D&inputs=%5B%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantUrl%22%2C%22description%22%3A%22Qdrant+URL%22%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantApiKey%22%2C%22description%22%3A%22Qdrant+API+Key%22%2C%22password%22%3Atrue%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22collectionName%22%2C%22description%22%3A%22Collection+Name%22%7D%5D) [![Install with Docker in VS Code Insiders](https://img.shields.io/badge/VS_Code_Insiders-Docker-24bfa5?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=qdrant&config=%7B%22command%22%3A%22docker%22%2C%22args%22%3A%5B%22run%22%2C%22-p%22%2C%228000%3A8000%22%2C%22-i%22%2C%22--rm%22%2C%22-e%22%2C%22QDRANT_URL%22%2C%22-e%22%2C%22QDRANT_API_KEY%22%2C%22-e%22%2C%22COLLECTION_NAME%22%2C%22mcp-server-qdrant%22%5D%2C%22env%22%3A%7B%22QDRANT_URL%22%3A%22%24%7Binput%3AqdrantUrl%7D%22%2C%22QDRANT_API_KEY%22%3A%22%24%7Binput%3AqdrantApiKey%7D%22%2C%22COLLECTION_NAME%22%3A%22%24%7Binput%3AcollectionName%7D%22%7D%7D&inputs=%5B%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantUrl%22%2C%22description%22%3A%22Qdrant+URL%22%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantApiKey%22%2C%22description%22%3A%22Qdrant+API+Key%22%2C%22password%22%3Atrue%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22collectionName%22%2C%22description%22%3A%22Collection+Name%22%7D%5D&quality=insiders)

#### Manual Installation

Add the following JSON block to your User Settings (JSON) file in VS Code. You can do this by pressing `Ctrl + Shift + P` and typing `Preferences: Open User Settings (JSON)`.

```json
{
  "mcp": {
    "inputs": [
      {
        "type": "promptString",
        "id": "qdrantUrl",
        "description": "Qdrant URL"
      },
      {
        "type": "promptString",
        "id": "qdrantApiKey",
        "description": "Qdrant API Key",
        "password": true
      },
      {
        "type": "promptString",
        "id": "collectionName",
        "description": "Collection Name"
      }
    ],
    "servers": {
      "qdrant": {
        "command": "uvx",
        "args": ["mcp-server-qdrant"],
        "env": {
          "QDRANT_URL": "${input:qdrantUrl}",
          "QDRANT_API_KEY": "${input:qdrantApiKey}",
          "COLLECTION_NAME": "${input:collectionName}"
        }
      }
    }
  }
}
```

Or if you prefer using Docker, add this configuration instead:

```json
{
  "mcp": {
    "inputs": [
      {
        "type": "promptString",
        "id": "qdrantUrl",
        "description": "Qdrant URL"
      },
      {
        "type": "promptString",
        "id": "qdrantApiKey",
        "description": "Qdrant API Key",
        "password": true
      },
      {
        "type": "promptString",
        "id": "collectionName",
        "description": "Collection Name"
      }
    ],
    "servers": {
      "qdrant": {
        "command": "docker",
        "args": [
          "run",
          "-p", "8000:8000",
          "-i",
          "--rm",
          "-e", "QDRANT_URL",
          "-e", "QDRANT_API_KEY",
          "-e", "COLLECTION_NAME",
          "mcp-server-qdrant"
        ],
        "env": {
          "QDRANT_URL": "${input:qdrantUrl}",
          "QDRANT_API_KEY": "${input:qdrantApiKey}",
          "COLLECTION_NAME": "${input:collectionName}"
        }
      }
    }
  }
}
```

Alternatively, you can create a `.vscode/mcp.json` file in your workspace with the following content:

```json
{
  "inputs": [
    {
      "type": "promptString",
      "id": "qdrantUrl",
      "description": "Qdrant URL"
    },
    {
      "type": "promptString",
      "id": "qdrantApiKey",
      "description": "Qdrant API Key",
      "password": true
    },
    {
      "type": "promptString",
      "id": "collectionName",
      "description": "Collection Name"
    }
  ],
  "servers": {
    "qdrant": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL": "${input:qdrantUrl}",
        "QDRANT_API_KEY": "${input:qdrantApiKey}",
        "COLLECTION_NAME": "${input:collectionName}"
      }
    }
  }
}
```

For workspace configuration with Docker, use this in `.vscode/mcp.json`:

```json
{
  "inputs": [
    {
      "type": "promptString",
      "id": "qdrantUrl",
      "description": "Qdrant URL"
    },
    {
      "type": "promptString",
      "id": "qdrantApiKey",
      "description": "Qdrant API Key",
      "password": true
    },
    {
      "type": "promptString",
      "id": "collectionName",
      "description": "Collection Name"
    }
  ],
  "servers": {
    "qdrant": {
      "command": "docker",
      "args": [
        "run",
        "-p", "8000:8000",
        "-i",
        "--rm",
        "-e", "QDRANT_URL",
        "-e", "QDRANT_API_KEY",
        "-e", "COLLECTION_NAME",
        "mcp-server-qdrant"
      ],
      "env": {
        "QDRANT_URL": "${input:qdrantUrl}",
        "QDRANT_API_KEY": "${input:qdrantApiKey}",
        "COLLECTION_NAME": "${input:collectionName}"
      }
    }
  }
}
```

## Contributing

If you have suggestions for how mcp-server-qdrant could be improved, or want to report a bug, open an issue!
We'd love all and any contributions.

### Testing `mcp-server-qdrant` locally

The [MCP inspector](https://github.com/modelcontextprotocol/inspector) is a developer tool for testing and debugging MCP
servers. It runs both a client UI (default port 5173) and an MCP proxy server (default port 3000). Open the client UI in
your browser to use the inspector.

```shell
QDRANT_URL=":memory:" COLLECTION_NAME="test" \
fastmcp dev src/mcp_server_qdrant/server.py
```

Once started, open your browser to http://localhost:5173 to access the inspector interface.

## License

This MCP server is licensed under the Apache License 2.0. This means you are free to use, modify, and distribute the
software, subject to the terms and conditions of the Apache License 2.0. For more details, please see the LICENSE file
in the project repository.
