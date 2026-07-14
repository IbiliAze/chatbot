# chatbot-api

A retrieval-augmented generation (RAG) chatbot library built on LangChain, pgvector, and OpenAI.

## What it does

`ChatBot` wires together an OpenAI-embedded [pgvector](https://github.com/pgvector/pgvector) store, a chat model, and a text splitter into a single class that can:

- **Ingest** raw text or `Document`s, splitting them into overlapping chunks and embedding them into Postgres.
- **Answer questions** grounded only in the ingested knowledge base, with source citations, a confidence rating (`high` / `medium` / `low`), key quotes, and suggested follow-up questions.
- **Retrieve** with either a plain similarity search or an advanced multi-query strategy, where the LLM rewrites the question into several search queries, retrieves each in parallel, and dedupes the results for better recall.
- **Manage conversation history** per session, in memory, with automatic trimming to the last 10 turns.

## Usage

```ts
import { ChatBot } from "./index";

const bot = new ChatBot({
  apiKey: process.env.OPENAI_API_KEY!,
  db: {
    dbHost: "localhost",
    dbPort: "5432",
    dbUsername: "postgres",
    dbPassword: "postgres",
    dbDatabase: "chatbot",
  },
});

await bot.addTexts({ texts: ["Some knowledge base content..."], source: "docs.md" });

const response = await bot.ask("What does the documentation say?");
console.log(response.answer, response.confidence, response.sources);
```

## Setup

```bash
yarn install
```

Requires a Postgres database with the [`pgvector`](https://github.com/pgvector/pgvector) extension enabled (see `init.sql`) and an `OPENAI_API_KEY`.

## Project structure

- `index.ts` — the `ChatBot` class.
- `schemas/` — Zod schemas for structured LLM output (chat responses, multi-query generation).
- `init.sql` — database bootstrap for the vector store.
