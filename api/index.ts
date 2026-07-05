////////////////////////////////////////////////////////////////////////////////??PACKAGES
import { OpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddingModelId, OpenAIChatModelId } from "@langchain/openai";
import { PGVectorStore } from "@langchain/pgvector";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { ChatResponse, ChatResponseSchema } from "./schemas/response.schema";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { MultiQuerySchema } from "./schemas/multiQuery.schema";
////////////////////////////////////////////////////////////////////////////////////////??

export class ChatBot {
  private readonly store: PGVectorStore;
  private readonly llm: ChatOpenAI;
  private readonly splitter: RecursiveCharacterTextSplitter;
  private readonly sessionStore: Map<string, InMemoryChatMessageHistory>;
  private readonly advancedRetriever: boolean = false;

  /**
   * Wires up the RAG stack: the pgvector store (with OpenAI embeddings), the
   * chat model, and the text splitter used when ingesting documents. Nothing
   * touches the database yet — the table is created lazily on first write.
   *
   * @param config Connection details, API key, model overrides, and whether to
   *   enable the multi-query retriever.
   */
  constructor(config: Config) {
    const {
      apiKey,
      advancedRetriever = false,
      openAiModel = "gpt-4o-mini",
      embeddingModel = "text-embedding-3-small",
      db,
    } = config;
    const { dbDatabase, dbHost, dbPassword, dbPort, dbUsername } = db;

    this.advancedRetriever = advancedRetriever;
    this.store = new PGVectorStore(new OpenAIEmbeddings({ apiKey, model: embeddingModel }), {
      postgresConnectionOptions: {
        host: dbHost,
        port: dbPort,
        user: dbUsername,
        password: dbPassword,
        database: dbDatabase,
      },
      tableName: "documents",
      columns: {
        idColumnName: "id",
        contentColumnName: "content",
        vectorColumnName: "embedding",
        metadataColumnName: "metadata",
      },
      distanceStrategy: "cosine",
    });

    this.sessionStore = new Map();
    this.llm = new ChatOpenAI({ apiKey, model: openAiModel, temperature: 0.2 });
    this.splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
      separators: ["\n\n", "\n", ". ", " ", ""],
    });

    console.log("Chatbot initialised");
    console.log(` Vector store: ${this.store.tableName}`);
  }

  /**
   * Answers a question against the ingested knowledge base. Retrieves relevant
   * chunks (multi-query when enabled), grounds the model on that context plus
   * the last 10 turns of history, and returns a structured answer with
   * citations and a confidence rating. The exchange is appended to history.
   *
   * @param question The user's natural-language question.
   * @returns The structured answer; `keyQuotes`/`followUpQuestions` default to [].
   */
  async ask(question: string): Promise<ChatResponse> {
    const history = this.getSessionHistory("new_test");
    const retriever = this.buildRetriever();
    const docs = await retriever.invoke(question);
    const context = this.formatDocsForRetrieve(docs);

    if (this.advancedRetriever) console.log(`Multi query returned: ${docs.length} documents`);

    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are an AI chatbot for an organisation. Answer questions based on the context only.

        Here are the rules:
        1. Only use information from the context below.
        2. If the context doesn't have the answer, or the context is unrelated to the question, say so.
        3. Cite the sources you used.
        4. Rate your confidence: high, medium, or low.`,
      ],
      new MessagesPlaceholder("history"),
      [
        "human",
        `Context documents:
        {context}

        Question:
        {question}

        Provide a clear answer with source citations.`,
      ],
    ]);

    const chain = RunnableSequence.from([prompt, this.llm.withStructuredOutput(ChatResponseSchema)]);
    const response = await chain.invoke({ context, question, history: (await history.getMessages()).slice(-10) });

    history.addMessage(new HumanMessage(question));
    history.addMessage(new AIMessage(response.answer));

    return {
      ...response,
      keyQuotes: response.keyQuotes ?? [],
      followUpQuestions: response.followUpQuestions ?? [],
    };
  }

  /**
   * Drops a conversation's in-memory history and removes it from the session
   * store. No-op (logs only) if the session doesn't exist.
   *
   * @param sessionId The conversation to clear.
   */
  clearSession(sessionId: string) {
    const session = this.sessionStore.get(sessionId);
    if (session) {
      session.clear();
      this.sessionStore.delete(sessionId);
      console.log(`Cleared session: ${sessionId}`);
    } else console.log(`No session to clear under: ${sessionId}`);
  }

  /**
   * Returns a conversation's history as plain `{ role, context }` objects for
   * display. Empty array if the session doesn't exist.
   *
   * @param sessionId The conversation to read.
   */
  async getSessionHistoryDisplay(sessionId: string) {
    const session = this.sessionStore.get(sessionId);

    if (!session) return [];

    return (await session.getMessages()).map((m) => ({
      role: m instanceof HumanMessage ? "human" : "assistant",
      context: m.content,
    }));
  }

  /**
   * Convenience wrapper over {@link addDocuments}: wraps raw strings as
   * `Document`s tagged with the given source, then ingests them.
   *
   * @param texts Raw text blocks to embed and store.
   * @param source Citation label applied to every text.
   * @returns The number of chunks written after splitting.
   */
  async addTexts({ source, texts }: { texts: string[]; source: string }): Promise<ChunkLength> {
    const docs: Document[] = texts.map((t) => new Document({ pageContent: t, metadata: { source } }));

    return await this.addDocuments({ documents: docs, source });
  }

  /**
   * Splits documents into overlapping chunks, stamps each with an `indexedAt`
   * timestamp, ensures the pgvector table exists, and embeds + stores them.
   *
   * @param documents Documents to ingest.
   * @param source Optional citation label; overwrites each document's
   *   `metadata.source` when provided.
   * @returns The number of chunks written.
   */
  async addDocuments({ documents, source }: { documents: Document[]; source?: string }): Promise<ChunkLength> {
    for (const doc of documents) if (source) doc.metadata.source = source;

    const chunks = await this.splitter.splitDocuments(documents);
    for (const chunk of chunks) chunk.metadata.indexedAt = new Date().toISOString();

    await this.store.ensureTableInDatabase(1536);
    await this.store.addDocuments(chunks);

    console.log(`Added ${chunks.length} chunks from ${documents.length} documents`);
    return chunks.length;
  }

  /**
   * Counts the chunks currently stored in the vector table (creating the table
   * first if it doesn't exist yet).
   *
   * @returns The total number of stored chunks.
   */
  async getDocumentCount(): Promise<number> {
    await this.store.ensureTableInDatabase(1536);
    const result = await this.store.pool.query<{ count: number }>(
      `SELECT COUNT(*)::int AS count FROM ${this.store.computedTableName}`,
    );

    return result.rows[0]?.count ?? 0;
  }

  /**
   * Deletes every chunk from the vector table, leaving the table itself in
   * place. Useful for re-ingesting a knowledge base from scratch.
   *
   * @returns The number of rows deleted.
   */
  async clearTable(): Promise<number> {
    await this.store.ensureTableInDatabase(1536);
    const result = await this.store.pool.query<{ count: number }>(
      `WITH deleted AS (
        DELETE FROM ${this.store.computedTableName}
        RETURNING 1
      )
      SELECT COUNT(*)::int AS count FROM deleted`,
    );

    return result.rows[0]?.count ?? 0;
  }

  /**
   * Renders retrieved documents into the numbered, source-labelled block that
   * gets injected into the prompt as grounding context.
   *
   * @param documents The retrieved chunks (may be empty/undefined).
   * @returns A formatted context string, or a "no documents" placeholder.
   */
  private formatDocsForRetrieve(documents?: Document[]): string {
    if (!documents || documents?.length === 0) return "No relevant documents found";

    return documents
      ?.map((doc, i) => {
        const source = doc.metadata.source;
        return `[Source ${i + 1}: ${source}]\n${doc.pageContent}`;
      })
      .join("\n\n --- \n\n");
  }

  /**
   * Builds the retriever used by {@link ask}. In basic mode this is a plain
   * similarity retriever (top 4). In advanced mode it wraps that in a
   * multi-query strategy: the LLM rewrites the question into several search
   * queries, each is retrieved in parallel, and the results are flattened and
   * deduped — improving recall for questions phrased differently to the source.
   *
   * @returns Either the base retriever or an object exposing the same
   *   `invoke(question)` contract.
   */
  private buildRetriever() {
    const baseRetriever = this.store.asRetriever({ searchType: "similarity", k: 4 });
    if (!this.advancedRetriever) return baseRetriever;

    const queryPrompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You generate search queries for a vector database.

Given a user question, create 4 different search queries that could retrieve relevant documents.

Rules:
- Keep each query short and focused.
- Use different wording for each query.
- Do not answer the question.
- Return only the structured output.`,
      ],
      ["human", "User question: {question}"],
    ]);

    const queryGenerator = queryPrompt.pipe(this.llm.withStructuredOutput(MultiQuerySchema));

    return {
      invoke: async (question: string): Promise<Document[]> => {
        const result = await queryGenerator.invoke({ question });
        const queries = [question, ...result.queries];
        console.debug(`Multi query result: ${queries}`);
        const nestedDocs = await Promise.all(queries.map((query) => baseRetriever.invoke(query)));
        const docs = nestedDocs.flat();
        return this.dedupeDocuments(docs);
      },
    };
  }

  /**
   * Gets the message history for a session, lazily creating an empty one on
   * first use so callers always receive a valid store.
   *
   * @param sessionId The conversation to look up.
   */
  private getSessionHistory(sessionId: string): InMemoryChatMessageHistory {
    let messageHistory = this.sessionStore.get(sessionId);
    if (!messageHistory) {
      messageHistory = new InMemoryChatMessageHistory();
      this.sessionStore.set(sessionId, messageHistory);
    }

    return messageHistory;
  }

  /**
   * Removes duplicate chunks that the multi-query fan-out returns more than
   * once, keyed on source + id + a content prefix. Keeps first occurrence.
   *
   * @param docs Possibly-overlapping documents from several queries.
   * @returns The documents with duplicates removed.
   */
  private dedupeDocuments(docs: Document[]): Document[] {
    const seen = new Set<string>();
    const unique: Document[] = [];

    for (const doc of docs) {
      const key = [doc.metadata?.source ?? "", doc.metadata?.id ?? "", doc.pageContent.slice(0, 200)].join("|");

      if (!seen.has(key)) {
        seen.add(key);
        unique.push(doc);
      }
    }

    return unique;
  }
}

interface Config {
  apiKey: string;
  embeddingModel?: OpenAIEmbeddingModelId;
  openAiModel?: OpenAIChatModelId;
  advancedRetriever?: boolean;
  db: {
    dbHost: string;
    dbPort: string;
    dbUsername: string;
    dbPassword: string;
    dbDatabase: string;
  };
}

type ChunkLength = number;
