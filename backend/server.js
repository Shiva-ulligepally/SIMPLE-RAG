import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import pdf from 'pdf-parse';
import { LocalVectorStore, chunkText, ChromaClient, queryGroqLLM } from './rag-engine.js';

// Setup __dirname for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log("[DEBUG] __dirname:", __dirname);
const documentsDir = path.join(__dirname, 'documents');
console.log("[DEBUG] documentsDir:", documentsDir);
if (fs.existsSync(documentsDir)) {
  console.log("[DEBUG] Documents found:");
  console.log(fs.readdirSync(documentsDir));
} else {
  console.log("[DEBUG] Documents directory missing");
}

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors({
  origin: true,
  credentials: true
}));
app.use(express.json({ limit: "25mb" }));
app.use(express.urlencoded({
  extended: true,
  limit: "25mb"
}));

// Helper: Scan documents directory dynamically for all PDF files
function getPdfFiles() {
  const documentsDir = path.join(__dirname, 'documents');
  try {
    if (!fs.existsSync(documentsDir)) {
      console.error("[PDF ERROR] Documents directory missing");
      return [];
    }
    const files = fs.readdirSync(documentsDir);
    return files
      .filter(f => f.toLowerCase().endsWith('.pdf'))
      .map(f => ({
        name: f,
        path: path.join(documentsDir, f)
      }));
  } catch (e) {
    console.error(`[PDF Scan Error]: ${e.message}`);
    return [];
  }
}

// Global in-memory vector store instance
const localStore = new LocalVectorStore();
let isIndexed = false;
let indexedMetadata = {
  fileName: '',
  chunkCount: 0,
  totalPages: 0,
  charCount: 0,
  vocabularySize: 0,
  indexTime: null,
  files: []
};

// Check if Chroma is available
async function getChromaHealth(url, apiKey) {
  const client = new ChromaClient(url || process.env.CHROMA_URL, apiKey || process.env.CHROMA_API_KEY);
  return await client.checkHealth();
}

// Shareable multi-document indexing workflow
async function performIndexing(chunkSize = 150, chunkOverlap = 30, chromaUrl = null, chromaApiKey = null) {
  const pdfFiles = getPdfFiles();
  if (pdfFiles.length === 0) {
    throw new Error("No PDF documents found in the project root directory.");
  }

  console.log(`[RAG Engine] Starting indexing for ${pdfFiles.length} documents...`);
  let allChunks = [];
  let totalPages = 0;
  let totalChars = 0;
  const fileStats = [];

  for (const file of pdfFiles) {
    console.log(`[RAG Engine] Indexing file: ${file.name}`);
    const dataBuffer = fs.readFileSync(file.path);
    const parsedPdf = await pdf(dataBuffer);
    const text = parsedPdf.text;
    
    totalPages += parsedPdf.numpages;
    totalChars += text.length;

    // Use enhanced heading propagation chunker
    const chunks = chunkText(text, chunkSize, chunkOverlap);
    
    // Prefix chunk IDs and store source file in metadata
    const fileSlug = file.name.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase();
    chunks.forEach((c) => {
      c.metadata = {
        ...c.metadata,
        source: file.name
      };
      c.id = `${fileSlug}_${c.id}`;
    });

    allChunks = allChunks.concat(chunks);
    fileStats.push({
      fileName: file.name,
      chunkCount: chunks.length,
      pages: parsedPdf.numpages,
      chars: text.length
    });
  }

  // Index all combined chunks in our local TF-IDF vector database
  localStore.indexChunks(allChunks);

  // Store in Chroma DB if available
  let chromaIndexed = false;
  let chromaError = null;
  const activeChromaUrl = chromaUrl || process.env.CHROMA_URL || 'http://localhost:8000';
  const activeChromaKey = chromaApiKey || process.env.CHROMA_API_KEY || '';
  
  const chromaHealthy = await getChromaHealth(activeChromaUrl, activeChromaKey);
  if (chromaHealthy) {
    try {
      console.log(`[RAG Engine] Connecting and indexing to Chroma DB collection at ${activeChromaUrl}...`);
      const chroma = new ChromaClient(activeChromaUrl, activeChromaKey);
      await chroma.addDocuments(allChunks, localStore);
      chromaIndexed = true;
    } catch (err) {
      console.error(`[RAG Engine] Chroma DB insertion failed: ${err.message}`);
      chromaError = err.message;
    }
  }

  isIndexed = true;
  indexedMetadata = {
    fileName: pdfFiles.map(f => f.name).join(', '),
    chunkCount: allChunks.length,
    totalPages,
    charCount: totalChars,
    vocabularySize: localStore.vocabulary.size,
    indexTime: new Date().toISOString(),
    chunkSize,
    chunkOverlap,
    files: fileStats
  };

  return {
    stats: indexedMetadata,
    chroma: {
      indexed: chromaIndexed,
      error: chromaError,
      url: activeChromaUrl
    }
  };
}

// API Health Check / System Status
app.get('/api/status', async (req, res) => {
  const queryChroma = req.query.chromaUrl;
  const queryChromaKey = req.query.chromaApiKey;
  
  const chromaUrl = queryChroma || process.env.CHROMA_URL || 'http://localhost:8000';
  const chromaApiKey = queryChromaKey || process.env.CHROMA_API_KEY || '';

  const pdfFiles = getPdfFiles();
  const chromaHealthy = await getChromaHealth(chromaUrl, chromaApiKey);

  res.json({
    status: 'online',
    pdfFound: pdfFiles.length > 0,
    pdfFiles: pdfFiles.map(f => f.name),
    pdfPath: pdfFiles.map(f => f.path).join(', '),
    isIndexed,
    indexStats: isIndexed ? indexedMetadata : null,
    chroma: {
      connected: chromaHealthy,
      url: chromaUrl,
      collection: 'constitution_part3_rag'
    }
  });
});

// Endpoint to index all root documents
app.post('/api/index-existing', async (req, res) => {
  try {
    const { chunkSize = 150, chunkOverlap = 30, chromaUrl, chromaApiKey } = req.body;
    
    const result = await performIndexing(chunkSize, chunkOverlap, chromaUrl, chromaApiKey);

    res.json({
      success: true,
      message: `Successfully parsed, chunked, and indexed all documents!`,
      stats: result.stats,
      chroma: result.chroma
    });

  } catch (error) {
    console.error(`[Indexer Error]: ${error.message}`);
    res.status(500).json({
      success: false,
      error: `Internal server indexing error: ${error.message}`
    });
  }
});

// Endpoint to query RAG across all indexed documents
app.post('/api/query', async (req, res) => {
  try {
    const {
      query,
      searchType = 'similarity', // 'similarity', 'keyword', 'chroma'
      numResults = 3,
      chunkSize = 150,
      chunkOverlap = 30,
      groqApiKey,
      chromaUrl,
      chromaApiKey
    } = req.body;

    if (!query) {
      return res.status(400).json({ success: false, error: 'Query parameter is required.' });
    }

    // Auto-index fallback if not indexed yet
    if (!isIndexed) {
      console.log(`[RAG Query] Server not indexed. Auto-indexing all PDF documents...`);
      try {
        await performIndexing(chunkSize, chunkOverlap, chromaUrl, chromaApiKey);
      } catch (err) {
        return res.status(404).json({
          success: false,
          error: `Auto-indexing failed: ${err.message}. Please upload documents to the project root.`
        });
      }
    }

    let retrievedResults = [];
    let usedEngine = 'local-similarity';

    const activeChromaUrl = chromaUrl || process.env.CHROMA_URL || 'http://localhost:8000';
    const activeChromaKey = chromaApiKey || process.env.CHROMA_API_KEY || '';

    // Route search queries based on searchType
    if (searchType === 'chroma') {
      const chromaHealthy = await getChromaHealth(activeChromaUrl, activeChromaKey);
      if (chromaHealthy) {
        try {
          console.log(`[RAG Query] Performing similarity search in Chroma DB...`);
          const chroma = new ChromaClient(activeChromaUrl, activeChromaKey);
          retrievedResults = await chroma.querySimilarity(query, numResults);
          usedEngine = 'chroma-similarity';
        } catch (err) {
          console.error(`[Chroma Search Error] Fallback to local similarity: ${err.message}`);
          retrievedResults = localStore.searchSimilarity(query, numResults);
          usedEngine = `local-similarity (Chroma error: ${err.message})`;
        }
      } else {
        console.log(`[RAG Query] Chroma DB offline. Fallback to local similarity...`);
        retrievedResults = localStore.searchSimilarity(query, numResults);
        usedEngine = 'local-similarity (Chroma Offline Fallback)';
      }
    } else if (searchType === 'keyword') {
      console.log(`[RAG Query] Performing local keyword search...`);
      retrievedResults = localStore.searchKeyword(query, numResults);
      usedEngine = 'local-keyword';
    } else {
      // Default: local similarity (TF-IDF + Cosine)
      console.log(`[RAG Query] Performing local TF-IDF similarity search...`);
      retrievedResults = localStore.searchSimilarity(query, numResults);
      usedEngine = 'local-similarity';
    }

    if (retrievedResults.length === 0) {
      retrievedResults = localStore.chunks.slice(0, numResults).map(c => ({ chunk: c, score: 0.0 }));
    }

    // Ingest results context and query Groq LLM
    const contextChunks = retrievedResults.map(r => r.chunk);
    const activeGroqKey = groqApiKey || process.env.GROQ_API_KEY;

    if (!activeGroqKey || activeGroqKey === 'your_groq_api_key_here' || activeGroqKey.trim() === '') {
      return res.status(400).json({
        success: false,
        error: 'Groq API Key is missing. Please provide it in settings or environment.',
        retrievedChunks: retrievedResults,
        usedEngine
      });
    }

    console.log(`[RAG Query] Invoking Groq LLM API with ${contextChunks.length} context chunks...`);
    const answer = await queryGroqLLM(activeGroqKey, query, contextChunks);

    const isUnrelated = answer.includes("I am sorry, but the provided document does not contain information");

    // Build full prompt inspector payload (tracking document source per chunk!)
    const contextText = contextChunks.map((c, i) => `[Chunk #${i + 1} from Source: ${c.metadata?.source || 'Unknown File'}]\n${c.text}`).join('\n\n');
    const fullSystemPrompt = `You are an expert Naive RAG Assistant specialized in analyzing documents.
You will be provided a document context below, followed by a user query. You MUST follow these rules strictly:

1. The answer for the query MUST be retrieved from the given document ONLY.
2. Do NOT take any information from the outside. Do NOT assume, speculate, or draw from external knowledge.
3. Your answer must NOT be hallucinated. If the context does not contain the answer, you must output EXACTLY the following text:
   "I am sorry, but the provided document does not contain information to answer your question. Please ask something related to the document."
4. If the user's query is completely unrelated to the topic of the document, you must output EXACTLY:
   "I am sorry, but the provided document does not contain information to answer your question. Please ask something related to the document."
5. Ground all explanations in direct facts from the text.

---
DOCUMENT CONTEXT:
${contextText}
---`;

    res.json({
      success: true,
      answer,
      isUnrelated,
      retrievedChunks: retrievedResults,
      usedEngine,
      inspector: {
        systemPrompt: fullSystemPrompt,
        userQuery: query
      }
    });

  } catch (error) {
    console.error(`[Query Route Error]: ${error.message}`);
    res.status(500).json({
      success: false,
      error: `RAG search or LLM generation failed: ${error.message}`
    });
  }
});

if (!process.env.VERCEL) {
  app.listen(PORT, () => {
    console.log(`[Server Ready] Naive RAG backend running on http://localhost:${PORT}`);
  });
}

export default app;
