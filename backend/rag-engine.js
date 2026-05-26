// Node 24.11.1 has native global fetch, so we do not need to import any fetch libraries.
import { traceable } from 'langsmith/traceable';

// Stopwords for cleaner TF-IDF vectors
const STOPWORDS = new Set([
  'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'arent', 'as', 'at',
  'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'cant', 'cannot', 'could',
  'couldnt', 'did', 'didnt', 'do', 'does', 'doesnt', 'doing', 'dont', 'down', 'during', 'each', 'few', 'for', 'from',
  'further', 'had', 'hadnt', 'has', 'hasnt', 'have', 'havent', 'having', 'he', 'hed', 'hell', 'hes', 'her', 'here',
  'heres', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'hows', 'i', 'id', 'ill', 'im', 'ive', 'if', 'in',
  'into', 'is', 'isnt', 'it', 'its', 'itself', 'lets', 'me', 'more', 'most', 'mustnt', 'my', 'myself', 'no', 'nor',
  'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
  'same', 'shant', 'she', 'shed', 'shell', 'shes', 'should', 'shouldnt', 'so', 'some', 'such', 'than', 'that',
  'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'theres', 'these', 'they', 'theyd',
  'theyll', 'theyre', 'theyve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
  'wasnt', 'we', 'wed', 'well', 'were', 'weve', 'werent', 'what', 'whats', 'when', 'whens', 'where', 'wheres',
  'which', 'while', 'who', 'whos', 'whom', 'why', 'whys', 'with', 'wont', 'would', 'wouldnt', 'you', 'youd',
  'youll', 'youre', 'youve', 'your', 'yours', 'yourself', 'yourselves'
]);

// Helper to tokenize and clean text
function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '') // remove punctuation
    .split(/\s+/)
    .filter(word => word && !STOPWORDS.has(word));
}

export class LocalVectorStore {
  constructor() {
    this.chunks = []; // Array of { id, text, metadata }
    this.idf = {};     // Inverse document frequency mapping
    this.vocabulary = new Set();
    this.vectors = []; // Array of TF-IDF vectors corresponding to chunks
  }

  clear() {
    this.chunks = [];
    this.idf = {};
    this.vocabulary.clear();
    this.vectors = [];
  }

  // Load and index text chunks
  indexChunks(chunks) {
    this.clear();
    this.chunks = chunks;

    if (chunks.length === 0) return;

    const totalDocs = chunks.length;
    const documentFrequencies = {};

    // 1. Calculate Term Frequencies (TF) for each chunk and document frequencies (DF)
    const tfs = chunks.map((chunk, idx) => {
      const words = tokenize(chunk.text);
      const tf = {};
      const uniqueWordsInDoc = new Set(words);

      words.forEach(word => {
        tf[word] = (tf[word] || 0) + 1;
        this.vocabulary.add(word);
      });

      // Normalize Term Frequencies by doc length (to avoid biasing long chunks)
      const docLength = words.length || 1;
      for (const word in tf) {
        tf[word] = tf[word] / docLength;
      }

      uniqueWordsInDoc.forEach(word => {
        documentFrequencies[word] = (documentFrequencies[word] || 0) + 1;
      });

      return tf;
    });

    // 2. Calculate IDF for all words in the vocabulary
    this.vocabulary.forEach(word => {
      const df = documentFrequencies[word] || 0;
      // Add 1 to numerator and denominator to prevent division by zero (smoothing)
      this.idf[word] = Math.log((totalDocs + 1) / (df + 1)) + 1;
    });

    // 3. Construct the TF-IDF vectors for each chunk
    this.vectors = tfs.map(tf => {
      const vector = {};
      for (const word in tf) {
        vector[word] = tf[word] * (this.idf[word] || 0);
      }
      return vector;
    });

    console.log(`[RAG Engine] Local TF-IDF indexed ${this.chunks.length} chunks with vocabulary size ${this.vocabulary.size}`);
  }

  // Perform Cosine Similarity Search with Exact Legal Article Boosting
  searchSimilarity(query, topK = 3) {
    if (this.chunks.length === 0) return [];

    const queryWords = tokenize(query);
    if (queryWords.length === 0) {
      // Return first few chunks if query tokenization results in nothing
      return this.chunks.slice(0, topK).map((chunk, idx) => ({
        chunk,
        score: 0.1
      }));
    }

    // Parse query for exact legal article numbers (e.g. "Article 19" or "19")
    const articleMatch = query.match(/\b(12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35)\b/);
    const targetArticle = articleMatch ? `Article ${articleMatch[1]}` : null;

    // 1. Build Query Vector (TF-IDF)
    const queryTf = {};
    queryWords.forEach(word => {
      queryTf[word] = (queryTf[word] || 0) + 1;
    });
    const queryLength = queryWords.length || 1;

    const queryVector = {};
    for (const word in queryTf) {
      if (this.vocabulary.has(word)) {
        // Query TF normalized * IDF
        queryVector[word] = (queryTf[word] / queryLength) * (this.idf[word] || 0);
      }
    }

    // 2. Compute Cosine Similarity for each chunk
    const results = this.vectors.map((chunkVector, idx) => {
      let dotProduct = 0;
      let chunkNormSquare = 0;
      let queryNormSquare = 0;

      // Since vectors are sparse maps, dot product is easy
      for (const word in queryVector) {
        if (chunkVector[word]) {
          dotProduct += queryVector[word] * chunkVector[word];
        }
        queryNormSquare += queryVector[word] * queryVector[word];
      }

      // Compute norms
      for (const word in chunkVector) {
        chunkNormSquare += chunkVector[word] * chunkVector[word];
      }

      const chunkNorm = Math.sqrt(chunkNormSquare);
      const queryNorm = Math.sqrt(queryNormSquare);

      let similarity = (chunkNorm > 0 && queryNorm > 0) 
        ? dotProduct / (chunkNorm * queryNorm) 
        : 0;

      // Apply highly precise heading boost to keep short legal questions targeted to the actual articles
      const chunkMeta = this.chunks[idx].metadata;
      if (targetArticle && chunkMeta && chunkMeta.article === targetArticle) {
        similarity += 10.0;
      }

      return {
        chunk: this.chunks[idx],
        score: similarity
      };
    });

    // 3. Sort by similarity descending
    return results
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  // Keyword Search / Exact Term Matching with Exact Legal Article Boosting
  searchKeyword(query, topK = 3) {
    if (this.chunks.length === 0) return [];

    const queryWords = tokenize(query);
    if (queryWords.length === 0) return [];

    // Parse query for exact legal article numbers
    const articleMatch = query.match(/\b(12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35)\b/);
    const targetArticle = articleMatch ? `Article ${articleMatch[1]}` : null;

    const results = this.chunks.map(chunk => {
      const chunkTextLower = chunk.text.toLowerCase();
      let matchCount = 0;

      queryWords.forEach(word => {
        if (chunkTextLower.includes(word)) {
          matchCount++;
        }
      });

      // Score normalized by query words count
      let score = matchCount / queryWords.length;

      // Apply highly precise heading boost
      if (targetArticle && chunk.metadata && chunk.metadata.article === targetArticle) {
        score += 10.0;
      }

      return {
        chunk,
        score
      };
    });

    return results
      .filter(r => r.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }
}

// Advanced sliding window word-based chunker with Heading Propagation and Metadata extraction
export function chunkText(text, chunkSizeWords = 150, overlapWords = 30) {
  const normalizedText = text.replace(/\s+/g, ' ').trim();
  
  // Parse all legal article headings and their character indices in the normalized string
  const headingRegex = /\b(12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35)[A-Z]?\.\s+([A-Z][a-zA-Z]+)/g;
  const headings = [];
  let match;
  while ((match = headingRegex.exec(normalizedText)) !== null) {
    headings.push({
      index: match.index,
      articleNum: match[1],
      title: match[2]
    });
  }

  const words = normalizedText.split(' ');
  const chunks = [];
  
  let index = 0;
  let chunkCount = 0;

  // Resolves the active article heading at any given character index
  function getActiveArticle(charIndex) {
    let active = null;
    for (let i = 0; i < headings.length; i++) {
      if (headings[i].index <= charIndex) {
        active = headings[i];
      } else {
        break;
      }
    }
    return active;
  }

  while (index < words.length) {
    const chunkWords = words.slice(index, index + chunkSizeWords);
    const chunkText = chunkWords.join(' ');
    
    // Track position in normalizedText to propagate the active heading
    const charPos = normalizedText.indexOf(chunkText);
    const activeArticle = getActiveArticle(charPos !== -1 ? charPos : 0);
    
    let prefix = "";
    if (activeArticle) {
      prefix = `[Article ${activeArticle.articleNum} - ${activeArticle.title}] `;
    }

    chunks.push({
      id: `chunk_${chunkCount++}`,
      text: prefix + chunkText,
      metadata: {
        startWord: index,
        endWord: index + chunkWords.length,
        article: activeArticle ? `Article ${activeArticle.articleNum}` : null
      }
    });

    // Move forward by chunk size minus overlap
    index += (chunkSizeWords - overlapWords);

    // Safeguard to prevent infinite loops if overlap >= chunk size
    if (chunkSizeWords <= overlapWords) {
      index += 1;
    }
  }

  return chunks;
}

// Chroma DB REST Client Integration
export class ChromaClient {
  constructor(url = 'http://localhost:8000', apiKey = '') {
    this.baseUrl = url.endsWith('/') ? url.slice(0, -1) : url;
    this.apiKey = apiKey;
    this.collectionName = 'constitution_part3_rag';
  }

  getHeaders() {
    const headers = {
      'Content-Type': 'application/json'
    };
    if (this.apiKey) {
      // Chroma token auth standard is Bearer or custom headers
      headers['Authorization'] = `Bearer ${this.apiKey}`;
      headers['x-chroma-token'] = this.apiKey;
    }
    return headers;
  }

  async checkHealth() {
    try {
      const response = await fetch(`${this.baseUrl}/api/v1`, { 
        method: 'GET',
        headers: this.getHeaders(),
        signal: AbortSignal.timeout(3000) // 3s timeout
      });
      return response.ok;
    } catch (e) {
      // Chroma DB not reachable or not running. Fail cleanly to fall back to Local TF-IDF store.
      return false;
    }
  }

  async getOrCreateCollection() {
    const response = await fetch(`${this.baseUrl}/api/v1/collections`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({
        name: this.collectionName,
        metadata: { "description": "Collection for simple constitution RAG" },
        get_or_create: true
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to get/create Chroma collection: ${errorText}`);
    }

    return await response.json();
  }

  async addDocuments(chunks, localVectorStore) {
    const collection = await this.getOrCreateCollection();
    const collectionId = collection.id;

    // Chroma DB HTTP API format for /add:
    // ids: string[]
    // documents: string[]
    // metadatas: object[]
    // embeddings: number[][] (optional, generated server-side if not provided, but since Chroma DB
    // server-side defaults might fail or be missing, we can pre-generate sparse vectors or use Chroma's defaults)
    const payload = {
      ids: chunks.map(c => c.id),
      documents: chunks.map(c => c.text),
      metadatas: chunks.map(c => ({
        startWord: c.metadata.startWord,
        endWord: c.metadata.endWord
      }))
    };

    // If localVectorStore is initialized, we can also calculate local TFIDF scores as embeddings fallback,
    // but standard Chroma server-side default embedding function is generally preferred. We will let Chroma handle it.
    const response = await fetch(`${this.baseUrl}/api/v1/collections/${collectionId}/add`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Chroma DB add error: ${errorText}`);
    }

    return await response.json();
  }

  async querySimilarity(queryText, nResults = 3) {
    const collection = await this.getOrCreateCollection();
    const collectionId = collection.id;

    // Query endpoint format:
    // query_embeddings: number[][] (optional)
    // query_texts: string[] (optional, only if Chroma has built-in embedding functions)
    // n_results: number
    const response = await fetch(`${this.baseUrl}/api/v1/collections/${collectionId}/query`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({
        query_texts: [queryText],
        n_results: nResults
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Chroma DB query error: ${errorText}`);
    }

    const data = await response.json();
    // Format response to match local store response format
    const results = [];
    if (data.documents && data.documents[0]) {
      for (let i = 0; i < data.documents[0].length; i++) {
        results.push({
          chunk: {
            id: data.ids[0][i],
            text: data.documents[0][i],
            metadata: data.metadatas[0][i] || {}
          },
          score: data.distances ? (1 - data.distances[0][i]) : 0.8 // convert distance to similarity score
        });
      }
    }
    return results;
  }
}

// Groq API Completion Client - Wrapped with LangSmith traceable for robust tracking
export const queryGroqLLM = traceable(
  async function (apiKey, query, contextChunks, systemPromptOverride) {
    if (!apiKey || apiKey === 'your_groq_api_key_here') {
      throw new Error('Groq API Key is not configured. Please supply a valid Groq API Key.');
    }

    // Combine context chunks
    const contextText = contextChunks.map((c, i) => `[Chunk #${i + 1}]\n${c.text}`).join('\n\n');

    // Strict anti-hallucination system prompt as requested:
    // "the answer for the query should be from the given document only should NOT TAKE FROM THE OUTSIDE. and also the answer should not be hallucinated."
    const defaultSystemPrompt = `You are an expert Naive RAG Assistant specialized in analyzing documents.
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

    const systemPrompt = systemPromptOverride || defaultSystemPrompt;

    const requestBody = {
      model: 'llama-3.3-70b-versatile', // Groq's high capacity fast LLM
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: query }
      ],
      temperature: 0.0, // Set temperature to 0 to minimize hallucinations and force deterministic answers
      max_tokens: 1024
    };

    try {
      const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        const errMsg = errData.error?.message || `HTTP ${response.status} ${response.statusText}`;
        throw new Error(`Groq API Error: ${errMsg}`);
      }

      const data = await response.json();
      return data.choices[0].message.content;
    } catch (error) {
      console.error(`[Groq LLM Error]: ${error.message}`);
      throw error;
    }
  },
  {
    name: "queryGroqLLM",
    run_type: "llm"
  }
);
