# enhanced_rag_app_fixed.py - Fixed Enhanced RAG with Speech Processing

from dotenv import load_dotenv
load_dotenv()

import os
import fitz
import re
import requests
import urllib.parse
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import numpy as np
from bs4 import BeautifulSoup
import streamlit as st
import tempfile
import json
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Google Gemini imports
import google.generativeai as genai
from google.generativeai import GenerativeModel

# Import fixed speech processor
try:
    from speech_processor import SpeechProcessor, ensure_dir, save_text, save_json
    SPEECH_PROCESSING_AVAILABLE = True
    print("✅ Speech processing library loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: Speech processing not available: {e}")
    SPEECH_PROCESSING_AVAILABLE = False
    # Create dummy classes
    class SpeechProcessor:
        def __init__(self, *args, **kwargs):
            pass
        def process_audio(self, *args, **kwargs):
            return {"error": "Speech processing not available"}
        def generate_podcast(self, *args, **kwargs):
            return False
    
    def ensure_dir(p): p.mkdir(parents=True, exist_ok=True)
    def save_text(p, t): p.write_text(t, encoding="utf-8")
    def save_json(p, o): p.write_text(json.dumps(o, indent=2), encoding="utf-8")

# Optional Supabase client
try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except Exception:
    create_client = None
    SUPABASE_AVAILABLE = False

# Environment variables
SUPABASE_DB_PASSWORD = os.environ.get("SUPABASE_DB_PASSWORD")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://zwtvfgptlsrmzimecpkp.supabase.co")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Google Gemini API configured")
else:
    print("⚠️ Warning: No Google API key found")

llm = GenerativeModel("gemini-2.0-flash-exp") if GOOGLE_API_KEY else None
embedding_model = GenerativeModel("embedding-001") if GOOGLE_API_KEY else None

# --------- Core Embedding ---------
class GoogleEmbedder:
    def __init__(self):
        self.model = embedding_model

    def get_embedding(self, texts: List[str]) -> List[List[float]]:
        if not GOOGLE_API_KEY or not self.model:
            print("❌ Google embedding model not available")
            return []
        embeddings = []
        for text in texts:
            try:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )
                # Flexible extraction
                emb = []
                if isinstance(result, dict):
                    if "embedding" in result:
                        maybe = result["embedding"]
                        if isinstance(maybe, dict) and "values" in maybe:
                            emb = maybe["values"]
                        elif isinstance(maybe, list):
                            emb = maybe
                    elif "values" in result:
                        emb = result["values"]
                elif hasattr(result, "embedding"):
                    maybe = result.embedding
                    if isinstance(maybe, dict) and "values" in maybe:
                        emb = maybe["values"]
                    elif isinstance(maybe, list):
                        emb = maybe
                if not emb:
                    try:
                        emb = result["embedding"]["values"]
                    except Exception:
                        emb = []
                emb = [float(x) for x in emb] if emb else []
                embeddings.append(emb)
            except Exception as e:
                print(f"Embedding error: {e}")
                embeddings.append([0.0] * 768)
        return embeddings

embeddings = GoogleEmbedder()

# --------- Chunking / Document Processing ---------
class SimpleChunker:
    def __init__(self, chunk_size: int = 1200, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document_text: str) -> List[Dict]:
        content = document_text or ""
        chunks = []
        start = 0
        idx = 0
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_text = content[start:end].strip()
            metadata = {"chunk_index": idx}
            chunks.append({"content": chunk_text, "metadata": metadata})
            idx += 1
            if end >= len(content):
                break
            start = end - self.overlap
        return chunks

class DocumentProcessor:
    def __init__(self):
        self.chunker = SimpleChunker()

    def extract_text_from_pdf(self, filepath: str) -> str:
        try:
            doc = fitz.open(filepath)
            full_text = ""
            for page in doc:
                try:
                    full_text += page.get_text("text") + "\n"
                except Exception:
                    continue
            doc.close()
            return full_text.strip()
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""

    def process_file(self, filepath: str) -> Dict[str, Any]:
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            if p.suffix.lower() == ".pdf":
                full_text = self.extract_text_from_pdf(filepath)
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    full_text = f.read()
            
            chunks = self.chunker.chunk(full_text)
            return {"full_text": full_text, "chunks": chunks}
        except Exception as e:
            print(f"File processing error: {e}")
            return {"full_text": "", "chunks": []}

# --------- In-Memory Semantic Store ---------
class InMemoryDocumentStore:
    def __init__(self):
        self.documents: Dict[str, Dict] = {}

    def add_document(self, filepath: str, full_text: str, chunks: List[Dict], embeddings_list: List[Optional[List[float]]]):
        self.documents[filepath] = {
            "full_text": full_text,
            "chunks": chunks,
            "embeddings": embeddings_list
        }

    def search_similar(self, query_embedding: List[float], max_results: int = 5) -> List[Dict]:
        results = []
        for filepath, doc in self.documents.items():
            chunks = doc.get("chunks", [])
            embeddings_list = doc.get("embeddings", [])
            for i, emb in enumerate(embeddings_list):
                if not emb:
                    continue
                sim = self._cosine_similarity(query_embedding, emb)
                results.append({
                    "source": filepath, 
                    "chunk_index": i, 
                    "content": chunks[i]["content"], 
                    "similarity": sim
                })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:max_results]

    @staticmethod
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
        try:
            a = np.array(v1, dtype=float)
            b = np.array(v2, dtype=float)
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0:
                return 0.0
            return float(np.dot(a, b) / denom)
        except Exception:
            return 0.0

# --------- Database Manager (Supabase) ---------
class DatabaseManager:
    def __init__(self, SUPABASE_URL: str, SUPABASE_KEY: str):
        self.supabase_url = SUPABASE_URL
        self.supabase_key = SUPABASE_KEY
        self.supabase = None
        self.connected = False
        if SUPABASE_AVAILABLE and create_client and SUPABASE_KEY:
            try:
                self.supabase = create_client(self.supabase_url, self.supabase_key)
                self.supabase.table('conversations').select('id').limit(1).execute()
                self.connected = True
                print("✅ Database connected successfully")
            except Exception as e:
                print(f"Database connection failed: {e}")
                self.connected = False
                self.supabase = None

    def is_connected(self) -> bool:
        return self.connected and self.supabase is not None

    def save_conversation(self, session_id: str, query: str, response: str, context: str = None):
        if not self.is_connected():
            return None
        try:
            conversation_data = {
                'session_id': session_id,
                'query': query,
                'response': response,
                'context': context,
                'timestamp': datetime.now().isoformat()
            }
            result = self.supabase.table('conversations').insert(conversation_data).execute()
            return result.data[0]['id'] if result.data else None
        except Exception as e:
            print(f"Error saving conversation: {str(e)}")
            return None

    def save_document_metadata(self, filepath: str, document_type: str, chunk_count: int):
        if not self.is_connected():
            return None
        try:
            doc_data = {
                'filepath': filepath,
                'document_type': document_type,
                'chunk_count': chunk_count,
                'processed_at': datetime.now().isoformat()
            }
            result = self.supabase.table('document_metadata').insert(doc_data).execute()
            return result.data[0]['id'] if result.data else None
        except Exception as e:
            print(f"Error saving document metadata: {str(e)}")
            return None

# --------- Conversation Memory ---------
class ConversationMemory:
    def __init__(self, db_manager: DatabaseManager = None):
        self.history = []
        self.context_window = 10
        self.db_manager = db_manager

    def add_exchange(self, query: str, response: str, context: str = None, session_id: str = None):
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "context": context
        }
        self.history.append(exchange)
        if len(self.history) > self.context_window:
            self.history = self.history[-self.context_window:]
        if self.db_manager and self.db_manager.is_connected() and session_id:
            self.db_manager.save_conversation(session_id, query, response, context)

    def get_context_string(self, max_exchanges: int = 5) -> str:
        if not self.history:
            return ""
        recent_history = self.history[-max_exchanges:]
        context_parts = []
        for exchange in recent_history:
            context_parts.append(f"Human: {exchange['query']}")
            context_parts.append(f"Assistant: {exchange['response'][:200]}...")
        return "\n".join(context_parts)

# --------- Enhanced RAG Agent with Speech Processing ---------
class EnhancedRAGAgent:
    def __init__(self):
        self.db_manager = DatabaseManager(SUPABASE_URL, SUPABASE_KEY)
        self.processor = DocumentProcessor()
        self.store = InMemoryDocumentStore()
        self.conversation_memory = ConversationMemory(self.db_manager)
        self.latest_doc_path: Optional[str] = None
        self.audio_results: Dict[str, Any] = {}
        
        # Initialize speech processor if available
        if SPEECH_PROCESSING_AVAILABLE:
            try:
                self.speech_processor = SpeechProcessor()
                print("✅ Speech processor initialized")
            except Exception as e:
                print(f"⚠️ Speech processor initialization failed: {e}")
                self.speech_processor = None
        else:
            self.speech_processor = None

    def process_and_store(self, filepath: str, document_type: str = "general") -> str:
        try:
            result = self.processor.process_file(filepath)
            full_text = result["full_text"]
            chunks = result["chunks"]
            
            if not full_text.strip():
                return f"⚠️ No text extracted from {Path(filepath).name}"
            
            embeddings_list = []
            for chunk in chunks:
                emb = embeddings.get_embedding([chunk["content"]])
                embeddings_list.append(emb[0] if emb else None)
            
            self.store.add_document(filepath, full_text, chunks, embeddings_list)
            self.latest_doc_path = filepath
            
            if self.db_manager and self.db_manager.is_connected():
                self.db_manager.save_document_metadata(filepath, document_type, len(chunks))
            
            return f"✅ Successfully processed and embedded {len(chunks)} chunks from {Path(filepath).name}"
        except Exception as e:
            return f"❌ Error processing file: {e}"

    def process_audio_file(self, audio_filepath: str, language: str = None) -> Dict[str, Any]:
        """Process audio file and extract insights"""
        if not self.speech_processor:
            return {"error": "Speech processing not available"}
        
        try:
            print(f"🎵 Starting audio processing for: {Path(audio_filepath).name}")
            results = self.speech_processor.process_audio(audio_filepath, language)
            
            if "error" in results:
                return results
            
            self.audio_results = results
            
            # Store transcript as a document if available
            transcript_text = results.get('transcript', {}).get('text', '')
            if transcript_text:
                chunks = self.processor.chunker.chunk(transcript_text)
                embeddings_list = []
                for chunk in chunks:
                    emb = embeddings.get_embedding([chunk["content"]])
                    embeddings_list.append(emb[0] if emb else None)
                
                # Store with audio filename
                audio_doc_path = f"audio_{Path(audio_filepath).stem}"
                self.store.add_document(audio_doc_path, transcript_text, chunks, embeddings_list)
                self.latest_doc_path = audio_doc_path
                print(f"✅ Audio transcript stored as document: {audio_doc_path}")
            
            return results
        except Exception as e:
            error_msg = f"❌ Error processing audio: {e}"
            print(error_msg)
            return {"error": error_msg}

    def _get_doc_excerpt_for_prompt(self, filepath: Optional[str], max_chars: int = 6000) -> str:
        if not filepath:
            return ""
        entry = self.store.documents.get(filepath)
        if not entry:
            return ""
        full = entry.get("full_text", "")
        if not full:
            chunks = entry.get("chunks", [])
            return " ".join(c["content"] for c in chunks[:5])
        return full[:max_chars] + ("..." if len(full) > max_chars else "")

    def _is_advice_query(self, q: str) -> bool:
        qlow = q.lower()
        advice_keywords = [
            "improv", "advice", "suggest", "strategy", "how to", "recommend", "optimi", "role", "which role",
            "cover letter", "apply", "cv", "resume", "format", "better", "strength", "weakness", "fix", "edit",
            "action item", "todo", "task", "follow up", "next step"
        ]
        return any(k in qlow for k in advice_keywords)

    def chat(self, query: str, session_id: str = "default", use_web: bool = False) -> Dict[str, Any]:
        if not GOOGLE_API_KEY or not llm:
            return {
                "response": "❌ Google Gemini API not available. Please check your API key configuration.",
                "context": "API unavailable",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "sources_used": []
            }
        
        try:
            doc_path = self.latest_doc_path
            doc_excerpt = self._get_doc_excerpt_for_prompt(doc_path, max_chars=6000)
            advice_needed = self._is_advice_query(query)
            context_string = self.conversation_memory.get_context_string()
            
            # Include audio analysis results if available
            audio_context = ""
            if self.audio_results and not self.audio_results.get('error'):
                audio_context = f"\nAUDIO ANALYSIS RESULTS:\n"
                if self.audio_results.get('summary'):
                    audio_context += f"Summary: {self.audio_results['summary']}\n"
                if self.audio_results.get('action_items'):
                    audio_context += f"Action Items: {', '.join(self.audio_results['action_items'][:5])}\n"
                if self.audio_results.get('keywords'):
                    kw_list = [kw for kw, score in self.audio_results['keywords'][:5]]
                    audio_context += f"Key Topics: {', '.join(kw_list)}\n"

            system_instructions = (
                "You are an expert assistant with access to documents and audio analysis. "
                "Use the DOCUMENT EXCERPTS and AUDIO ANALYSIS for factual answers.\n"
                "- When the user asks for improvements, editing suggestions, role advice, or strategies, "
                "provide recommendations based on available content and mark each as [FROM_DOCUMENT], [FROM_AUDIO], or [INFERRED].\n"
                "- For audio content, consider action items, key moments, and extracted insights.\n"
                "- Provide concise actionable steps and examples.\n"
                "- If you must guess context, provide confidence level (High/Medium/Low) and evidence."
            )

            prompt_parts = [system_instructions]
            if doc_excerpt:
                prompt_parts.append("DOCUMENT EXCERPTS:\n" + doc_excerpt)
            if audio_context:
                prompt_parts.append(audio_context)
            if not doc_excerpt and not audio_context:
                prompt_parts.append("No document or audio content available.")
            if context_string:
                prompt_parts.append("Recent conversation context:\n" + context_string)
            prompt_parts.append("\nUSER QUESTION:\n" + query)

            if advice_needed:
                prompt_parts.append(
                    "\nUSER REQUEST TYPE: ADVICE/IMPROVEMENT. Provide:\n"
                    "1) Short diagnosis based on available content.\n"
                    "2) Top actionable improvements with source tags.\n"
                    "3) Example implementations if applicable.\n"
                    "4) Next steps or action items."
                )

            final_prompt = "\n\n".join(prompt_parts)
            
            try:
                response = llm.generate_content(final_prompt)
                text = response.text if hasattr(response, "text") else str(response)
                clean_text = re.sub(r"\n{3,}", "\n\n", text).strip()
            except Exception as e:
                clean_text = f"❌ Error generating response with Gemini: {e}"

            self.conversation_memory.add_exchange(
                query, clean_text, 
                f"Used document: {Path(doc_path).name if doc_path else 'None'}", 
                session_id
            )

            sources_used = []
            if doc_excerpt:
                sources_used.append("📄 Uploaded document")
            if self.audio_results and not self.audio_results.get('error'):
                sources_used.append("🎵 Audio analysis")

            return {
                "response": clean_text,
                "context": f"Used: {', '.join(sources_used) if sources_used else 'No sources'}",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "sources_used": sources_used
            }
        except Exception as e:
            return {
                "response": f"❌ Error processing query: {e}",
                "context": "",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "sources_used": []
            }

    def get_audio_insights(self, session_id: str = "default") -> Dict[str, Any]:
        """Get detailed insights from processed audio"""
        if not self.audio_results or self.audio_results.get('error'):
            return {
                "response": "❌ No audio processed yet or processing failed. Please upload an audio file first.",
                "context": "",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "sources_used": []
            }

        insights_text = "# 🎵 Audio Analysis Results\n\n"
        
        if self.audio_results.get('summary'):
            insights_text += f"## 📝 Summary\n{self.audio_results['summary']}\n\n"
        
        if self.audio_results.get('keywords'):
            kw_text = ", ".join([f"**{kw}** ({score:.2f})" for kw, score in self.audio_results['keywords'][:10]])
            insights_text += f"## 🔑 Key Topics\n{kw_text}\n\n"
        
        if self.audio_results.get('action_items'):
            insights_text += "## ✅ Action Items\n"
            for i, action in enumerate(self.audio_results['action_items'][:10], 1):
                insights_text += f"{i}. {action}\n"
            insights_text += "\n"
        
        if self.audio_results.get('key_moments'):
            insights_text += "## ⭐ Key Moments\n"
            for moment in self.audio_results['key_moments'][:5]:
                start_time = moment['start']
                insights_text += f"**{start_time:.1f}s**: {moment['text']} (Score: {moment['score']:.2f})\n"
            insights_text += "\n"

        if self.audio_results.get('mindmap'):
            insights_text += "## 🧠 Mind Map Structure\n"
            mindmap = self.audio_results['mindmap']
            if mindmap.get('mermaid'):
                insights_text += "```mermaid\n" + mindmap['mermaid'] + "\n```\n\n"

        return {
            "response": insights_text,
            "context": "Audio analysis insights",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "sources_used": ["🎵 Audio analysis"]
        }

    def generate_podcast_from_content(self, session_id: str = "default") -> str:
        """Generate podcast audio from processed content"""
        if not self.speech_processor:
            return "❌ Speech processing not available"
        
        if not self.latest_doc_path and not self.audio_results:
            return "❌ No content available to generate podcast"
        
        # Use summary if available, otherwise use document excerpt
        content_for_podcast = ""
        if self.audio_results and self.audio_results.get('summary'):
            content_for_podcast = self.audio_results['summary']
        elif self.latest_doc_path:
            content_for_podcast = self._get_doc_excerpt_for_prompt(self.latest_doc_path, max_chars=3000)
        
        if not content_for_podcast:
            return "❌ No suitable content found for podcast generation"
        
        # Generate podcast audio
        output_dir = Path("./outputs")
        ensure_dir(output_dir)
        podcast_path = output_dir / f"podcast_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        
        try:
            success = self.speech_processor.generate_podcast(content_for_podcast, str(podcast_path))
            if success:
                return f"✅ Podcast generated successfully: {podcast_path}"
            else:
                return "❌ Failed to generate podcast"
        except Exception as e:
            return f"❌ Error generating podcast: {e}"

    def get_status(self) -> Dict[str, Any]:
        return {
            "documents_in_memory": len(self.store.documents),
            "conversation_length": len(self.conversation_memory.history),
            "latest_doc": Path(self.latest_doc_path).name if self.latest_doc_path else None,
            "audio_processed": bool(self.audio_results and not self.audio_results.get('error')),
            "google_gemini_available": GOOGLE_API_KEY is not None,
            "database_connected": self.db_manager.is_connected() if self.db_manager else False,
            "speech_processor_available": SPEECH_PROCESSING_AVAILABLE and self.speech_processor is not None
        }

# --------- Enhanced Streamlit UI ---------
def create_streamlit_app():
    st.set_page_config(
        page_title="MemoryPal - RAG + Speech AI", 
        page_icon="🧠", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main > div {
        max-width: 100%;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if "rag_agent" not in st.session_state:
        with st.spinner("Initializing MemoryPal..."):
            st.session_state.rag_agent = EnhancedRAGAgent()
    rag = st.session_state.rag_agent

    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().timestamp()}"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.title("🧠 MemoryPal")
        st.markdown("*RAG + Speech AI Assistant*")
        
        st.subheader("📊 System Status")
        status = rag.get_status()
        for key, value in status.items():
            if isinstance(value, bool):
                icon = "✅" if value else "❌"
                st.write(f"{icon} {key.replace('_', ' ').title()}")
            else:
                st.write(f"📊 {key.replace('_', ' ').title()}: {value}")

        st.divider()
        
        # Document Upload Section
        st.subheader("📄 Document Upload")
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "md"])
        document_type = st.selectbox("Document Type", 
            ["general", "technical", "research", "resume", "legal", "medical"])
        
        if uploaded_file is not None:
            if st.button("📄 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    # Create temp directory if it doesn't exist
                    temp_dir = Path("temp")
                    temp_dir.mkdir(exist_ok=True)
                    
                    file_path = temp_dir / f"temp_{uploaded_file.name}"
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    result_msg = rag.process_and_store(str(file_path), document_type)
                    
                    if "✅" in result_msg:
                        st.success(result_msg)
                    else:
                        st.error(result_msg)

        st.divider()
        
        # Audio Upload Section
        st.subheader("🎵 Audio Upload & Processing")
        if not SPEECH_PROCESSING_AVAILABLE:
            st.warning("⚠️ Speech processing not available. Install required dependencies.")
        else:
            uploaded_audio = st.file_uploader("Choose an audio file", 
                type=["mp3", "wav", "m4a", "flac", "ogg"])
            audio_language = st.selectbox("Audio Language", 
                ["auto-detect", "en", "hi", "es", "fr", "de", "it", "pt", "ru", "ja", "ko"])
            
            if uploaded_audio is not None:
                if st.button("🎵 Process Audio", type="primary"):
                    with st.spinner("Processing audio (this may take several minutes)..."):
                        # Save uploaded file temporarily
                        temp_dir = Path("temp")
                        temp_dir.mkdir(exist_ok=True)
                        
                        audio_extension = uploaded_audio.name.split('.')[-1]
                        temp_audio_path = temp_dir / f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{audio_extension}"
                        
                        with open(temp_audio_path, "wb") as f:
                            f.write(uploaded_audio.getvalue())
                        
                        # Process audio
                        language = None if audio_language == "auto-detect" else audio_language
                        audio_results = rag.process_audio_file(str(temp_audio_path), language)
                        
                        if "error" in audio_results:
                            st.error(f"❌ {audio_results['error']}")
                        else:
                            st.success("✅ Audio processed successfully!")
                            
                            # Show brief results
                            if audio_results.get('summary'):
                                with st.expander("📝 Quick Summary"):
                                    st.write(audio_results['summary'])
                            
                            if audio_results.get('action_items'):
                                st.info(f"✅ Found {len(audio_results['action_items'])} action items")
                            
                            if audio_results.get('keywords'):
                                st.info(f"🔑 Extracted {len(audio_results['keywords'])} key topics")
                        
                        # Clean up temp file
                        try:
                            temp_audio_path.unlink()
                        except:
                            pass

        st.divider()
        
        # Quick Tools Section
        st.subheader("🚀 Quick Tools")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Analyze Content"):
                with st.spinner("Analyzing..."):
                    if not rag.latest_doc_path and not rag.audio_results:
                        st.warning("⚠️ No content to analyze. Upload a document or audio first.")
                    else:
                        # Simple analysis based on available content
                        content_type = []
                        if rag.latest_doc_path:
                            content_type.append("document")
                        if rag.audio_results and not rag.audio_results.get('error'):
                            content_type.append("audio")
                        
                        analysis_query = f"Analyze the available {' and '.join(content_type)} content and provide key insights, improvements, and recommendations."
                        outcome = rag.chat(analysis_query, st.session_state.session_id)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": outcome["response"], 
                            "sources": outcome["sources_used"]
                        })
        
        with col2:
            if st.button("🎵 Audio Insights"):
                with st.spinner("Extracting insights..."):
                    outcome = rag.get_audio_insights(st.session_state.session_id)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": outcome["response"], 
                        "sources": outcome["sources_used"]
                    })
        
        if st.button("🎧 Generate Podcast", type="secondary"):
            with st.spinner("Generating podcast..."):
                result = rag.generate_podcast_from_content(st.session_state.session_id)
                st.info(result)

        st.divider()
        
        # Processed Content Summary
        st.subheader("📚 Processed Content")
        if rag.store.documents:
            for path in list(rag.store.documents.keys())[:5]:
                if path.startswith("audio_"):
                    st.write(f"🎵 {path}")
                else:
                    st.write(f"📄 {Path(path).name}")
            
            if len(rag.store.documents) > 5:
                st.caption(f"... and {len(rag.store.documents) - 5} more")
        else:
            st.caption("No content processed yet.")

    # Main Chat Interface
    st.title("💬 Chat with MemoryPal")
    st.caption("Ask questions about your documents, get audio insights, or request analysis and improvements!")

    # Display chat messages
    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(message.get("content", ""))
            if message.get("sources"):
                with st.expander("📚 Sources Used"):
                    for s in message["sources"]:
                        st.write(s)

    # Chat input
    if prompt := st.chat_input("Ask anything about your content (documents, audio, improvements, analysis)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = rag.chat(prompt, st.session_state.session_id, use_web=False)
                st.markdown(res["response"])
                if res.get("sources_used"):
                    with st.expander("📚 Sources Used"):
                        for s in res["sources_used"]:
                            st.write(s)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": res["response"], 
                    "sources": res.get("sources_used", [])
                })

    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tips**: Upload documents for analysis • Process audio for transcription and insights • "
        "Ask for improvements and recommendations • Generate podcasts from your content"
    )

if __name__ == "__main__":
    create_streamlit_app()