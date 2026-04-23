# speech_processor_fixed.py - Fixed Speech Processing Library

from __future__ import annotations

import os
import re
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import warnings

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# NLTK setup with better error handling
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data with better error handling
def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            print("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            print("✅ NLTK data downloaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not download NLTK data: {e}")
            print("Using basic sentence splitting as fallback")

# Initialize NLTK
ensure_nltk_data()

from nltk.tokenize import sent_tokenize
import whisper
from tqdm import tqdm

# Import transformers with error handling
try:
    from transformers import pipeline, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

# Import other dependencies with fallbacks
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: spaCy not available")
    SPACY_AVAILABLE = False

try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: YAKE not available")
    YAKE_AVAILABLE = False

try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: Wikipedia not available")
    WIKIPEDIA_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: gTTS not available")
    GTTS_AVAILABLE = False

# Utility functions
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")

def save_json(path: Path, obj: Any):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def chunk_text(text: str, max_words: int = 800) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

def fallback_sentence_tokenize(text: str) -> List[str]:
    """Fallback sentence tokenizer if NLTK fails"""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

# Speech Processing Classes
class AudioTranscriber:
    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self.model = None
    
    def load_model(self):
        if self.model is None:
            try:
                print(f"[ASR] Loading Whisper model: {self.model_size}")
                self.model = whisper.load_model(self.model_size)
                print("✅ Whisper model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load Whisper model: {e}")
                raise
    
    def transcribe(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        self.load_model()
        print(f"[ASR] Transcribing: {audio_path}")
        try:
            # Add parameters to avoid FP16 warnings
            result = self.model.transcribe(
                str(audio_path), 
                language=language,
                fp16=False,  # Force FP32 to avoid warnings
                verbose=False
            )
            print("✅ Audio transcription completed")
            return result
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return {"text": "", "segments": []}

class TextSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        self.summarizer = None
        self.available = TRANSFORMERS_AVAILABLE
    
    def load_model(self):
        if not self.available:
            print("❌ Transformers not available for summarization")
            return False
            
        if self.summarizer is None:
            try:
                print(f"[SUM] Loading summarization model: {self.model_name}")
                self.summarizer = pipeline(
                    "summarization", 
                    model=self.model_name,
                    device=-1  # Force CPU usage
                )
                print("✅ Summarization model loaded successfully")
                return True
            except Exception as e:
                print(f"❌ Failed to load summarization model: {e}")
                self.available = False
                return False
        return True
    
    def summarize(self, text: str, max_chunk_words: int = 800,
                  max_length: int = 150, min_length: int = 50) -> str:
        if not text.strip():
            return ""
        
        if not self.available or not self.load_model():
            # Fallback: return first few sentences
            try:
                sentences = sent_tokenize(text)
            except:
                sentences = fallback_sentence_tokenize(text)
            
            return " ".join(sentences[:3]) if len(sentences) > 3 else text
        
        chunks = chunk_text(text, max_chunk_words)
        summaries = []
        
        print(f"[SUM] Summarizing in {len(chunks)} chunk(s)...")
        for ch in tqdm(chunks):
            try:
                # Adjust max_length based on input length
                input_length = len(ch.split())
                adjusted_max_length = min(max_length, max(min_length, input_length // 2))
                
                out = self.summarizer(
                    ch, 
                    max_length=adjusted_max_length, 
                    min_length=min_length, 
                    do_sample=False
                )
                summaries.append(out[0]['summary_text'].strip())
            except Exception as e:
                print(f"⚠️ Summarization warning: {e}")
                # Use first sentence as fallback
                try:
                    sentences = sent_tokenize(ch)
                    summaries.append(sentences[0] if sentences else ch[:200])
                except:
                    summaries.append(ch[:200])
        
        # If multiple chunks, create a meta-summary
        if len(summaries) > 1:
            joined = "\n".join(summaries)
            try:
                input_length = len(joined.split())
                adjusted_max_length = min(max_length, max(min_length, input_length // 2))
                
                meta = self.summarizer(
                    joined, 
                    max_length=adjusted_max_length, 
                    min_length=min_length, 
                    do_sample=False
                )
                return meta[0]['summary_text'].strip()
            except Exception as e:
                print(f"⚠️ Meta-summary warning: {e}")
                return summaries[0] if summaries else ""
        else:
            return summaries[0] if summaries else ""

class KeywordExtractor:
    def __init__(self):
        self.available = YAKE_AVAILABLE
        if self.available:
            self.extractor = yake.KeywordExtractor(lan="en", n=1, top=10)
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[Tuple[str, float]]:
        if not text.strip():
            return []
        
        if not self.available:
            # Simple fallback: extract most frequent words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [(word, 1.0/freq) for word, freq in sorted_words[:max_keywords]]
        
        try:
            return self.extractor.extract_keywords(text)[:max_keywords]
        except Exception as e:
            print(f"⚠️ Keyword extraction error: {e}")
            return []

class ActionItemDetector:
    ACTION_PATTERNS = [
        r"\bwe (?:will|should|need to|must|can)\b",
        r"\blet's\b",
        r"\baction item\b",
        r"\bto do\b|\btodo\b",
        r"\bdeadline\b|\bETA\b|\bby (?:Mon|Tue|Wed|Thu|Fri|Sat|Sun|\d{1,2}\/?\d{1,2})\b",
        r"\bfollow up\b|\bfollow-up\b",
        r"\bnext step\b|\bnext steps\b",
        r"\bassign\b|\bowner\b|\bresponsible\b",
    ]
    
    def detect_actions(self, text: str) -> List[str]:
        actions = []
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = fallback_sentence_tokenize(text)
        
        for sentence in sentences:
            if any(re.search(pat, sentence.lower()) for pat in self.ACTION_PATTERNS):
                actions.append(sentence.strip())
        
        return actions
    
    def score_segment(self, text: str, kw_extractor) -> float:
        if not text.strip():
            return 0.0
        
        # Base score from keywords
        keywords = kw_extractor.extract_keywords(text) if kw_extractor else []
        if not keywords:
            score = 1.0
        else:
            inv_scores = [1.0 / max(score, 1e-6) for _, score in keywords[:8]]
            score = float(sum(inv_scores))
        
        # Boost if action pattern detected
        if any(re.search(pat, text.lower()) for pat in self.ACTION_PATTERNS):
            score *= 1.5
        
        return score

class KeyMomentsExtractor:
    def __init__(self):
        self.action_detector = ActionItemDetector()
        self.kw_extractor = KeywordExtractor() if YAKE_AVAILABLE else None
    
    def extract_key_moments(self, segments: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        scored = []
        for seg in segments:
            text = seg.get('text', '')
            score = self.action_detector.score_segment(text, self.kw_extractor)
            scored.append((score, seg))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [dict(score=float(s), start=float(seg['start']), end=float(seg['end']), text=seg['text'])
                for s, seg in scored[:top_k]]

class MindMapGenerator:
    def __init__(self):
        self.available = SPACY_AVAILABLE
        if self.available:
            self.nlp = self._load_spacy_model()
    
    def _load_spacy_model(self):
        try:
            import spacy
            return spacy.load("en_core_web_sm")
        except OSError:
            try:
                spacy.cli.download("en_core_web_sm")
                return spacy.load("en_core_web_sm")
            except Exception as e:
                print(f"⚠️ Could not load spaCy model: {e}")
                self.available = False
                return None
    
    def generate_mindmap(self, text: str, language: str = "en") -> Dict[str, Any]:
        if not self.available or not self.nlp:
            # Simple fallback mindmap
            return self._create_simple_mindmap(text)
        
        try:
            doc = self.nlp(text)
            nodes = {}
            edges = []
            
            def add_node(label: str):
                if not label:
                    return None
                key = label.lower()
                if key not in nodes:
                    nodes[key] = {"id": key, "label": label}
                return key
            
            sentences = list(doc.sents)
            for sent in sentences:
                noun_chunks = [nc.text.strip() for nc in sent.noun_chunks]
                verbs = [t.lemma_ for t in sent if t.pos_ == "VERB"]
                if len(noun_chunks) >= 2:
                    head = add_node(noun_chunks[0])
                    for nc in noun_chunks[1:]:
                        tail = add_node(nc)
                        rel = verbs[0] if verbs else "related-to"
                        if head and tail:
                            edges.append({"source": head, "target": tail, "relation": rel})
            
            graph = {"nodes": list(nodes.values()), "edges": edges}
            mermaid = self._generate_mermaid(graph)
            return {"graph": graph, "mermaid": mermaid}
            
        except Exception as e:
            print(f"⚠️ Mindmap generation error: {e}")
            return self._create_simple_mindmap(text)
    
    def _create_simple_mindmap(self, text: str) -> Dict[str, Any]:
        """Simple fallback mindmap creation"""
        words = re.findall(r'\b[A-Z][a-z]+\b', text)  # Extract capitalized words
        unique_words = list(set(words))[:10]  # Take first 10 unique
        
        nodes = [{"id": word.lower(), "label": word} for word in unique_words]
        edges = []
        
        # Simple connections between consecutive words
        for i in range(len(unique_words) - 1):
            edges.append({
                "source": unique_words[i].lower(),
                "target": unique_words[i+1].lower(),
                "relation": "connects-to"
            })
        
        graph = {"nodes": nodes, "edges": edges}
        mermaid = self._generate_mermaid(graph)
        return {"graph": graph, "mermaid": mermaid}
    
    def _generate_mermaid(self, graph: Dict) -> str:
        """Generate Mermaid mindmap syntax"""
        root = graph["nodes"][0]["label"] if graph["nodes"] else "Conversation"
        mermaid_lines = ["mindmap", f"  root){root}"]
        
        adjacency = {}
        for e in graph["edges"]:
            adjacency.setdefault(e["source"], set()).add(e["target"])
        
        for src, targets in list(adjacency.items())[:5]:  # Limit to avoid clutter
            src_label = next((n["label"] for n in graph["nodes"] if n["id"] == src), src)
            mermaid_lines.append(f"    {src_label}")
            for t in list(targets)[:3]:  # Limit connections per node
                t_label = next((n["label"] for n in graph["nodes"] if n["id"] == t), t)
                mermaid_lines.append(f"      {t_label}")
        
        return "\n".join(mermaid_lines)

class DeepResearcher:
    def __init__(self):
        self.keyword_extractor = KeywordExtractor()
        self.available = WIKIPEDIA_AVAILABLE
    
    def research_keywords(self, text: str, max_items: int = 5) -> str:
        """Return a markdown snippet with short Wikipedia summaries for top keywords."""
        if not self.available:
            return "# Deep Research\n\nWikipedia integration not available.\n"
        
        keywords = [kw for kw, score in self.keyword_extractor.extract_keywords(text, max_items)]
        
        md = ["# Deep Research", "", "> Auto-collected context from Wikipedia for top topics.", ""]
        for kw in keywords:
            try:
                wikipedia.set_lang("en")
                page = wikipedia.page(kw, auto_suggest=True, redirect=True)
                summary = wikipedia.summary(kw, sentences=3, auto_suggest=True, redirect=True)
                md.append(f"## {page.title}")
                md.append(f"Source: {page.url}")
                md.append("")
                md.append(summary)
                md.append("")
            except Exception as e:
                md.append(f"## {kw}")
                md.append(f"Research not available: {str(e)[:100]}...\n")
        
        return "\n".join(md)

class TextToSpeechGenerator:
    def __init__(self):
        self.available = GTTS_AVAILABLE
    
    def generate_audio(self, text: str, output_path: str, language: str = "en"):
        if not self.available:
            print("❌ gTTS not available for text-to-speech")
            return False
            
        if not text.strip():
            print("⚠️ No text provided for TTS")
            return False
            
        try:
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(output_path)
            print(f"✅ Audio saved to {output_path}")
            return True
        except Exception as e:
            print(f"❌ TTS generation failed: {e}")
            return False

# Main Speech Processor Class
class SpeechProcessor:
    def __init__(self, whisper_size: str = "small", summary_model: str = "facebook/bart-large-cnn"):
        self.transcriber = AudioTranscriber(whisper_size)
        self.summarizer = TextSummarizer(summary_model)
        self.keyword_extractor = KeywordExtractor()
        self.action_detector = ActionItemDetector()
        self.moments_extractor = KeyMomentsExtractor()
        self.mindmap_generator = MindMapGenerator()
        self.researcher = DeepResearcher()
        self.tts_generator = TextToSpeechGenerator()
    
    def process_audio(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """Complete audio processing pipeline"""
        results = {}
        print(f"🎵 Processing audio file: {audio_path}")
        
        try:
            # 1. Transcribe
            transcript_result = self.transcriber.transcribe(audio_path, language)
            transcript_text = transcript_result.get('text', '').strip()
            segments = transcript_result.get('segments', [])
            
            results['transcript'] = {
                'text': transcript_text,
                'segments': segments,
                'full_result': transcript_result
            }
            
            if not transcript_text:
                print("⚠️ No transcript text generated")
                return results
            
            # 2. Summarize
            summary = self.summarizer.summarize(transcript_text)
            results['summary'] = summary
            
            # 3. Extract keywords
            keywords = self.keyword_extractor.extract_keywords(transcript_text)
            results['keywords'] = keywords
            
            # 4. Detect action items
            actions = self.action_detector.detect_actions(transcript_text)
            results['action_items'] = actions
            
            # 5. Extract key moments
            key_moments = self.moments_extractor.extract_key_moments(segments)
            results['key_moments'] = key_moments
            
            # 6. Generate mindmap
            mindmap = self.mindmap_generator.generate_mindmap(transcript_text)
            results['mindmap'] = mindmap
            
            # 7. Deep research
            research = self.researcher.research_keywords(summary or transcript_text[:1000])
            results['research'] = research
            
            print("✅ Audio processing completed successfully")
            return results
            
        except Exception as e:
            print(f"❌ Audio processing failed: {e}")
            results['error'] = str(e)
            return results
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text without audio transcription"""
        results = {}
        
        # 1. Summarize
        summary = self.summarizer.summarize(text)
        results['summary'] = summary
        
        # 2. Extract keywords
        keywords = self.keyword_extractor.extract_keywords(text)
        results['keywords'] = keywords
        
        # 3. Detect action items
        actions = self.action_detector.detect_actions(text)
        results['action_items'] = actions
        
        # 4. Generate mindmap
        mindmap = self.mindmap_generator.generate_mindmap(text)
        results['mindmap'] = mindmap
        
        # 5. Deep research
        research = self.researcher.research_keywords(summary or text[:1000])
        results['research'] = research
        
        return results
    
    def generate_podcast(self, text: str, output_path: str, language: str = "en"):
        """Generate podcast audio from text"""
        return self.tts_generator.generate_audio(text, output_path, language)