import json
import os
import uuid
import time
import asyncio
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import argparse
from dotenv import load_dotenv

# Scientific computing and NLP
from scipy.spatial.distance import cosine, euclidean, cityblock, minkowski
try:
    from langchain_openai import AzureOpenAIEmbeddings
except ImportError as e:
    print(f"Warning: langchain_openai not found. Please run 'pip install langchain_openai'.")
    AzureOpenAIEmbeddings = None

# Asynchronous operations in a synchronous Dash context
import nest_asyncio
nest_asyncio.apply()

try:
    from agent import TravelAgent
except ImportError:
    print("Warning: Agent module not found. Please ensure 'agent.py' is in the same directory.")
    TravelAgent = None

# --- Logging Setup ---
APP_LOGS = []
class ListLogHandler(logging.Handler):
    def emit(self, record):
        APP_LOGS.insert(0, self.format(record))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    log_handler = ListLogHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)

# --- Core Data Structures and Classes ---
class SimilarityMethod(Enum):
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    MINKOWSKI = "minkowski"
    JACCARD = "jaccard"

class InfluenceType(Enum):
    DIRECT = "direct"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    LLM_DERIVED = "llm_derived"

@dataclass
class ToolInfluence:
    source_tool_idx: int
    target_tool_idx: int
    influence_type: InfluenceType
    influence_score: float
    influence_explanation: str

@dataclass
class PreComputedEmbeddings:
    input_embedding: np.ndarray
    output_embedding: np.ndarray
    input_rope_embedding: np.ndarray
    output_rope_embedding: np.ndarray
    timestamp: float

@dataclass
class ToolCall:
    tool_name: str
    tool_input: Any
    tool_output: Any
    timestamp: float
    duration: float
    tool_idx: int = -1
    influences_from: List[Dict] = field(default_factory=list)
    embeddings: Optional[PreComputedEmbeddings] = None

@dataclass
class InfluenceChain:
    tool_idx: int
    tool_name: str
    influence_score: float
    influence_type: str
    explanation: str
    level: int
    connected_tools: List[int] = field(default_factory=list)

class AdvancedRoPEEnhancer:
    def __init__(self, d_model: int = 3072):
        self.d_model = d_model

    def apply_rope(self, vector: np.ndarray, position: float, temperature: float = 10000.0) -> np.ndarray:
        if not isinstance(vector, np.ndarray) or vector.size == 0: 
            return np.array([])
        original_size = vector.shape[0]
        if original_size % 2 != 0: 
            vector = np.append(vector, 0)
        d_model = vector.shape[0]
        theta = position / (temperature ** (np.arange(0, d_model, 2, dtype=np.float32) / d_model))
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        v_pairs = vector.reshape(-1, 2)
        rotated_pairs = np.zeros_like(v_pairs)
        rotated_pairs[:, 0] = v_pairs[:, 0] * cos_theta - v_pairs[:, 1] * sin_theta
        rotated_pairs[:, 1] = v_pairs[:, 0] * sin_theta + v_pairs[:, 1] * cos_theta
        return rotated_pairs.flatten()[:original_size]

    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray, method: SimilarityMethod) -> float:
        if vec1.size == 0 or vec2.size == 0 or vec1.shape != vec2.shape: 
            return 0.0
        try:
            if method == SimilarityMethod.COSINE: 
                return max(0.0, 1 - cosine(vec1, vec2))
            if method == SimilarityMethod.DOT_PRODUCT: 
                return max(0.0, np.dot(vec1/np.linalg.norm(vec1), vec2/np.linalg.norm(vec2)))
            if method == SimilarityMethod.EUCLIDEAN: 
                return 1 / (1 + euclidean(vec1, vec2))
            if method == SimilarityMethod.MANHATTAN: 
                return 1 / (1 + cityblock(vec1, vec2))
            if method == SimilarityMethod.MINKOWSKI: 
                return 1 / (1 + minkowski(vec1, vec2, p=3))
            if method == SimilarityMethod.JACCARD:
                bin1=(vec1>np.percentile(vec1,75))
                bin2=(vec2>np.percentile(vec2,75))
                return np.sum(bin1&bin2)/np.sum(bin1|bin2) if np.sum(bin1|bin2)>0 else 0.0
        except Exception: 
            return 0.0
        return 0.0

    def get_combined_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        methods = [
            SimilarityMethod.COSINE,
            SimilarityMethod.DOT_PRODUCT,
            SimilarityMethod.EUCLIDEAN,
            SimilarityMethod.MANHATTAN
        ]
        scores = [self.compute_similarity(vec1, vec2, method) for method in methods]
        # Use weighted average with emphasis on cosine and dot product
        weights = [0.4, 0.3, 0.15, 0.15]
        return sum(score * weight for score, weight in zip(scores, weights))

class AdvancedCausalAnalyzer:
    def __init__(self):
        self.embeddings_model = None
        self.rope_enhancer = AdvancedRoPEEnhancer(d_model=3072)
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        if not AzureOpenAIEmbeddings: 
            return
        try:
            self.embeddings_model = AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("OPENAI_API_VERSION")
            )
            logger.info("Embeddings model initialized with 3072 dimensions for text-embedding-3-large.")
        except Exception as e: 
            logger.error(f"Failed to initialize embeddings: {e}")

    async def create_enhanced_session(self, response_text: str, tool_calls: List[ToolCall], query: str) -> Dict:
        """Create session with pre-computed RoPE embeddings for all tools"""
        logger.info(f"Creating enhanced session with {len(tool_calls)} tool calls")
        
        # Add tool indices
        for idx, tc in enumerate(tool_calls):
            tc.tool_idx = idx
        
        # Pre-compute embeddings and RoPE transformations for all tools
        await self._precompute_all_embeddings(tool_calls, query, response_text)
        
        # Calculate comprehensive influences using O(n²) approach focusing on output-to-output
        self._calculate_output_to_output_influences(tool_calls)
        
        logger.info("Session created with pre-computed embeddings and O(n²) output-to-output analysis")
        
        return {
            "tool_calls": [self._serialize_tool_call(tc) for tc in tool_calls],
            "response_text": response_text,
            "query": query,
            "embeddings_computed": True,
            "total_tools": len(tool_calls)
        }

    async def _precompute_all_embeddings(self, tool_calls: List[ToolCall], query: str, response_text: str):
        """Pre-compute embeddings and RoPE transformations for all tool inputs and outputs"""
        if not self.embeddings_model:
            logger.warning("Embeddings model not available - skipping pre-computation")
            return
        
        logger.info("Pre-computing embeddings for all tools...")
        
        # Prepare all texts for batch embedding
        all_texts = []
        text_indices = {}  # Map text position to (tool_idx, 'input'/'output')
        
        for idx, tc in enumerate(tool_calls):
            tool_input_str = to_string_for_pre(tc.tool_input)
            tool_output_str = to_string_for_pre(tc.tool_output)
            
            # Create comprehensive text representations
            input_text = f"Tool: {tc.tool_name} | Query Context: {query[:200]} | Input: {tool_input_str[:800]}"
            output_text = f"Tool: {tc.tool_name} | Query Context: {query[:200]} | Output: {tool_output_str[:800]}"
            
            text_indices[len(all_texts)] = (idx, 'input')
            all_texts.append(input_text)
            
            text_indices[len(all_texts)] = (idx, 'output')
            all_texts.append(output_text)
        
        # Also add the final response for LLM detection
        response_text_formatted = f"Final Response: {response_text[:1000]}"
        all_texts.append(response_text_formatted)
        response_idx = len(all_texts) - 1
        
        # Batch embed all texts
        try:
            all_embeddings = await self.embeddings_model.aembed_documents(all_texts)
            logger.info(f"Successfully embedded {len(all_embeddings)} texts")
            
            # Store response embedding for LLM detection
            response_embedding = np.array(all_embeddings[response_idx])
            response_rope = self.rope_enhancer.apply_rope(response_embedding, time.time())
            
            # Apply RoPE transformations and store embeddings
            for text_idx, embedding in enumerate(all_embeddings[:-1]):  # Exclude response embedding
                tool_idx, io_type = text_indices[text_idx]
                tc = tool_calls[tool_idx]
                
                if tc.embeddings is None:
                    tc.embeddings = PreComputedEmbeddings(
                        input_embedding=np.array([]),
                        output_embedding=np.array([]),
                        input_rope_embedding=np.array([]),
                        output_rope_embedding=np.array([]),
                        timestamp=tc.timestamp
                    )
                
                base_embedding = np.array(embedding)
                rope_embedding = self.rope_enhancer.apply_rope(base_embedding, tc.timestamp)
                
                if io_type == 'input':
                    tc.embeddings.input_embedding = base_embedding
                    tc.embeddings.input_rope_embedding = rope_embedding
                else:
                    tc.embeddings.output_embedding = base_embedding
                    tc.embeddings.output_rope_embedding = rope_embedding
                    
                    # Calculate LLM derivation score
                    llm_similarity = self.rope_enhancer.get_combined_similarity(
                        rope_embedding, response_rope
                    )
                    tc.llm_derivation_score = llm_similarity
                    
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")

    def _calculate_output_to_output_influences(self, tool_calls: List[ToolCall]):
        """Calculate influences using O(n²) output-to-output embedding similarity"""
        logger.info("Calculating O(n²) output-to-output influences...")
        
        n_tools = len(tool_calls)
        
        # Create similarity matrix for all tool outputs
        similarity_matrix = np.zeros((n_tools, n_tools))
        
        for i in range(n_tools):
            for j in range(n_tools):
                if i == j:
                    continue
                    
                tool_i, tool_j = tool_calls[i], tool_calls[j]
                
                if (tool_i.embeddings and tool_j.embeddings and 
                    tool_i.embeddings.output_rope_embedding.size > 0 and 
                    tool_j.embeddings.output_rope_embedding.size > 0):
                    
                    similarity = self.rope_enhancer.get_combined_similarity(
                        tool_i.embeddings.output_rope_embedding,
                        tool_j.embeddings.output_rope_embedding
                    )
                    similarity_matrix[i][j] = similarity
        
        # Build influences based on similarity and temporal ordering
        for target_idx in range(n_tools):
            target_tool = tool_calls[target_idx]
            
            for source_idx in range(n_tools):
                if source_idx >= target_idx:  # Only consider earlier tools
                    continue
                    
                source_tool = tool_calls[source_idx]
                output_similarity = similarity_matrix[source_idx][target_idx]
                
                # Enhanced similarity threshold (no artificial cap)
                if output_similarity > 0.2:
                    # Calculate temporal influence
                    temporal_score = self._calculate_enhanced_temporal_influence(source_tool, target_tool)
                    
                    # Combined influence score
                    combined_score = 0.7 * output_similarity + 0.3 * temporal_score
                    
                    if combined_score > 0.25:
                        influence_type = self._determine_influence_type(
                            source_tool, target_tool, output_similarity, temporal_score
                        )
                        
                        explanation = self._generate_influence_explanation(
                            source_tool, target_tool, influence_type, combined_score, 
                            output_similarity, temporal_score
                        )
                        
                        target_tool.influences_from.append({
                            'source_tool_idx': source_idx,
                            'target_tool_idx': target_idx,
                            'influence_type': influence_type.value,
                            'influence_score': combined_score,
                            'output_similarity': output_similarity,
                            'temporal_score': temporal_score,
                            'influence_explanation': explanation
                        })
        
        # Log influence statistics
        total_influences = sum(len(tc.influences_from) for tc in tool_calls)
        logger.info(f"Calculated {total_influences} total output-to-output influences across {n_tools} tools")

    def _determine_influence_type(self, source: ToolCall, target: ToolCall, 
                                 output_sim: float, temporal_score: float) -> InfluenceType:
        """Determine the type of influence based on various factors"""
        # Check for LLM derivation
        if hasattr(target, 'llm_derivation_score') and target.llm_derivation_score > 0.6:
            return InfluenceType.LLM_DERIVED
        
        # High output similarity suggests semantic relationship
        if output_sim > 0.7:
            return InfluenceType.SEMANTIC
        
        # High temporal score with moderate similarity suggests direct flow
        if temporal_score > 0.6 and output_sim > 0.4:
            return InfluenceType.DIRECT
        
        return InfluenceType.SEMANTIC

    def _generate_influence_explanation(self, source: ToolCall, target: ToolCall, 
                                       influence_type: InfluenceType, combined_score: float,
                                       output_sim: float, temporal_score: float) -> str:
        """Generate detailed explanation for the influence"""
        base_explanation = f"{source.tool_name} → {target.tool_name}"
        
        if influence_type == InfluenceType.LLM_DERIVED:
            return f"{base_explanation}: Output content derived by LLM from tool results (similarity: {output_sim:.3f})"
        elif influence_type == InfluenceType.DIRECT:
            return f"{base_explanation}: Direct data flow detected (output similarity: {output_sim:.3f}, temporal: {temporal_score:.3f})"
        elif influence_type == InfluenceType.SEMANTIC:
            return f"{base_explanation}: Semantic relationship in outputs (similarity: {output_sim:.3f})"
        else:
            return f"{base_explanation}: Temporal influence (score: {combined_score:.3f})"

    def _calculate_enhanced_temporal_influence(self, source: ToolCall, target: ToolCall) -> float:
        """Enhanced temporal influence calculation"""
        if target.timestamp <= source.timestamp:
            return 0.0
        
        time_diff = target.timestamp - source.timestamp
        
        # Very close temporal proximity (likely sequential)
        if time_diff < 1.0:
            return 0.9
        elif time_diff < 3.0:
            return 0.8
        elif time_diff < 10.0:
            return max(0.4, 0.8 - (time_diff / 25.0))
        elif time_diff < 30.0:
            return max(0.2, 0.5 - (time_diff / 60.0))
        
        return 0.1

    async def analyze_selection_with_precomputed(self, selected_text: str, session_data: Dict) -> Dict:
        """Analyze selection using pre-computed embeddings with improved scoring"""
        if not session_data.get('embeddings_computed'):
            return {"error": "No pre-computed embeddings available"}
        
        tool_calls = [self._deserialize_tool_call(tc_data) for tc_data in session_data['tool_calls']]
        query = session_data['query']
        
        logger.info(f"Analyzing selection: '{selected_text[:50]}...' using pre-computed embeddings")
        
        # Find best matching tools using pre-computed embeddings (no artificial caps)
        best_matches = await self._find_best_matches_enhanced(selected_text, tool_calls, query)
        
        if not best_matches:
            return {"error": "No tool matches found", "selected_text": selected_text}
        
        # Build enhanced influence chains
        influence_chains = self._build_output_based_influence_chains(best_matches, tool_calls)
        
        # Detect LLM-generated vs tool-derived content
        content_analysis = self._analyze_content_origin(selected_text, tool_calls, query)
        
        return {
            "selected_text": selected_text,
            "best_matches": best_matches,
            "influence_chains": influence_chains,
            "content_analysis": content_analysis,
            "analysis_type": "enhanced_precomputed_embeddings"
        }

    async def _find_best_matches_enhanced(self, selected_text: str, 
                                        tool_calls: List[ToolCall], query: str) -> List[Dict]:
        """Find best tool matches with enhanced scoring (no artificial caps)"""
        if not self.embeddings_model:
            return []
        
        # Embed the selected text
        cleaned_selection = self._clean_text_for_analysis(selected_text)
        selection_text = f"Query: {query} | Selected text: {cleaned_selection}"
        
        try:
            selection_embeddings = await self.embeddings_model.aembed_documents([selection_text])
            selection_embedding = np.array(selection_embeddings[0])
            selection_rope = self.rope_enhancer.apply_rope(selection_embedding, time.time())
        except Exception as e:
            logger.error(f"Error embedding selection: {e}")
            return []
        
        tool_matches = []
        
        for tc in tool_calls:
            if not tc.embeddings:
                continue
            
            # Calculate enhanced similarities with pre-computed embeddings
            input_rope_sim = self.rope_enhancer.get_combined_similarity(
                selection_rope, tc.embeddings.input_rope_embedding
            )
            output_rope_sim = self.rope_enhancer.get_combined_similarity(
                selection_rope, tc.embeddings.output_rope_embedding
            )
            
            # Also calculate with non-RoPE embeddings
            input_base_sim = self.rope_enhancer.get_combined_similarity(
                selection_embedding, tc.embeddings.input_embedding
            )
            output_base_sim = self.rope_enhancer.get_combined_similarity(
                selection_embedding, tc.embeddings.output_embedding
            )
            
            # Enhanced scoring without artificial caps
            rope_score = max(input_rope_sim, output_rope_sim)
            base_score = max(input_base_sim, output_base_sim)
            
            # Weighted combination with emphasis on RoPE
            combined_score = 0.8 * rope_score + 0.2 * base_score
            
            # Text-based overlap detection
            tool_output_str = to_string_for_pre(tc.tool_output)
            text_overlap_score = self._calculate_text_overlap_score(cleaned_selection, tool_output_str)
            
            # Final score - take the maximum to avoid suppressing high similarities
            final_score = max(combined_score, text_overlap_score * 0.9)
            
            # Lower threshold to capture more matches
            if final_score > 0.08:
                # Use actual execution order (tool_idx + 1) for display
                tool_number = tc.tool_idx + 1
                tool_matches.append({
                    "tool_idx": tc.tool_idx,
                    "tool_number": tool_number,
                    "tool_name": tc.tool_name,
                    "similarity_score": final_score,
                    "rope_input_sim": input_rope_sim,
                    "rope_output_sim": output_rope_sim,
                    "base_input_sim": input_base_sim,
                    "base_output_sim": output_base_sim,
                    "text_overlap_score": text_overlap_score,
                    "tool_output": tool_output_str,
                    "tool_input": to_string_for_pre(tc.tool_input),
                    # Use proper sequential numbering
                    "debug_trace_ref": f"Tool #{tool_number}: {tc.tool_name}"
                })
        
        # Sort and return matches (no artificial limit)
        tool_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        logger.info(f"Found {len(tool_matches)} enhanced tool matches")
        
        return tool_matches[:5]  # Return top 5 matches

    def _calculate_text_overlap_score(self, selection: str, tool_output: str) -> float:
        """Enhanced text overlap calculation"""
        selection_clean = re.sub(r'\W+', ' ', selection.lower()).strip()
        output_clean = re.sub(r'\W+', ' ', tool_output.lower()).strip()
        
        if not selection_clean or not output_clean:
            return 0.0
        
        # Exact substring match (high score)
        if len(selection_clean) > 10 and selection_clean in output_clean:
            return 0.95
        
        # Phrase matching
        selection_words = selection_clean.split()
        phrase_matches = 0
        total_phrases = 0
        
        if len(selection_words) >= 3:
            for i in range(len(selection_words) - 2):
                phrase = " ".join(selection_words[i:i+3])
                total_phrases += 1
                if phrase in output_clean:
                    phrase_matches += 1
        
        if total_phrases > 0:
            phrase_score = phrase_matches / total_phrases
            if phrase_score > 0.5:
                return min(0.9, phrase_score)
        
        # Word overlap
        sel_words = set(selection_words)
        out_words = set(output_clean.split())
        overlap = len(sel_words.intersection(out_words))
        
        if len(sel_words) > 0:
            overlap_ratio = overlap / len(sel_words)
            if overlap_ratio > 0.6:
                return min(0.8, overlap_ratio)
        
        return 0.0

    def _build_output_based_influence_chains(self, best_matches: List[Dict], 
                                           tool_calls: List[ToolCall]) -> List[Dict]:
        """Build influence chains based on output-to-output similarities"""
        logger.info(f"Building output-based influence chains for {len(best_matches)} matches")
        
        influence_chains = []
        
        for match in best_matches:
            tool_idx = match['tool_idx']
            chain = self._build_output_chain_recursively(tool_idx, tool_calls, level=0, max_depth=4)
            
            if chain:
                chain_depth = self._count_chain_depth(chain)
                
                influence_chains.append({
                    "root_tool": match,
                    "chain": self._serialize_influence_chain(chain),
                    "chain_length": chain_depth,
                    "chain_type": "output_based"
                })
        
        logger.info(f"Built {len(influence_chains)} output-based influence chains")
        return influence_chains

    def _build_output_chain_recursively(self, tool_idx: int, tool_calls: List[ToolCall], 
                                      level: int, max_depth: int, visited: set = None) -> Optional[InfluenceChain]:
        """Build chain focusing on output-to-output relationships"""
        if visited is None:
            visited = set()
        
        if level >= max_depth or tool_idx >= len(tool_calls) or tool_idx in visited:
            return None
        
        visited.add(tool_idx)
        current_tool = tool_calls[tool_idx]
        
        # Create chain node
        chain_node = InfluenceChain(
            tool_idx=tool_idx,
            tool_name=current_tool.tool_name,
            influence_score=1.0 if level == 0 else max(0.2, 0.85 ** level),
            influence_type="root" if level == 0 else "output_derived",
            explanation=self._get_output_level_explanation(current_tool, level),
            level=level
        )
        
        # Find strongest output-based influence
        strongest_influence = None
        max_output_sim = 0.0
        
        for influence in current_tool.influences_from:
            output_sim = influence.get('output_similarity', 0.0)
            if output_sim > max_output_sim:
                max_output_sim = output_sim
                strongest_influence = influence
        
        if strongest_influence and level < max_depth - 1:
            parent_tool_idx = strongest_influence['source_tool_idx']
            
            if parent_tool_idx not in visited and parent_tool_idx < len(tool_calls):
                parent_chain = self._build_output_chain_recursively(
                    parent_tool_idx, tool_calls, level + 1, max_depth, visited.copy()
                )
                
                if parent_chain:
                    chain_node.connected_tools = [parent_tool_idx]
                    parent_tool_name = tool_calls[parent_tool_idx].tool_name
                    chain_node.explanation = f"Level {level}: Output influenced by {parent_tool_name} (similarity: {max_output_sim:.3f})"
                    setattr(chain_node, 'parent_chain', parent_chain)
        
        return chain_node

    def _analyze_content_origin(self, selected_text: str, tool_calls: List[ToolCall], query: str) -> Dict:
        """Analyze whether content is LLM-generated or tool-derived"""
        content_analysis = {
            "is_llm_generated": False,
            "tool_derived_confidence": 0.0,
            "llm_generated_confidence": 0.0,
            "source_analysis": []
        }
        
        # Check for high tool-derived confidence
        max_tool_confidence = 0.0
        for tc in tool_calls:
            if hasattr(tc, 'llm_derivation_score'):
                tool_confidence = 1.0 - tc.llm_derivation_score  # Inverse for tool derivation
                max_tool_confidence = max(max_tool_confidence, tool_confidence)
                
                content_analysis["source_analysis"].append({
                    "tool_name": tc.tool_name,
                    "tool_idx": tc.tool_idx,
                    "tool_derivation_confidence": tool_confidence,
                    "llm_derivation_confidence": tc.llm_derivation_score
                })
        
        content_analysis["tool_derived_confidence"] = max_tool_confidence
        content_analysis["llm_generated_confidence"] = 1.0 - max_tool_confidence
        content_analysis["is_llm_generated"] = max_tool_confidence < 0.6
        
        return content_analysis

    def _get_output_level_explanation(self, tool: ToolCall, level: int) -> str:
        """Generate explanation focused on output relationships"""
        if level == 0:
            return f"Primary match: {tool.tool_name} output directly matches selected content"
        elif level == 1:
            return f"Output influence: {tool.tool_name} output semantically influenced the primary match"
        else:
            return f"Level {level} output chain: {tool.tool_name} contributed to the semantic chain"

    # Utility methods remain the same...
    def _count_chain_depth(self, chain: Optional[InfluenceChain]) -> int:
        if not chain:
            return 0
        depth = 1
        current = chain
        while hasattr(current, 'parent_chain') and current.parent_chain:
            depth += 1
            current = current.parent_chain
        return depth

    def _serialize_influence_chain(self, chain: Optional[InfluenceChain]) -> Optional[Dict]:
        if not chain:
            return None
        
        result = {
            "tool_idx": chain.tool_idx,
            "tool_name": chain.tool_name,
            "influence_score": chain.influence_score,
            "influence_type": chain.influence_type,
            "explanation": chain.explanation,
            "level": chain.level,
            "connected_tools": chain.connected_tools
        }
        
        if hasattr(chain, 'parent_chain') and chain.parent_chain:
            result["parent_chain"] = self._serialize_influence_chain(chain.parent_chain)
        
        return result

    def _clean_text_for_analysis(self, text: str) -> str:
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _serialize_tool_call(self, tc: ToolCall) -> Dict:
        embeddings_data = None
        if tc.embeddings:
            embeddings_data = {
                "input_embedding": tc.embeddings.input_embedding.tolist() if tc.embeddings.input_embedding.size > 0 else [],
                "output_embedding": tc.embeddings.output_embedding.tolist() if tc.embeddings.output_embedding.size > 0 else [],
                "input_rope_embedding": tc.embeddings.input_rope_embedding.tolist() if tc.embeddings.input_rope_embedding.size > 0 else [],
                "output_rope_embedding": tc.embeddings.output_rope_embedding.tolist() if tc.embeddings.output_rope_embedding.size > 0 else [],
                "timestamp": tc.embeddings.timestamp
            }
        
        return {
            "tool_name": tc.tool_name,
            "tool_input": tc.tool_input,
            "tool_output": tc.tool_output,
            "timestamp": tc.timestamp,
            "duration": tc.duration,
            "tool_idx": tc.tool_idx,
            "influences_from": tc.influences_from,
            "embeddings": embeddings_data,
            "llm_derivation_score": getattr(tc, 'llm_derivation_score', 0.0)
        }

    def _deserialize_tool_call(self, tc_data: Dict) -> ToolCall:
        embeddings = None
        if tc_data.get("embeddings"):
            emb_data = tc_data["embeddings"]
            embeddings = PreComputedEmbeddings(
                input_embedding=np.array(emb_data["input_embedding"]) if emb_data["input_embedding"] else np.array([]),
                output_embedding=np.array(emb_data["output_embedding"]) if emb_data["output_embedding"] else np.array([]),
                input_rope_embedding=np.array(emb_data["input_rope_embedding"]) if emb_data["input_rope_embedding"] else np.array([]),
                output_rope_embedding=np.array(emb_data["output_rope_embedding"]) if emb_data["output_rope_embedding"] else np.array([]),
                timestamp=emb_data["timestamp"]
            )
        
        tc = ToolCall(
            tool_name=tc_data["tool_name"],
            tool_input=tc_data["tool_input"],
            tool_output=tc_data["tool_output"],
            timestamp=tc_data["timestamp"],
            duration=tc_data["duration"],
            tool_idx=tc_data.get("tool_idx", -1),
            influences_from=tc_data.get("influences_from", []),
            embeddings=embeddings
        )
        
        tc.llm_derivation_score = tc_data.get("llm_derivation_score", 0.0)
        return tc

# Initialize the analyzer
analyzer = AdvancedCausalAnalyzer()

# --- Enhanced Dash App Layout with Draggable Sidebar ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE], suppress_callback_exceptions=True)

# Add custom CSS for resizable sidebar
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
        .resizer {
            width: 5px;
            background: #444;
            cursor: col-resize;
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            z-index: 1000;
        }
        .resizer:hover {
            background: #007bff;
        }
        .analysis-sidebar-container {
            position: relative;
            transition: width 0.3s ease;
        }
        .analysis-content {
            padding-left: 10px;
        }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = dbc.Container([
    dcc.Store(id="session-store", data={}),
    dcc.Store(id="current-session-id"),
    dcc.Store(id="analysis-trigger-store"),
    dcc.Store(id="last-processed-timestamp-store", data=None),
    dcc.Store(id="sidebar-width-store", data=300),  # Store for analysis sidebar width in pixels
    dcc.Interval(id='selection-poller', interval=500),
    
    # Floating analyze button
    html.Div(id='floating-button-container', children=[
        dbc.Button("🔍 Analyze Selection", id='floating-analyze-btn', size='sm', color="info")
    ], style={'position': 'absolute', 'zIndex': 1000, 'display': 'none'}),
    
    # Header
    html.Header(className="text-center my-4", children=[
        html.H2("🧠 Enhanced Travel Agent with RoPE & Chain Analysis"),
        dbc.Badge("3072D Embeddings + Enhanced O(n²) Analysis", color="success")
    ]),
    
    # Main layout with resizable sidebar
    dbc.Row([
        # Sessions sidebar
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("📚 Sessions", className="mb-0"),
                    dbc.Button("⟵", id="collapse-sidebar-btn", size="sm", color="secondary", className="ms-auto")
                ], className="d-flex align-items-center"), 
                dbc.CardBody(id="session-list", style={"height": "75vh", "overflowY": "auto"})
            ])
        ], id="sidebar-col", width=3, style={"transition": "all 0.3s ease"}),
        
        # Main content area
        dbc.Col([
            # Query input card
            dbc.Card(dbc.CardBody([
                dbc.Textarea(id="query-input", placeholder="Type your travel query...", rows=4, className="mb-2"),
                dbc.Row([
                    dbc.Col(dcc.Dropdown(
                        id="model-dropdown", 
                        options=[{"label": m, "value": f"openai:{m}"} for m in ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]], 
                        value="openai:gpt-4o"
                    ), width=9),
                    dbc.Col(dbc.Button("Send", id="run-button", className="w-100", color="primary"), width=3),
                ]),
            ])),
            
            # Loading and main content with tabs
            dcc.Loading(id="loading-main", children=[
                html.Div(id="response-tabs-container", className="mt-3")
            ]),
        ], id="main-content-col", width="auto", style={"flex": "1"}),
        
        # Analysis sidebar (resizable)
        html.Div([
            # Resizer handle
            html.Div(className="resizer", id="analysis-resizer"),
            
            # Analysis content
            dbc.Collapse([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("⛓️ Enhanced Influence Analysis", className="mb-0"),
                        dbc.Badge("RoPE + O(n²) + LLM Detection", color="success", className="ms-2"),
                        dbc.Button("✕", id="close-analysis-btn", size="sm", color="outline-secondary", className="ms-auto")
                    ], className="d-flex align-items-center"),
                    dbc.CardBody(
                        dcc.Loading(html.Div(id="causal-analysis-content"), className="analysis-content"), 
                        style={"height": "75vh", "overflowY": "auto"}
                    )
                ])
            ], id="analysis-sidebar", is_open=False)
        ], id="analysis-sidebar-container", className="analysis-sidebar-container", 
           style={"width": "300px", "position": "relative"})
    ], className="flex-fill", style={"display": "flex"}),
    
    # Debug logs
    dbc.Row(dbc.Col(dbc.Accordion([
        dbc.AccordionItem(
            dcc.Loading(html.Pre(id='log-viewer', style={
                'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all', 'maxHeight': '300px', 'overflowY': 'auto'
            })), 
            title="📋 Debug Logs & System Status"
        )
    ], start_collapsed=True), className="mt-3"))
], fluid=True, className="vh-100 d-flex flex-column")

# --- Clientside Callbacks for UI Interactions ---
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks === 0 || n_clicks === null) {
            return dash_clientside.no_update;
        }
        const selection = window.getSelection();
        const selectionText = selection.toString().trim();
        if (selectionText) {
            return {text: selectionText, timestamp: Date.now()};
        }
        return dash_clientside.no_update;
    }
    """,
    Output("analysis-trigger-store", "data"),
    Input("floating-analyze-btn", "n_clicks")
)

app.clientside_callback(
    """
    function(n_intervals) {
        const selection = window.getSelection();
        const selectionText = selection.toString().trim();
        const targetNode = document.getElementById('final-response-text');
        
        if (selectionText && targetNode && selection.anchorNode && targetNode.contains(selection.anchorNode)) {
            const range = selection.getRangeAt(0);
            const rect = range.getBoundingClientRect();
            const top = rect.bottom + window.scrollY + 5;
            const left = rect.right + window.scrollX + 5;
            
            return {
                'position': 'absolute', 
                'zIndex': 1000, 
                'display': 'block', 
                'top': `${top}px`, 
                'left': `${left}px`
            };
        }
        return {'display': 'none'};
    }
    """,
    Output("floating-button-container", "style"),
    Input("selection-poller", "n_intervals")
)

# Resizable sidebar functionality
app.clientside_callback(
    """
    function() {
        let isResizing = false;
        let startX = 0;
        let startWidth = 0;
        
        const resizer = document.getElementById('analysis-resizer');
        const sidebar = document.getElementById('analysis-sidebar-container');
        
        if (!resizer || !sidebar) {
            return dash_clientside.no_update;
        }
        
        resizer.addEventListener('mousedown', function(e) {
            isResizing = true;
            startX = e.clientX;
            startWidth = parseInt(window.getComputedStyle(sidebar).width, 10);
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', stopResize);
            e.preventDefault();
        });
        
        function handleMouseMove(e) {
            if (!isResizing) return;
            const width = startWidth - (e.clientX - startX);
            if (width >= 250 && width <= 600) {
                sidebar.style.width = width + 'px';
            }
        }
        
        function stopResize() {
            isResizing = false;
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', stopResize);
        }
        
        return dash_clientside.no_update;
    }
    """,
    Output("sidebar-width-store", "data", allow_duplicate=True),
    Input("analysis-sidebar", "is_open"),
    prevent_initial_call=True
)

# Sidebar collapse functionality
@app.callback(
    [Output("sidebar-col", "width"), Output("main-content-col", "style")],
    Input("collapse-sidebar-btn", "n_clicks"),
    State("sidebar-col", "width")
)
def toggle_sidebar(n_clicks, current_width):
    if not n_clicks:
        raise PreventUpdate
    
    if current_width == 3:
        return 1, {"flex": "1", "marginLeft": "10px"}  # Collapsed
    else:
        return 3, {"flex": "1"}  # Expanded

# Close analysis sidebar
@app.callback(
    Output("analysis-sidebar", "is_open", allow_duplicate=True),
    Input("close-analysis-btn", "n_clicks"),
    prevent_initial_call=True
)
def close_analysis_sidebar(n_clicks):
    if n_clicks:
        return False
    raise PreventUpdate

# --- Main Server-side Callbacks ---
@app.callback(
    [Output("response-tabs-container", "children"), 
     Output("session-store", "data"), 
     Output("current-session-id", "data")],
    Input("run-button", "n_clicks"),
    [State("query-input", "value"), 
     State("model-dropdown", "value"), 
     State("session-store", "data")]
)
def run_agent_and_create_session(n_clicks, query, model, sessions):
    if not n_clicks or not query: 
        raise PreventUpdate
        
    logger.info(f"Running agent for query: '{query}'")
    sessions = sessions or {}
    
    try:
        travel_agent = TravelAgent()
        resp, trace = travel_agent.run(query, model=model)
        
        if isinstance(trace, bytes): 
            trace = json.loads(trace.decode('utf-8'))
        
        tool_calls = extract_tool_calls_from_trace(trace or [])
        
        # Create enhanced session with pre-computed embeddings
        analysis_results = asyncio.run(
            analyzer.create_enhanced_session(resp, tool_calls, query)
        )

        session_id = str(uuid.uuid4())[:8]
        sessions[session_id] = {
            "id": session_id, 
            "query": query, 
            "response": resp, 
            "model": model,
            "timestamp": datetime.now().isoformat(), 
            "analysis": analysis_results,
            "trace": ensure_json_serializable(trace)
        }
        
        logger.info(f"Enhanced session {session_id} created with {len(tool_calls)} tool calls")
        layout = create_tabbed_response_layout(resp, sessions[session_id]['trace'])
        
        return layout, sessions, session_id
        
    except Exception as e:
        logger.exception("Error running agent.")
        return dbc.Alert(f"Agent Error: {e}", color="danger"), sessions, no_update

@app.callback(
    [Output("causal-analysis-content", "children"), 
     Output("analysis-sidebar", "is_open"), 
     Output("last-processed-timestamp-store", "data")],
    Input("analysis-trigger-store", "data"),
    [State("current-session-id", "data"), 
     State("session-store", "data"), 
     State("last-processed-timestamp-store", "data")],
)
def display_enhanced_analysis(selection_data, session_id, sessions, last_timestamp):
    if not selection_data or not selection_data.get('text') or not session_id:
        raise PreventUpdate
    
    current_timestamp = selection_data['timestamp']
    if current_timestamp == last_timestamp:
        raise PreventUpdate
    
    session = sessions.get(session_id, {})
    analysis_data = session.get('analysis', {})
    selected_text = selection_data['text']

    if not analysis_data or not analysis_data.get('embeddings_computed'):
        return dbc.Alert("No pre-computed embeddings for this session.", color="warning"), True, current_timestamp
        
    logger.info(f"Enhanced analysis for: '{selected_text[:50]}...'")
    
    try:
        detailed_analysis = asyncio.run(
            analyzer.analyze_selection_with_precomputed(selected_text, analysis_data)
        )
        
        if "error" in detailed_analysis:
            error_msg = detailed_analysis["error"]
            return dbc.Alert(f"Analysis Error: {error_msg}", color="warning"), True, current_timestamp
        
        analysis_layout = create_enhanced_analysis_layout(detailed_analysis)
        return analysis_layout, True, current_timestamp
        
    except Exception as e:
        logger.exception("Error in enhanced analysis")
        return dbc.Alert(f"Analysis failed: {str(e)}", color="danger"), True, current_timestamp

@app.callback(
    Output("session-list", "children"),
    Input("session-store", "data")
)
def update_session_list(sessions):
    if not sessions: 
        return [html.P("No sessions yet.", className="text-muted text-center p-3")]
    
    sorted_sessions = sorted(sessions.values(), key=lambda x: x['timestamp'], reverse=True)
    
    session_cards = []
    for s in sorted_sessions:
        try:
            total_tools = s.get('analysis', {}).get('total_tools', 0)
            embeddings_computed = s.get('analysis', {}).get('embeddings_computed', False)
            
            card = dbc.Card(dbc.CardBody([
                html.H6(s['query'][:40] + "...", className="card-title"),
                html.P([
                    f"ID: {s['id']} | Tools: {total_tools} | ",
                    dbc.Badge("✓", color="success" if embeddings_computed else "secondary", size="sm")
                ], className="card-text small text-muted")
            ]), 
            id={"type": "session-card", "index": s['id']}, 
            className="mb-2", 
            style={'cursor': 'pointer', 'transition': 'all 0.2s ease'})
            
            session_cards.append(card)
        except Exception as e:
            logger.error(f"Error creating session card: {e}")
            continue
    
    return session_cards if session_cards else [html.P("No valid sessions found.", className="text-muted text-center p-3")]

@app.callback(
    [Output("response-tabs-container", "children", allow_duplicate=True),
     Output("current-session-id", "data", allow_duplicate=True)],
    Input({"type": "session-card", "index": ALL}, "n_clicks"),
    State("session-store", "data"),
    prevent_initial_call=True
)
def load_session(n_clicks, sessions):
    ctx = callback_context
    if not ctx.triggered or not any(n_clicks): 
        raise PreventUpdate
        
    session_id = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])['index']
    session = sessions.get(session_id)
    
    if not session: 
        return dbc.Alert("Session not found.", color="danger"), no_update
        
    logger.info(f"Loading session {session_id}")
    return create_tabbed_response_layout(session['response'], session['trace']), session_id

@app.callback(
    Output("log-viewer", "children"), 
    Input("analysis-trigger-store", "data"), 
    prevent_initial_call=True
)
def update_log_viewer(_): 
    return "\n".join(APP_LOGS[:50])  # Show last 50 log entries

# --- Enhanced Layout Creation Functions ---

def create_tabbed_response_layout(response: str, trace: List[Dict]) -> dbc.Card:
    """Create tabbed layout separating response and traces"""
    trace_parts = extract_trace_parts(trace)
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("🤖 Agent Response & Execution", className="mb-0"),
            dbc.Badge(f"{len(trace_parts)} trace items", color="info", className="ms-2")
        ]),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab([
                    html.Div([
                        dcc.Markdown(response, className="p-3 bg-dark rounded", id="final-response-text")
                    ], className="mt-3")
                ], label="🗨️ Response", tab_id="response-tab"),
                
                dbc.Tab([
                    html.Div([
                        create_grouped_trace_items(trace_parts)
                    ], className="mt-3")
                ], label="🔧 Execution Trace", tab_id="trace-tab")
            ], active_tab="response-tab")
        ])
    ])

def create_grouped_trace_items(trace_parts: List[Dict]) -> html.Div:
    """Create grouped trace items where tool calls and returns are combined"""
    if not trace_parts:
        return html.P("No trace data available.", className="text-muted")
    
    # Group tool calls and returns by tool name, preserving execution order
    tool_groups = {}
    messages = []
    tool_execution_order = []  # Track the order of first tool call appearance
    
    for i, part in enumerate(trace_parts):
        part_kind = part.get('part_kind', 'unknown')
        tool_name = part.get('tool_name', 'Message')
        
        if part_kind in ['tool-call', 'tool-return']:
            if tool_name not in tool_groups:
                tool_groups[tool_name] = {'calls': [], 'returns': [], 'indices': [], 'first_call_index': i}
                tool_execution_order.append(tool_name)
            
            if part_kind == 'tool-call':
                tool_groups[tool_name]['calls'].append(part)
            else:
                tool_groups[tool_name]['returns'].append(part)
            tool_groups[tool_name]['indices'].append(i)
        else:
            messages.append((i, part))
    
    components = []
    
    # Create grouped tool accordion items in execution order
    if tool_groups:
        accordion_items = []
        
        for tool_number, tool_name in enumerate(tool_execution_order, 1):
            group_data = tool_groups[tool_name]
            calls = group_data['calls']
            returns = group_data['returns']
            
            # Create content showing both calls and returns for this tool
            tool_content = []
            
            # Add tool calls
            if calls:
                tool_content.append(html.H6("🔧 Tool Call:", className="fw-bold mt-2 mb-2"))
                for call in calls:
                    tool_content.append(
                        dbc.Card([
                            dbc.CardHeader(dbc.Badge("INPUT", color="primary")),
                            dbc.CardBody([
                                html.Pre(
                                    to_string_for_pre(call.get('args', {})),
                                    style={
                                        'whiteSpace': 'pre-wrap',
                                        'wordBreak': 'break-word',
                                        'maxHeight': '300px',
                                        'overflowY': 'auto',
                                        'fontSize': '0.85rem'
                                    }
                                )
                            ])
                        ], className="mb-2")
                    )
            
            # Add tool returns
            if returns:
                tool_content.append(html.H6("📤 Tool Output:", className="fw-bold mt-3 mb-2"))
                for return_item in returns:
                    tool_content.append(
                        dbc.Card([
                            dbc.CardHeader(dbc.Badge("OUTPUT", color="success")),
                            dbc.CardBody([
                                html.Pre(
                                    to_string_for_pre(return_item.get('content', {})),
                                    style={
                                        'whiteSpace': 'pre-wrap',
                                        'wordBreak': 'break-word',
                                        'maxHeight': '400px',
                                        'overflowY': 'auto',
                                        'fontSize': '0.85rem'
                                    }
                                )
                            ])
                        ], className="mb-2")
                    )
            
            # Create accordion item for this tool with proper sequential numbering
            accordion_items.append(
                dbc.AccordionItem(
                    html.Div(tool_content),
                    title=f"Tool #{tool_number}: {tool_name}",
                    id=f"tool-group-{tool_number}-{tool_name.replace(' ', '-')}"
                )
            )
        
        components.append(
            html.Div([
                html.H6("🔧 Tool Executions (Sequential Order)", className="fw-bold mb-3"),
                dbc.Accordion(
                    accordion_items,
                    start_collapsed=False,
                    always_open=True
                )
            ])
        )
    
    # Messages section (if any)
    if messages:
        components.append(
            html.Div([
                html.H6("💬 Messages", className="fw-bold mb-3 mt-4"),
                dbc.Accordion([
                    dbc.AccordionItem([
                        html.Pre(
                            to_string_for_pre(part.get('content', part.get('args', {}))),
                            style={
                                'whiteSpace': 'pre-wrap',
                                'wordBreak': 'break-word',
                                'maxHeight': '300px',
                                'overflowY': 'auto',
                                'fontSize': '0.85rem'
                            }
                        )
                    ], title=f"#{i+1} {part.get('part_kind', 'unknown').title()}")
                    for i, part in messages
                ], start_collapsed=True, always_open=False)
            ])
        )
    
    return html.Div(components) if components else html.Div("No trace components found.", className="text-muted")

def create_enhanced_analysis_layout(analysis_results: Dict) -> html.Div:
    """Create enhanced analysis layout with improved UI and debugging info"""
    selected_text = analysis_results['selected_text']
    best_matches = analysis_results['best_matches']
    influence_chains = analysis_results['influence_chains']
    content_analysis = analysis_results.get('content_analysis', {})
    
    # Build layout children dynamically
    layout_children = [
        # Header with content origin analysis
        html.Div([
            html.H6("🧠 Enhanced RoPE Analysis", className="fw-bold mb-0"),
            dbc.Badge("3072D + Enhanced O(n²)", color="success", className="ms-2")
        ], className="d-flex align-items-center mb-3"),
        
        # Content origin analysis
        html.Div([
            html.H6("🔍 Content Origin Analysis", className="fw-bold mb-2"),
            dbc.Badge(
                "LLM Generated" if content_analysis.get('is_llm_generated') else "Tool Derived", 
                color="info" if content_analysis.get('is_llm_generated') else "success",
                className="me-2"
            ),
            dbc.Badge(f"Confidence: {content_analysis.get('tool_derived_confidence', 0):.3f}", color="secondary")
        ], className="mb-3"),
        
        # Selected text
        html.H6("Selected Text:", className="fw-bold"),
        html.Blockquote(f'"{selected_text}"', className="border-start border-4 border-info ps-2 mb-4"),
        
        # Enhanced tool matches with debugging info
        html.H5("🎯 Tool Matches (Sequential Order)", className="mb-3"),
        dbc.Accordion([
            dbc.AccordionItem([
                create_enhanced_tool_match_content(match)
            ], title=f"{match['debug_trace_ref']} | Score: {match['similarity_score']:.4f}")
            for match in best_matches
        ], start_collapsed=False, always_open=True),
        
        html.Hr(),
        
        # Enhanced influence chains
        html.H5("⛓️ Output-Based Influence Chains", className="mb-3"),
        html.P("Semantic relationships traced through tool output similarities:", 
               className="text-muted mb-3"),
    ]
    
    # Add influence chains if they exist
    for chain in influence_chains:
        chain_viz = create_enhanced_influence_chain_visualization(chain)
        if chain_viz:
            layout_children.append(chain_viz)
    
    return html.Div(layout_children)

def create_enhanced_tool_match_content(match: Dict) -> html.Div:
    """Enhanced tool match content with full input/output and debugging"""
    try:
        debug_trace_ref = match.get('debug_trace_ref', 'Unknown Tool')
        tool_number = match.get('tool_number', match.get('tool_idx', 0) + 1)
        similarity_score = match.get('similarity_score', 0.0)
        
        # Get all similarity scores with defaults
        rope_input_sim = match.get('rope_input_sim', 0.0)
        rope_output_sim = match.get('rope_output_sim', 0.0)
        base_input_sim = match.get('base_input_sim', 0.0)
        base_output_sim = match.get('base_output_sim', 0.0)
        text_overlap_score = match.get('text_overlap_score', 0.0)
        
        tool_input = match.get('tool_input', 'No input data')
        tool_output = match.get('tool_output', 'No output data')
        
        return html.Div([
            # Debugging and reference info
            dbc.Alert([
                html.Strong("Debug Info: "),
                f"Execution Order: Tool #{tool_number} | ",
                f"Tool Name: {match.get('tool_name', 'Unknown')}"
            ], color="light", className="mb-3"),
            
            # Enhanced similarity breakdown
            dbc.Row([
                dbc.Col([
                    html.H6("Similarity Breakdown:", className="fw-bold mb-2"),
                    dbc.Badge(f"Final: {similarity_score:.4f}", color="primary", className="me-2 mb-1"),
                    dbc.Badge(f"RoPE In: {rope_input_sim:.4f}", color="info", className="me-2 mb-1"),
                    dbc.Badge(f"RoPE Out: {rope_output_sim:.4f}", color="info", className="me-2 mb-1"),
                    html.Br(),
                    dbc.Badge(f"Base In: {base_input_sim:.4f}", color="secondary", className="me-2 mb-1"),
                    dbc.Badge(f"Base Out: {base_output_sim:.4f}", color="secondary", className="me-2 mb-1"),
                    dbc.Badge(f"Text Overlap: {text_overlap_score:.4f}", color="success", className="me-2 mb-1"),
                ], width=12)
            ], className="mb-3"),
            
            # Full tool input (no truncation)
            html.Strong("Tool Input (Full):", className="d-block mt-3 mb-2"),
            dbc.Card([
                dbc.CardBody([
                    html.Pre(str(tool_input), style={
                        'whiteSpace': 'pre-wrap', 
                        'wordBreak': 'break-word',
                        'fontSize': '0.85rem',
                        'maxHeight': '200px',
                        'overflowY': 'auto'
                    })
                ])
            ], className="mb-3"),
            
            # Full tool output (no truncation) 
            html.Strong("Tool Output (Full):", className="d-block mb-2"),
            dbc.Card([
                dbc.CardBody([
                    dcc.Markdown(str(tool_output), style={
                        'maxHeight': '300px',
                        'overflowY': 'auto',
                        'fontSize': '0.85rem'
                    })
                ])
            ])
        ])
        
    except Exception as e:
        logger.error(f"Error creating tool match content: {e}")
        return html.Div([
            dbc.Alert(f"Error displaying tool match: {str(e)}", color="warning")
        ])

def create_enhanced_influence_chain_visualization(chain_data: Dict) -> dbc.Card:
    """Enhanced influence chain visualization"""
    try:
        root_tool = chain_data.get('root_tool', {})
        chain = chain_data.get('chain', {})
        chain_length = chain_data.get('chain_length', 0)
        chain_type = chain_data.get('chain_type', 'output_based')
        
        if not root_tool or not chain:
            return html.Div()  # Return empty div if invalid data
        
        debug_trace_ref = root_tool.get('debug_trace_ref', 'Unknown Tool')
        
        card_content = create_enhanced_chain_level_content(chain, level=0)
        
        return dbc.Card([
            dbc.CardHeader([
                html.H6(f"🔗 {chain_type.title()} Chain", className="mb-0"),
                dbc.Badge(f"{debug_trace_ref}", color="info", className="me-2"),
                dbc.Badge(f"{chain_length} levels", color="secondary")
            ], className="d-flex align-items-center"),
            dbc.CardBody([card_content])
        ], className="mb-3")
        
    except Exception as e:
        logger.error(f"Error creating influence chain visualization: {e}")
        return html.Div()  # Return empty div on error

def create_enhanced_chain_level_content(chain_node: Dict, level: int, max_display_level: int = 4) -> html.Div:
    """Enhanced chain level content with better visualization"""
    if not chain_node or level > max_display_level:
        return html.Div()
    
    colors = ["primary", "secondary", "success", "info", "warning"]
    color = colors[min(level, len(colors) - 1)]
    margin_left = f"{level * 25}px"
    
    # Use proper sequential tool numbering (tool_idx is 0-based, so add 1)
    tool_number = chain_node['tool_idx'] + 1
    
    # Build children list dynamically to avoid None values
    div_children = [
        html.Div([
            dbc.Badge(f"L{level}", color=color, className="me-2"),
            html.Strong(f"🔧 Tool #{tool_number}: {chain_node['tool_name']}", className="me-2"),
            dbc.Badge(f"Score: {chain_node['influence_score']:.4f}", color="light", className="text-dark"),
        ], className="d-flex align-items-center mb-2"),
        
        html.P(chain_node['explanation'], className="text-muted small mb-2"),
    ]
    
    # Add parent chain indicator if exists
    if chain_node.get('parent_chain'):
        div_children.append(html.Div([
            html.I(className="bi bi-arrow-up-right me-2"),
            "Output influenced by..."
        ], className="text-info small mb-2"))
    
    # Main div content
    main_div_children = [
        html.Div(div_children, style={
            "marginLeft": margin_left, 
            "paddingLeft": "15px", 
            "borderLeft": f"3px solid var(--bs-{color})",
            "paddingTop": "10px",
            "paddingBottom": "10px"
        })
    ]
    
    # Add parent chain recursively if exists
    if chain_node.get('parent_chain'):
        parent_content = create_enhanced_chain_level_content(
            chain_node.get('parent_chain'), level + 1, max_display_level
        )
        if parent_content:
            main_div_children.append(parent_content)
    
    return html.Div(main_div_children, className="mb-2")

# --- Utility Functions ---
def extract_tool_calls_from_trace(trace_data: List[Dict]) -> List[ToolCall]:
    tool_calls, buffer = [], {}
    
    for span in trace_data:
        if not isinstance(span, dict): 
            continue
            
        for part in span.get('parts', []):
            kind, name = part.get('part_kind', ''), part.get('tool_name', 'unknown')
            
            if kind == 'tool-call': 
                buffer[name] = {"input": part.get('args', {}), "ts": time.time()}
            elif kind == 'tool-return' and name in buffer:
                call_data = buffer.pop(name)
                tool_calls.append(ToolCall(
                    tool_name=name, 
                    tool_input=call_data['input'], 
                    tool_output=part.get('content', ''),
                    timestamp=call_data['ts'], 
                    duration=time.time() - call_data['ts']
                ))
    
    return tool_calls

def extract_trace_parts(trace_data: List[Dict]) -> List[Dict]:
    return [part for span in trace_data if isinstance(span, dict) for part in span.get('parts', [])]

def to_string_for_pre(content: Any) -> str:
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return content
    try: 
        return json.dumps(content, indent=2, ensure_ascii=False)
    except (TypeError, ValueError): 
        return str(content)

def ensure_json_serializable(obj: Any) -> Any:
    if isinstance(obj, (bytes, bytearray)): 
        return obj.decode('utf-8', 'replace')
    if isinstance(obj, dict): 
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): 
        return [ensure_json_serializable(i) for i in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Travel Agent with RoPE and Chain Analysis')
    parser.add_argument('--env', default=".env", help='Path to .env file')
    parser.add_argument('--port', type=int, default=8050, help='Port number')
    args = parser.parse_args()
    
    if os.path.exists(args.env):
        load_dotenv(args.env)
        logger.info(f"Loaded environment variables from {args.env}")
    else:
        logger.warning(f".env file not found at {args.env}. Using system environment variables.")
    
    app.run(debug=True, host="0.0.0.0", port=args.port)