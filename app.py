from flask import Flask, request, jsonify, render_template_string
import os
import sys
import traceback
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from dotenv import load_dotenv

# Phi framework imports
from phi.agent import Agent
from phi.model.groq import Groq
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.lancedb import LanceDb
from phi.vectordb.search import SearchType
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder

# Additional libraries for enhanced functionality
import requests
from sentence_transformers import SentenceTransformer
import re
from collections import defaultdict

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('soybot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Enums and Data Classes
class QueryType(Enum):
    DISEASE_PEST = "disease_pest"
    FERTILIZER_NUTRITION = "fertilizer_nutrition"
    CULTIVATION_PRACTICES = "cultivation_practices"
    WEATHER_CLIMATE = "weather_climate"
    MARKET_ECONOMICS = "market_economics"
    GENERAL = "general"

class Language(Enum):
    ENGLISH = "english"
    HINDI = "hindi"
    MARATHI = "marathi"
    MIXED = "mixed"

@dataclass
class QueryContext:
    location: Optional[str] = None
    season: Optional[str] = None
    crop_stage: Optional[str] = None
    farm_size: Optional[str] = None
    soil_type: Optional[str] = None

@dataclass
class AgentResponse:
    response: str
    confidence: float
    agent_used: str
    sources: List[str]
    reasoning_chain: List[str]
    language: str

class LanguageProcessor:
    """Advanced multilingual processing for agricultural queries"""
    
    def __init__(self):
        self.hindi_chars_range = (0x0900, 0x097F)
        self.marathi_chars_range = (0x0900, 0x097F)  # Devanagari script
        self.english_pattern = re.compile(r'[a-zA-Z]')
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]')
        
        # Common agricultural terms mapping
        self.agricultural_terms = {
            'disease': ['बीमारी', 'रोग', 'आजार'],
            'pest': ['कीट', 'किड़े', 'पेस्ट'],
            'fertilizer': ['खाद', 'उर्वरक', 'खत'],
            'irrigation': ['सिंचाई', 'पानी', 'पाणी'],
            'harvest': ['फसल', 'कटाई', 'हार्वेस्ट'],
            'soil': ['मिट्टी', 'जमीन', 'माती'],
            'seed': ['बीज', 'सीड', 'बियाणे']
        }
    
    def detect_language(self, text: str) -> Language:
        """Enhanced language detection with code-mixing support"""
        if not text:
            return Language.ENGLISH
        
        devanagari_chars = len(self.devanagari_pattern.findall(text))
        english_chars = len(self.english_pattern.findall(text))
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return Language.ENGLISH
        
        devanagari_ratio = devanagari_chars / total_chars
        english_ratio = english_chars / total_chars
        
        # Code-mixing detection
        if devanagari_ratio > 0.3 and english_ratio > 0.3:
            return Language.MIXED
        elif devanagari_ratio > 0.5:
            return Language.HINDI  # Could also be Marathi
        else:
            return Language.ENGLISH
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for better processing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize common variations
        text = text.replace('ए', 'े')  # Normalize vowel marks
        text = text.replace('ओ', 'ो')
        
        return text
    
    def extract_agricultural_intent(self, text: str) -> QueryType:
        """Extract agricultural intent from text"""
        text_lower = text.lower()
        
        # Disease/Pest keywords
        disease_keywords = ['disease', 'pest', 'bug', 'infection', 'रोग', 'बीमारी', 'कीट', 'किड़े']
        if any(keyword in text_lower for keyword in disease_keywords):
            return QueryType.DISEASE_PEST
        
        # Fertilizer/Nutrition keywords
        fertilizer_keywords = ['fertilizer', 'nutrition', 'nutrient', 'खाद', 'उर्वरक', 'पोषक']
        if any(keyword in text_lower for keyword in fertilizer_keywords):
            return QueryType.FERTILIZER_NUTRITION
        
        # Cultivation keywords
        cultivation_keywords = ['planting', 'sowing', 'cultivation', 'बुआई', 'खेती', 'बोना']
        if any(keyword in text_lower for keyword in cultivation_keywords):
            return QueryType.CULTIVATION_PRACTICES
        
        # Weather keywords
        weather_keywords = ['weather', 'climate', 'rain', 'मौसम', 'बारिश', 'जलवायु']
        if any(keyword in text_lower for keyword in weather_keywords):
            return QueryType.WEATHER_CLIMATE
        
        return QueryType.GENERAL

class ConfidenceCalculator:
    """Calculate confidence scores for agent responses"""
    
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def calculate_confidence(self, query: str, response: str, knowledge_chunks: List[str]) -> float:
        """Calculate confidence based on multiple factors"""
        try:
            # Factor 1: Semantic similarity with knowledge base
            query_embedding = self.sentence_model.encode([query])
            
            if knowledge_chunks:
                knowledge_embeddings = self.sentence_model.encode(knowledge_chunks)
                similarities = np.dot(query_embedding, knowledge_embeddings.T)[0]
                max_similarity = np.max(similarities)
                similarity_score = max_similarity
            else:
                similarity_score = 0.3  # Low confidence if no knowledge chunks
            
            # Factor 2: Response length and structure (reasonable responses are detailed)
            response_length_score = min(len(response.split()) / 50, 1.0)
            
            # Factor 3: Presence of specific agricultural terms
            agricultural_terms = ['soybean', 'fertilizer', 'irrigation', 'pest', 'disease', 'soil', 'harvest']
            term_presence = sum(1 for term in agricultural_terms if term.lower() in response.lower())
            term_score = min(term_presence / 3, 1.0)
            
            # Factor 4: Avoid uncertain language patterns
            uncertain_phrases = ['i think', 'maybe', 'possibly', 'not sure', 'might be']
            uncertainty_penalty = sum(0.1 for phrase in uncertain_phrases if phrase in response.lower())
            
            # Weighted combination
            confidence = (
                0.4 * similarity_score +
                0.2 * response_length_score +
                0.2 * term_score +
                0.2 * (1.0 - uncertainty_penalty)
            )
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

class KnowledgeEnhancer:
    """Enhance responses with external context"""
    
    def __init__(self):
        self.weather_cache = {}
        self.cache_timeout = 3600  # 1 hour
    
    def get_weather_context(self, location: str = None) -> str:
        """Get weather information for agricultural context"""
        try:
            # Simple weather context (in production, use actual weather API)
            current_season = self._get_current_season()
            return f"Current season: {current_season}. Consider seasonal agricultural practices."
        except Exception as e:
            logger.error(f"Error getting weather context: {e}")
            return ""
    
    def _get_current_season(self) -> str:
        """Determine current agricultural season in India"""
        month = datetime.now().month
        if month in [6, 7, 8, 9]:
            return "Kharif (Monsoon)"
        elif month in [10, 11, 12, 1, 2, 3]:
            return "Rabi (Winter)"
        else:
            return "Zaid (Summer)"
    
    def enhance_response(self, base_response: str, query_context: QueryContext) -> str:
        """Enhance response with contextual information"""
        enhancements = []
        
        # Add seasonal context
        weather_context = self.get_weather_context(query_context.location)
        if weather_context:
            enhancements.append(weather_context)
        
        # Add location-specific notes
        if query_context.location:
            enhancements.append(f"Note: Consider local conditions in {query_context.location}")
        
        if enhancements:
            enhancement_text = "\n\n**Additional Context:**\n" + "\n".join(enhancements)
            return base_response + enhancement_text
        
        return base_response

class MultiAgentSystem:
    """Multi-agent system for agricultural advisory"""
    
    def __init__(self, groq_api_key: str, knowledge_base: PDFKnowledgeBase):
        self.groq_api_key = groq_api_key
        self.knowledge_base = knowledge_base
        self.language_processor = LanguageProcessor()
        self.confidence_calculator = ConfidenceCalculator()
        self.knowledge_enhancer = KnowledgeEnhancer()
        
        # Initialize specialized agents
        self._initialize_agents()
        
        # Initialize coordinator
        self._initialize_coordinator()
    
    def _initialize_agents(self):
        """Initialize specialized agricultural agents"""
        
        # Disease and Pest Management Agent
        self.disease_agent = Agent(
            name="Plant Health Specialist",
            role="Expert in soybean diseases, pests, and plant health management",
            model=Groq(id="llama-3.3-70b-versatile", api_key=self.groq_api_key),
            knowledge=self.knowledge_base,
            instructions=[
                "You are a plant pathology expert specializing in soybean diseases and pests.",
                "Focus on accurate disease diagnosis, pest identification, and treatment recommendations.",
                "Provide integrated pest management (IPM) strategies.",
                "Always consider organic and sustainable solutions alongside chemical treatments.",
                "Respond in the same language as the query.",
                "Use simple, farmer-friendly language.",
            ],
            show_tool_calls=False,
            markdown=False
        )
        
        # Nutrition and Fertilizer Agent
        self.nutrition_agent = Agent(
            name="Soil & Nutrition Expert",
            role="Specialist in soil health, fertilizer recommendations, and plant nutrition",
            model=Groq(id="llama-3.3-70b-versatile", api_key=self.groq_api_key),
            knowledge=self.knowledge_base,
            instructions=[
                "You are a soil scientist and plant nutrition expert for soybean cultivation.",
                "Provide specific fertilizer recommendations based on soil conditions and crop stage.",
                "Focus on balanced nutrition and cost-effective fertilizer programs.",
                "Consider organic and inorganic fertilizer options.",
                "Respond in the same language as the query.",
                "Explain the 'why' behind recommendations.",
            ],
            show_tool_calls=False,
            markdown=False
        )
        
        # Cultivation Practices Agent
        self.cultivation_agent = Agent(
            name="Crop Management Specialist",
            role="Expert in soybean cultivation practices, planting, and crop management",
            model=Groq(id="llama-3.3-70b-versatile", api_key=self.groq_api_key),
            knowledge=self.knowledge_base,
            instructions=[
                "You are an agronomy expert specializing in soybean cultivation practices.",
                "Provide detailed guidance on sowing, spacing, irrigation, and crop management.",
                "Focus on best practices for maximum yield and quality.",
                "Consider different farming systems and scales.",
                "Respond in the same language as the query.",
                "Provide practical, implementable advice.",
            ],
            show_tool_calls=False,
            markdown=False
        )
        
        # General Agricultural Agent
        self.general_agent = Agent(
            name="General Agriculture Advisor",
            role="Comprehensive agricultural advisor for general soybean farming queries",
            model=Groq(id="llama-3.3-70b-versatile", api_key=self.groq_api_key),
            knowledge=self.knowledge_base,
            instructions=[
                "You are a comprehensive agricultural advisor for soybean farming.",
                "Handle general queries about soybean cultivation, economics, and farming practices.",
                "Provide holistic farming advice considering all aspects of agriculture.",
                "Respond in the same language as the query.",
                "Be encouraging and supportive to farmers.",
            ],
            show_tool_calls=False,
            markdown=False
        )
    
    def _initialize_coordinator(self):
        """Initialize the master coordinator agent"""
        self.coordinator = Agent(
            name="SoyBot Coordinator",
            role="Master coordinator for routing queries to appropriate specialists",
            model=Groq(id="llama-3.3-70b-versatile", api_key=self.groq_api_key),
            instructions=[
                "You are the master coordinator for SoyBot agricultural advisory system.",
                "Analyze queries and determine the most appropriate specialist agent.",
                "If a query involves multiple domains, coordinate responses from multiple agents.",
                "Always ensure responses are practical and farmer-friendly.",
                "Respond in the same language as the query.",
            ],
            show_tool_calls=False,
            markdown=False
        )
    
    def route_query(self, query: str, query_context: QueryContext) -> AgentResponse:
        """Route query to appropriate agent based on intent"""
        try:
            # Detect language and intent
            language = self.language_processor.detect_language(query)
            intent = self.language_processor.extract_agricultural_intent(query)
            
            logger.info(f"Query routed - Language: {language.value}, Intent: {intent.value}")
            
            # Select appropriate agent
            agent_mapping = {
                QueryType.DISEASE_PEST: (self.disease_agent, "Plant Health Specialist"),
                QueryType.FERTILIZER_NUTRITION: (self.nutrition_agent, "Soil & Nutrition Expert"),
                QueryType.CULTIVATION_PRACTICES: (self.cultivation_agent, "Crop Management Specialist"),
                QueryType.GENERAL: (self.general_agent, "General Agriculture Advisor"),
                QueryType.WEATHER_CLIMATE: (self.general_agent, "General Agriculture Advisor"),
                QueryType.MARKET_ECONOMICS: (self.general_agent, "General Agriculture Advisor")
            }
            
            agent, agent_name = agent_mapping.get(intent, (self.general_agent, "General Agriculture Advisor"))
            
            # Generate response
            response = agent.run(query)
            
            # Calculate confidence
            knowledge_chunks = self._get_relevant_knowledge_chunks(query)
            confidence = self.confidence_calculator.calculate_confidence(query, response, knowledge_chunks)
            
            # Enhance response with context
            enhanced_response = self.knowledge_enhancer.enhance_response(response, query_context)
            
            return AgentResponse(
                response=enhanced_response,
                confidence=confidence,
                agent_used=agent_name,
                sources=self._extract_sources(response),
                reasoning_chain=self._extract_reasoning_chain(response),
                language=language.value
            )
            
        except Exception as e:
            logger.error(f"Error in query routing: {e}")
            return AgentResponse(
                response="मुझे खुशी होगी आपकी मदद करने में, लेकिन अभी कुछ तकनीकी समस्या है। कृपया बाद में पूछें।",
                confidence=0.1,
                agent_used="Error Handler",
                sources=[],
                reasoning_chain=[],
                language=language.value if 'language' in locals() else Language.ENGLISH.value
            )
    
    def _get_relevant_knowledge_chunks(self, query: str) -> List[str]:
        """Get relevant knowledge chunks for confidence calculation"""
        try:
            # This would ideally retrieve the actual chunks used by the agent
            # For now, return a placeholder
            return ["Knowledge from soybean cultivation guide"]
        except:
            return []
    
    def _extract_sources(self, response: str) -> List[str]:
        """Extract source citations from response"""
        # In a real implementation, this would extract actual source citations
        return ["ICAR-IISR Package of Practices for Soybean"]
    
    def _extract_reasoning_chain(self, response: str) -> List[str]:
        """Extract reasoning chain from agent's decision process"""
        # In a real implementation, this would capture the agent's reasoning steps
        return ["Analyzed query intent", "Retrieved relevant knowledge", "Generated contextual response"]

class SoyBotSystem:
    """Main SoyBot system class"""
    
    def __init__(self):
        self.multi_agent_system = None
        self.is_initialized = False
        self.groq_api_key = None
        self.knowledge_base = None
    
    def initialize(self) -> bool:
        """Initialize the SoyBot system"""
        try:
            logger.info("🔄 Initializing SoyBot system...")
            
            # Check for API key
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            if not self.groq_api_key:
                logger.error("❌ GROQ_API_KEY not found in environment variables")
                return False
            
            # Check for PDF file
            pdf_path = "Soybeanpackageofpractices.pdf"
            if not os.path.exists(pdf_path):
                logger.error(f"❌ PDF file not found: {pdf_path}")
                return False
            
            logger.info("📚 Setting up knowledge base...")
            
            # Knowledge Base setup with enhanced embeddings
            self.knowledge_base = PDFKnowledgeBase(
                path=pdf_path,
                vector_db=LanceDb(
                    table_name="soybean_practices_enhanced",
                    uri="./vectordb/soybot_enhanced_db",
                    search_type=SearchType.vector,
                    embedder=SentenceTransformerEmbedder(
                        model="all-MiniLM-L6-v2",
                        dimensions=384
                    ),
                )
            )
            
            # Load knowledge base
            logger.info("🔍 Loading PDF knowledge...")
            self.knowledge_base.load(recreate=False)
            
            # Initialize multi-agent system
            logger.info("🤖 Creating multi-agent system...")
            self.multi_agent_system = MultiAgentSystem(self.groq_api_key, self.knowledge_base)
            
            self.is_initialized = True
            logger.info("✅ SoyBot system initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing SoyBot: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def process_query(self, query: str, context_data: Dict = None) -> Dict:
        """Process a query through the multi-agent system"""
        if not self.is_initialized:
            return {
                'success': False,
                'error': 'SoyBot system not initialized'
            }
        
        try:
            # Parse context data
            query_context = QueryContext(
                location=context_data.get('location') if context_data else None,
                season=context_data.get('season') if context_data else None,
                crop_stage=context_data.get('crop_stage') if context_data else None,
                farm_size=context_data.get('farm_size') if context_data else None,
                soil_type=context_data.get('soil_type') if context_data else None
            )
            
            # Process query through multi-agent system
            start_time = time.time()
            agent_response = self.multi_agent_system.route_query(query, query_context)
            processing_time = time.time() - start_time
            
            # Format response
            return {
                'success': True,
                'response': agent_response.response,
                'confidence': agent_response.confidence,
                'agent_used': agent_response.agent_used,
                'language': agent_response.language,
                'sources': agent_response.sources,
                'reasoning_chain': agent_response.reasoning_chain,
                'processing_time': round(processing_time, 2),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Global SoyBot instance
soybot_system = SoyBotSystem()

# Flask Routes
@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'SoyBot Enhanced Multi-Agent Agricultural Advisory System',
        'version': '2.0',
        'status': 'active' if soybot_system.is_initialized else 'initializing',
        'endpoints': {
            'chat': '/chat',
            'health': '/health',
            'agents': '/agents/info'
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if soybot_system.is_initialized else 'initializing',
        'initialized': soybot_system.is_initialized,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/agents/info')
def agents_info():
    """Get information about available agents"""
    if not soybot_system.is_initialized:
        return jsonify({'error': 'System not initialized'}), 500
    
    return jsonify({
        'agents': [
            {
                'name': 'Plant Health Specialist',
                'specialization': 'Disease and pest management',
                'handles': ['diseases', 'pests', 'plant health', 'IPM']
            },
            {
                'name': 'Soil & Nutrition Expert',
                'specialization': 'Soil health and plant nutrition',
                'handles': ['fertilizers', 'nutrients', 'soil health', 'deficiencies']
            },
            {
                'name': 'Crop Management Specialist',
                'specialization': 'Cultivation practices and crop management',
                'handles': ['planting', 'irrigation', 'cultivation', 'crop stages']
            },
            {
                'name': 'General Agriculture Advisor',
                'specialization': 'General agricultural guidance',
                'handles': ['general queries', 'economics', 'weather', 'best practices']
            }
        ],
        'routing': 'Automatic based on query intent analysis',
        'languages': ['English', 'Hindi', 'Marathi', 'Mixed']
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        query = data['message'].strip()
        context_data = data.get('context', {})
        
        if not query:
            return jsonify({'error': 'Empty message not allowed'}), 400
        
        # Process query
        result = soybot_system.process_query(query, context_data)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Feedback endpoint for continuous improvement"""
    try:
        data = request.get_json()
        
        # Log feedback for analysis
        logger.info(f"Feedback received: {json.dumps(data, ensure_ascii=False)}")
        
        return jsonify({
            'success': True,
            'message': 'Feedback received successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in feedback endpoint: {e}")
        return jsonify({'error': 'Failed to process feedback'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Initialize system on startup
if __name__ == '__main__':
    logger.info("🌱 Starting SoyBot Enhanced System...")
    
    if not soybot_system.initialize():
        logger.error("Failed to initialize SoyBot system")
        sys.exit(1)
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )
else:
    # Initialize when imported as module
    soybot_system.initialize()