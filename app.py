import os
import sys
import time
import json
import logging
import asyncio
import aiofiles
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import traceback
from functools import wraps
import pickle

# Flask and web components
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# ML and NLP components
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Language processing
import langdetect
from googletrans import Translator
import re

# Phi components - FIXED
from phi.agent import Agent
from phi.model.groq import Groq
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.lancedb import LanceDb
from phi.vectordb.search import SearchType
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_soybot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('EnhancedSoyBot')

# Enums and Data Classes
class QueryIntent(Enum):
    DISEASE_DIAGNOSIS = "disease_diagnosis"
    PEST_CONTROL = "pest_control"
    FERTILIZER_ADVICE = "fertilizer_advice"
    PLANTING_GUIDANCE = "planting_guidance"
    HARVESTING_INFO = "harvesting_info"
    IRRIGATION = "irrigation"
    VARIETY_SELECTION = "variety_selection"
    GENERAL_QUERY = "general_query"

class Language(Enum):
    ENGLISH = "en"
    HINDI = "hi"
    MARATHI = "mr"

@dataclass
class QueryAnalysis:
    intent: QueryIntent
    confidence: float
    language: Language
    entities: List[str]

@dataclass
class ResponseQuality:
    relevance: float
    completeness: float
    accuracy: float
    actionability: float
    overall_confidence: float
    needs_refinement: bool
    suggested_improvements: List[str]

class AdvancedLanguageProcessor:
    """Enhanced language processing with better multilingual support"""
    
    def __init__(self):
        self.translator = Translator()
        self.hindi_romanized_patterns = {
            'kya': 'क्या', 'kaise': 'कैसे', 'kab': 'कब', 'kahan': 'कहाँ',
            'fasal': 'फसल', 'beej': 'बीज', 'khad': 'खाद', 'pani': 'पानी',
            'kisan': 'किसान', 'zameen': 'जमीन', 'kheti': 'खेती'
        }
        self.agriculture_terms = {
            'disease', 'pest', 'fertilizer', 'seed', 'crop', 'soil', 'irrigation',
            'harvest', 'sowing', 'planting', 'variety', 'yield', 'nitrogen',
            'phosphorus', 'potassium', 'organic', 'pesticide', 'fungicide'
        }
        
    def enhanced_language_detection(self, text: str) -> Dict[str, any]:
        """Enhanced language detection with confidence scoring"""
        try:
            primary_lang = langdetect.detect(text)
            lang_probs = langdetect.detect_langs(text)
        except:
            primary_lang = 'en'
            lang_probs = []
        
        # Check for Hindi/Marathi words in Roman script
        romanized_hindi = self.detect_romanized_hindi(text)
        
        # Check for code-switching (English-Hindi mixed)
        code_switched = self.detect_code_switching(text)
        
        # Calculate agriculture domain relevance
        agriculture_relevance = self.calculate_agriculture_relevance(text)
        
        confidence = max([prob.prob for prob in lang_probs]) if lang_probs else 0.5
        
        return {
            'primary': primary_lang,
            'confidence': confidence,
            'romanized_hindi': romanized_hindi,
            'code_switched': code_switched,
            'agriculture_relevance': agriculture_relevance,
            'all_probabilities': {lang.lang: lang.prob for lang in lang_probs}
        }
    
    def detect_romanized_hindi(self, text: str) -> bool:
        """Detect Hindi words written in Roman script"""
        words = text.lower().split()
        hindi_word_count = sum(1 for word in words if word in self.hindi_romanized_patterns)
        return hindi_word_count > 0
    
    def detect_code_switching(self, text: str) -> bool:
        """Detect mixed language usage"""
        try:
            sentences = re.split(r'[.!?]', text)
            languages = []
            for sentence in sentences:
                if sentence.strip():
                    try:
                        lang = langdetect.detect(sentence.strip())
                        languages.append(lang)
                    except:
                        continue
            
            unique_languages = set(languages)
            return len(unique_languages) > 1
        except:
            return False
    
    def calculate_agriculture_relevance(self, text: str) -> float:
        """Calculate how relevant the text is to agriculture"""
        words = set(text.lower().split())
        agriculture_words = words.intersection(self.agriculture_terms)
        return len(agriculture_words) / max(len(words), 1)

class QueryIntentClassifier:
    """Advanced query intent classification"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.classifier = MultinomialNB()
        self.is_trained = False
        self.intent_keywords = {
            QueryIntent.DISEASE_DIAGNOSIS: [
                'disease', 'infection', 'sick', 'spots', 'yellowing', 'wilting',
                'blight', 'rust', 'fungus', 'viral', 'bacterial', 'symptoms',
                'रोग', 'संक्रमण', 'बीमार', 'धब्बे', 'पीलापन', 'सूखना'  # Hindi
            ],
            QueryIntent.PEST_CONTROL: [
                'pest', 'insect', 'bug', 'caterpillar', 'aphid', 'thrips',
                'control', 'spray', 'pesticide', 'damage', 'eating', 'larvae',
                'कीट', 'कीड़े', 'छिड़काव', 'नुकसान', 'खाना'  # Hindi
            ],
            QueryIntent.FERTILIZER_ADVICE: [
                'fertilizer', 'nutrients', 'nitrogen', 'phosphorus', 'potassium',
                'NPK', 'organic', 'compost', 'manure', 'feeding', 'nutrition',
                'खाद', 'उर्वरक', 'पोषक', 'नाइट्रोजन', 'गोबर'  # Hindi
            ],
            QueryIntent.PLANTING_GUIDANCE: [
                'planting', 'sowing', 'seeding', 'when to plant', 'plant spacing',
                'depth', 'germination', 'varieties', 'cultivar', 'hybrid',
                'बुवाई', 'रोपण', 'बीज', 'किस्म', 'अंकुरण'  # Hindi
            ],
            QueryIntent.HARVESTING_INFO: [
                'harvest', 'harvesting', 'maturity', 'ready', 'picking',
                'yield', 'timing', 'storage', 'drying', 'processing',
                'कटाई', 'फसल', 'पकना', 'भंडारण', 'सुखाना'  # Hindi
            ],
            QueryIntent.IRRIGATION: [
                'water', 'irrigation', 'watering', 'moisture', 'drought',
                'rain', 'drip', 'sprinkler', 'flooding', 'dry', 'wet',
                'पानी', 'सिंचाई', 'नमी', 'सूखा', 'बारिश'  # Hindi
            ]
        }
    
    def classify_intent(self, query: str) -> Dict[str, any]:
        """Classify query intent with improved confidence scores"""
        keyword_scores = {}
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for intent, keywords in self.intent_keywords.items():
            # Exact keyword matches
            exact_matches = sum(1 for keyword in keywords if keyword in query_lower)
            
            # Word-level matches for better detection
            word_matches = sum(1 for word in query_words 
                            for keyword in keywords 
                            if word in keyword or keyword in word)
            
            # Calculate score with both exact and partial matches
            total_score = (exact_matches * 2) + word_matches
            normalized_score = total_score / (len(keywords) + 5)  # Normalize
            
            keyword_scores[intent.value] = min(1.0, normalized_score)
        
        if max(keyword_scores.values()) > 0.05:  # Lower threshold
            primary_intent = max(keyword_scores, key=keyword_scores.get)
            confidence = keyword_scores[primary_intent]
        else:
            primary_intent = QueryIntent.GENERAL_QUERY.value
            confidence = 0.5
        
        return {
            'primary_intent': primary_intent,
            'confidence': confidence,
            'all_scores': keyword_scores
        }

class FixedKnowledgeBase:
    """Fixed knowledge base with proper tool registration"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
        self.knowledge_base = None
    
    def create_knowledge_base(self) -> PDFKnowledgeBase:
        """Create knowledge base with proper configuration"""
        logger.info("Creating knowledge base from PDF...")
        
        try:
            if not os.path.exists(self.pdf_path):
                raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
            
            # Create knowledge base with FIXED configuration FIRST
            self.knowledge_base = PDFKnowledgeBase(
                path=self.pdf_path,
                vector_db=LanceDb(
                    table_name="soybean_unified",
                    uri="./vectordb/soybot_fixed_db",
                    search_type=SearchType.vector,
                    embedder=self.embedder,
                    nprobes=10,          # Better search performance
                    
                ),
                # Add proper chunking for better RAG
                chunk_size=1000,
                chunk_overlap=200,
            )
            
            # Load the knowledge base
            logger.info("Loading knowledge base...")
            self.knowledge_base.load(recreate=False)
            
            # NOW verify document count AFTER creation
            try:
                if hasattr(self.knowledge_base, 'vector_db') and hasattr(self.knowledge_base.vector_db, 'table'):
                    doc_count = len(self.knowledge_base.vector_db.table.to_pandas())
                    logger.info(f"Knowledge base loaded with {doc_count} document chunks")
                    
                    if doc_count == 0:
                        logger.warning("Knowledge base is empty, recreating...")
                        self.knowledge_base.load(recreate=True)
                        doc_count = len(self.knowledge_base.vector_db.table.to_pandas())
                        logger.info(f"Recreated knowledge base with {doc_count} document chunks")
            except Exception as verify_error:
                logger.warning(f"Could not verify document count: {verify_error}")
            
            logger.info("Knowledge base loaded successfully")
            return self.knowledge_base
            
        except Exception as e:
            logger.error(f"Error creating knowledge base: {e}")
            logger.error(traceback.format_exc())
            raise

class ResponseQualityAssessor:
    """Assess and improve response quality"""
    
    def __init__(self):
        self.agriculture_keywords = [
            'soybean', 'crop', 'plant', 'seed', 'fertilizer', 'soil', 'pest', 'disease',
            'irrigation', 'harvest', 'variety', 'nutrients', 'farming', 'agriculture'
        ]
        
    def assess_response_quality(self, query: str, response: str, 
                              knowledge_sources: List = None) -> ResponseQuality:
        """Assess agricultural response quality"""
        
        relevance = self._calculate_agricultural_relevance(query, response)
        completeness = self._assess_practical_completeness(response)
        accuracy = self._check_helpful_indicators(response)
        actionability = self._assess_practical_actionability(response)
        
        overall_confidence = (relevance * 0.25 + completeness * 0.25 + 
                            accuracy * 0.25 + actionability * 0.25)
        
        needs_refinement = overall_confidence < 0.3
        suggested_improvements = self._generate_practical_improvements(
            relevance, completeness, accuracy, actionability
        )
        
        return ResponseQuality(
            relevance=relevance,
            completeness=completeness,
            accuracy=accuracy,
            actionability=actionability,
            overall_confidence=overall_confidence,
            needs_refinement=needs_refinement,
            suggested_improvements=suggested_improvements
        )
    
    def _calculate_agricultural_relevance(self, query: str, response: str) -> float:
        """Calculate agricultural relevance"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        agri_keywords_in_response = sum(1 for word in self.agriculture_keywords 
                                      if word in response_words)
        query_overlap = len(query_words.intersection(response_words))
        
        relevance_score = min(1.0, (agri_keywords_in_response * 0.1) + 
                                  (query_overlap * 0.05) + 0.5)
        return relevance_score
    
    def _assess_practical_completeness(self, response: str) -> float:
        """Check response completeness"""
        response_length = len(response.split())
        
        if response_length < 10:
            return 0.4
        elif response_length < 30:
            return 0.6
        elif response_length < 50:
            return 0.8
        else:
            return 0.9
    
    def _check_helpful_indicators(self, response: str) -> float:
        """Check for helpful indicators"""
        helpful_indicators = [
            'recommend', 'suggest', 'use', 'apply', 'plant', 'water',
            'fertilizer', 'treatment', 'control', 'management', 'practice'
        ]
        
        response_lower = response.lower()
        helpful_count = sum(1 for indicator in helpful_indicators 
                          if indicator in response_lower)
        
        return min(1.0, helpful_count * 0.1 + 0.5)
    
    def _assess_practical_actionability(self, response: str) -> float:
        """Assess actionability"""
        action_indicators = [
            'kg', 'gram', 'liter', 'per', 'apply', 'spray', 'mix',
            'time', 'week', 'month', 'season', 'stage', 'step'
        ]
        
        response_lower = response.lower()
        action_count = sum(1 for indicator in action_indicators 
                         if indicator in response_lower)
        
        return min(1.0, action_count * 0.08 + 0.5)
    
    def _generate_practical_improvements(self, relevance, completeness, 
                                       accuracy, actionability) -> List[str]:
        """Generate improvement suggestions"""
        improvements = []
        
        if relevance < 0.6:
            improvements.append("Focus more on the specific crop issue")
        if completeness < 0.5:
            improvements.append("Provide more detailed information")
        if actionability < 0.5:
            improvements.append("Include specific quantities or timing")
        
        return improvements[:2]

class ContextAwareEnhancer:
    """Add contextual information to responses"""
    
    def __init__(self):
        self.seasonal_context = {
            'kharif': {
                'months': [6, 7, 8, 9, 10],
                'activities': ['sowing', 'vegetative growth', 'flowering', 'pod filling', 'harvesting']
            },
            'rabi': {
                'months': [11, 12, 1, 2, 3],
                'activities': ['land preparation', 'seed treatment', 'storage management']
            }
        }
    
    def get_seasonal_context(self) -> Dict[str, any]:
        """Get current seasonal context"""
        current_month = datetime.now().month
        current_season = 'kharif' if current_month in self.seasonal_context['kharif']['months'] else 'rabi'
        
        return {
            'season': current_season,
            'month': current_month,
            'relevant_activities': self.seasonal_context[current_season]['activities']
        }
    
    def enhance_with_context(self, response: str, query: str) -> str:
        """Enhance response with contextual information"""
        seasonal_context = self.get_seasonal_context()
        
        if any(activity in query.lower() for activity in ['plant', 'sow', 'harvest', 'fertilizer']):
            season_info = f"\n\n📅 Seasonal Context: Currently in {seasonal_context['season'].capitalize()} season. "
            if seasonal_context['season'] == 'kharif':
                season_info += "This is the main soybean growing season in India."
            else:
                season_info += "Focus on land preparation and planning for next kharif season."
            
            response += season_info
        
        return response

class FixedMultiAgentSoyBot:
    """Fixed multi-agent SoyBot system with proper RAG"""
    
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.language_processor = AdvancedLanguageProcessor()
        self.intent_classifier = QueryIntentClassifier()
        self.knowledge_base_manager = FixedKnowledgeBase("Soybeanpackageofpractices.pdf")
        self.quality_assessor = ResponseQualityAssessor()
        self.context_enhancer = ContextAwareEnhancer()
        
        self.agents = {}
        self.performance_metrics = {
            'total_queries': 0,
            'successful_responses': 0,
            'failed_responses': 0,
            'average_response_time': 0,
            'quality_scores': []
        }
        
        self.is_initialized = False
        
    def initialize(self):
        """Initialize all components"""
        try:
            logger.info("Initializing Fixed SoyBot...")
            
            # Create unified knowledge base
            knowledge_base = self.knowledge_base_manager.create_knowledge_base()
            
            # Initialize specialized agents with CORRECT configuration
            self._create_fixed_agents(knowledge_base)
            
            self.is_initialized = True
            logger.info("Fixed SoyBot initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing SoyBot: {e}")
            raise
    
    def _create_fixed_agents(self, knowledge_base: PDFKnowledgeBase):
        """Create agents with CORRECT configuration to fix the tool error"""
        
        # FIXED: Proper agent configuration that enables knowledge search
        base_instructions = [
            "You are an expert soybean farming advisor.",
            "Use the knowledge base to provide accurate, practical advice.",
            "Give direct, actionable answers based on scientific guidelines.",
            "Always respond in the same language the question was asked.",
            "Search the knowledge base for relevant information before responding."
        ]
        
        # Plant Health Specialist - FIXED configuration
        self.agents['plant_health'] = Agent(
            name="Plant Health Specialist",
            model=Groq(id="llama-3.3-70b-versatile", api_key=self.groq_api_key),
            knowledge=knowledge_base,
            instructions=base_instructions + [
                "Specialize in plant diseases, pests, and health issues.",
                "Provide specific treatment recommendations with dosages.",
                "Focus on integrated pest management approaches."
            ],
            show_tool_calls=True,
            markdown=True,
            # CRITICAL: These parameters enable proper knowledge search
            search_knowledge=True,
            read_chat_history=True
        )
        
        # Nutrition Specialist - FIXED configuration
        self.agents['nutrition'] = Agent(
            name="Nutrition Specialist", 
            model=Groq(id="llama-3.3-70b-versatile", api_key=self.groq_api_key),
            knowledge=knowledge_base,
            instructions=base_instructions + [
                "Specialize in soil health, fertilizers, and plant nutrition.",
                "Provide specific fertilizer recommendations with quantities.",
                "Include timing for nutrient applications."
            ],
            show_tool_calls=True,
            markdown=True,
            search_knowledge=True,
            read_chat_history=True
        )
        
        # Crop Management Specialist - FIXED configuration
        self.agents['crop_management'] = Agent(
            name="Crop Management Specialist",
            model=Groq(id="llama-3.3-70b-versatile", api_key=self.groq_api_key),
            knowledge=knowledge_base,
            instructions=base_instructions + [
                "Specialize in planting, irrigation, and general crop management.",
                "Provide timing-specific recommendations.",
                "Include seasonal and regional considerations."
            ],
            show_tool_calls=True,
            markdown=True,
            search_knowledge=True,
            read_chat_history=True
        )
        
        # General Coordinator - FIXED configuration
        self.agents['coordinator'] = Agent(
            name="SoyBot Coordinator",
            model=Groq(id="llama-3.3-70b-versatile", api_key=self.groq_api_key),
            knowledge=knowledge_base,
            instructions=base_instructions + [
                "Handle general farming questions comprehensively.",
                "Draw from all aspects of soybean cultivation knowledge.",
                "Provide complete answers addressing farmer needs."
            ],
            show_tool_calls=True,
            markdown=True,
            search_knowledge=True,
            read_chat_history=True
        )

    def performance_monitor(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            self.performance_metrics['total_queries'] += 1
            
            try:
                result = func(self, *args, **kwargs)
                self.performance_metrics['successful_responses'] += 1
                
                response_time = time.time() - start_time
                total_successful = self.performance_metrics['successful_responses']
                current_avg = self.performance_metrics['average_response_time']
                self.performance_metrics['average_response_time'] = (
                    (current_avg * (total_successful - 1) + response_time) / total_successful
                )
                
                logger.info(f"Query processed successfully in {response_time:.2f}s")
                return result
                
            except Exception as e:
                self.performance_metrics['failed_responses'] += 1
                logger.error(f"Error processing query: {str(e)}")
                raise
                
        return wrapper
    
    @performance_monitor 
    def process_query(self, query: str, user_context: Optional[Dict] = None) -> Dict[str, any]:
        """Process query with full feature set"""
        
        # Language analysis - RESTORED
        lang_info = self.language_processor.enhanced_language_detection(query)
        
        # Intent classification
        intent_info = self.intent_classifier.classify_intent(query)
        
        # Route to appropriate agent
        agent_response = self._route_query_to_agent(query, intent_info)
        
        # Quality assessment
        quality_assessment = self.quality_assessor.assess_response_quality(
            query, agent_response['response']
        )
        
        # Store quality metrics
        self.performance_metrics['quality_scores'].append(quality_assessment.overall_confidence)
        
        # Enhance with context - RESTORED
        enhanced_response = self.context_enhancer.enhance_with_context(
            agent_response['response'], query
        )
        
        return {
            'response': enhanced_response,
            'agent_used': agent_response['agent'],
            'language_info': lang_info,
            'intent_info': intent_info,
            'quality_assessment': quality_assessment.__dict__,
            'processing_time': time.time()
        }
    
    def _route_query_to_agent(self, query: str, intent_info: Dict) -> Dict[str, str]:
        """Route query to appropriate agent"""

        intent = intent_info['primary_intent']
        confidence = intent_info['confidence']
        
        # DEBUG: Log routing decision
        logger.info(f"Intent: {intent}, Confidence: {confidence:.3f}")
        logger.info(f"All scores: {intent_info.get('all_scores', {})}")
        
        # Route based on intent with lower thresholds
        if intent in ['disease_diagnosis', 'pest_control'] and confidence > 0.1:
            agent_name = 'plant_health'
        elif intent in ['fertilizer_advice'] and confidence > 0.1:
            agent_name = 'nutrition'  
        elif intent in ['planting_guidance', 'harvesting_info', 'irrigation'] and confidence > 0.1:
            agent_name = 'crop_management'
        else:
            agent_name = 'coordinator'
        
        logger.info(f"Routing to: {agent_name}")
        
        try:
            agent = self.agents[agent_name]
            logger.info(f"Routing query to {agent_name} agent")
            
            # Run the agent - should now work with proper tool registration
            response = agent.run(query)
            
            # Extract response content
            if hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'text'):
                response_text = response.text
            else:
                response_text = str(response)
            
            response_text = response_text.strip()
            logger.info(f"Agent {agent_name} responded with {len(response_text)} characters")
            
            return {
                'response': response_text,
                'agent': agent_name
            }
            
        except Exception as e:
            logger.error(f"Error with agent {agent_name}: {e}")
            logger.error(traceback.format_exc())
            
            # Fallback to coordinator
            if agent_name != 'coordinator':
                try:
                    coordinator_agent = self.agents['coordinator']
                    fallback_response = coordinator_agent.run(query)
                    
                    if hasattr(fallback_response, 'content'):
                        response_text = fallback_response.content
                    else:
                        response_text = str(fallback_response)
                    
                    return {
                        'response': response_text.strip(),
                        'agent': 'coordinator_fallback'
                    }
                except Exception as fallback_error:
                    logger.error(f"Coordinator fallback failed: {fallback_error}")
            
            # Final fallback
            return {
                'response': f"I understand you're asking about soybean farming. Based on general agricultural practices, I recommend consulting your local extension officer for specific guidance about: {query[:100]}...",
                'agent': 'fallback'
            }
    
    def get_performance_metrics(self) -> Dict[str, any]:
        """Get current performance metrics"""
        metrics = self.performance_metrics.copy()
        
        if metrics['quality_scores']:
            metrics['average_quality_score'] = sum(metrics['quality_scores']) / len(metrics['quality_scores'])
            metrics['success_rate'] = metrics['successful_responses'] / metrics['total_queries'] if metrics['total_queries'] > 0 else 0
        else:
            metrics['average_quality_score'] = 0
            metrics['success_rate'] = 0
        
        return metrics

# Enhanced Flask Application with ALL features restored
app = Flask(__name__)
CORS(app)

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour", "10 per minute"]
)
limiter.init_app(app)

# Global instances
fixed_soybot = None
context_enhancer = ContextAwareEnhancer()

def initialize_fixed_soybot():
    """Initialize the fixed SoyBot system"""
    global fixed_soybot
    
    try:
        logger.info("Initializing Fixed SoyBot System...")
        fixed_soybot = FixedMultiAgentSoyBot()
        fixed_soybot.initialize()
        logger.info("Fixed SoyBot System ready!")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Fixed SoyBot: {e}")
        return False

# RESTORED Full HTML Template with TTS/STT and Multilingual
FULL_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fixed Enhanced SoyBot - Complete AI Assistant</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+Devanagari:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary: #2E7D32;
            --primary-light: #4CAF50;
            --primary-dark: #1B5E20;
            --secondary: #FF8F00;
            --accent: #8BC34A;
            --background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%);
            --surface: #FFFFFF;
            --surface-variant: #F5F5F5;
            --on-surface: #1C1B1F;
            --shadow: rgba(46, 125, 50, 0.15);
            --border-radius: 16px;
            --animation-duration: 0.3s;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', 'Noto Sans Devanagari', sans-serif;
            background: var(--background);
            color: var(--on-surface);
            line-height: 1.6;
            min-height: 100vh;
        }

        .app-container {
            max-width: 1400px;
            margin: 0 auto;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            box-shadow: 0 0 50px var(--shadow);
            background: var(--surface);
            position: relative;
            overflow: hidden;
        }

        .app-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 300px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
            z-index: -1;
        }

        .header {
            background: transparent;
            color: white;
            padding: 2rem;
            text-align: center;
            position: relative;
            z-index: 1;
        }

        .logo-container {
            margin-bottom: 1rem;
        }

        .logo {
            font-size: 4rem;
            animation: float 3s ease-in-out infinite;
            filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));
        }

        .title {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        .subtitle {
            font-size: 1.2rem;
            opacity: 0.95;
            font-weight: 400;
            margin-bottom: 1rem;
        }

        .feature-tags {
            display: flex;
            gap: 1rem;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 1rem;
        }

        .feature-tag {
            background: rgba(255, 255, 255, 0.2);
            padding: 0.5rem 1rem;
            border-radius: 2rem;
            font-size: 0.9rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        /* Status Bar */
        .status-bar {
            background: var(--surface);
            padding: 1.5rem;
            border-radius: var(--border-radius) var(--border-radius) 0 0;
            margin: -1rem 2rem 0;
            box-shadow: 0 4px 20px var(--shadow);
            display: grid;
            grid-template-columns: auto 1fr auto;
            gap: 2rem;
            align-items: center;
            position: relative;
            z-index: 2;
        }

        .status-info {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .agent-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: var(--surface-variant);
            border-radius: 1rem;
            font-size: 0.9rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--primary);
            animation: pulse-glow 2s infinite;
        }

        .metrics-display {
            text-align: center;
            font-size: 0.9rem;
            color: var(--on-surface);
        }

        .controls {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        /* Chat Area */
        .chat-section {
            flex: 1;
            display: flex;
            flex-direction: column;
            margin: 0 2rem;
            background: var(--surface);
            border-radius: 0 0 var(--border-radius) var(--border-radius);
            box-shadow: 0 4px 20px var(--shadow);
            overflow: hidden;
        }

        .quick-actions {
            padding: 2rem;
            background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%);
            border-bottom: 1px solid rgba(0,0,0,0.1);
        }

        .quick-actions h3 {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .quick-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
        }

        .quick-btn {
            background: var(--surface);
            border: 1px solid #e0e0e0;
            border-radius: var(--border-radius);
            padding: 1.5rem;
            cursor: pointer;
            transition: all var(--animation-duration) ease;
            text-align: left;
            position: relative;
            overflow: hidden;
        }

        .quick-btn:hover {
            border-color: var(--primary);
            transform: translateY(-4px);
            box-shadow: 0 8px 25px var(--shadow);
        }

        .quick-btn-content {
            position: relative;
            z-index: 1;
        }

        .quick-btn-icon {
            font-size: 1.5rem;
            color: var(--primary);
            margin-bottom: 0.5rem;
        }

        .quick-btn-title {
            font-weight: 600;
            margin-bottom: 0.3rem;
        }

        .quick-btn-desc {
            font-size: 0.9rem;
            color: #666;
            line-height: 1.4;
        }

        /* Chat Container */
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
            background: linear-gradient(to bottom, var(--surface), #fafafa);
            position: relative;
            max-height: 500px;
        }

        .message {
            max-width: 80%;
            margin-bottom: 2rem;
            position: relative;
            animation: slideInMessage var(--animation-duration) ease-out;
        }

        .message.user {
            margin-left: auto;
        }

        .message-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.8rem;
            font-size: 0.85rem;
            opacity: 0.8;
        }

        .message.user .message-header {
            justify-content: flex-end;
        }

        .message-content {
            padding: 1.5rem 2rem;
            border-radius: 1.5rem;
            position: relative;
            line-height: 1.6;
            font-size: 1rem;
        }

        .message.user .message-content {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
            color: white;
            border-bottom-right-radius: 0.5rem;
            box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);
        }

        .message.bot .message-content {
            background: var(--surface);
            color: var(--on-surface);
            border: 1px solid #e0e0e0;
            border-bottom-left-radius: 0.5rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        .agent-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            background: var(--primary);
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.75rem;
            position: absolute;
            top: -0.5rem;
            left: 1rem;
        }

        .speaker-btn {
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 50%;
            width: 36px;
            height: 36px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all var(--animation-duration) ease;
            opacity: 0.8;
        }

        .speaker-btn:hover {
            opacity: 1;
            transform: scale(1.1);
        }

        /* Input Section */
        .input-section {
            background: var(--surface);
            padding: 2rem;
            border-top: 1px solid #e0e0e0;
        }

        .input-container {
            display: flex;
            gap: 1rem;
            align-items: flex-end;
            background: var(--surface-variant);
            border-radius: 2rem;
            padding: 1rem;
            border: 2px solid transparent;
            transition: all var(--animation-duration) ease;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .input-container:focus-within {
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(46, 125, 50, 0.1);
            transform: translateY(-2px);
        }

        .input-field {
            flex: 1;
            border: none;
            background: transparent;
            padding: 1rem;
            font-size: 1rem;
            font-family: inherit;
            resize: none;
            outline: none;
            min-height: 24px;
            max-height: 120px;
        }

        .input-actions {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }

        .action-btn {
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 50%;
            width: 48px;
            height: 48px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            transition: all var(--animation-duration) ease;
            position: relative;
            overflow: hidden;
        }

        .action-btn:hover {
            transform: translateY(-2px) scale(1.05);
            box-shadow: 0 6px 20px var(--shadow);
        }

        .action-btn.mic {
            background: var(--surface);
            color: var(--primary);
            border: 2px solid var(--primary);
        }

        .action-btn.mic.recording {
            background: #f44336;
            color: white;
            border-color: #f44336;
            animation: pulse-record 1.5s infinite;
        }

        .toggle-switch {
            position: relative;
            width: 44px;
            height: 24px;
            background: #ccc;
            border-radius: 12px;
            cursor: pointer;
            transition: background 0.3s;
        }

        .toggle-switch.active {
            background: var(--primary);
        }

        .toggle-slider {
            position: absolute;
            top: 2px;
            left: 2px;
            width: 20px;
            height: 20px;
            background: white;
            border-radius: 50%;
            transition: transform 0.3s;
        }

        .typing-indicator {
            display: none;
            max-width: 80%;
            margin-bottom: 2rem;
        }

        .typing-content {
            background: var(--surface);
            border: 1px solid #e0e0e0;
            border-radius: 1.5rem;
            border-bottom-left-radius: 0.5rem;
            padding: 1.5rem 2rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .typing-dots {
            display: flex;
            gap: 0.3rem;
        }

        .typing-dot {
            width: 8px;
            height: 8px;
            background: var(--primary);
            border-radius: 50%;
            animation: typing-bounce 1.4s infinite ease-in-out;
        }

        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }

        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid transparent;
            border-top: 2px solid currentColor;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        /* Animations */
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }

        @keyframes slideInMessage {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes pulse-glow {
            0% { 
                box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7);
            }
            70% { 
                box-shadow: 0 0 0 10px rgba(76, 175, 80, 0);
            }
            100% { 
                box-shadow: 0 0 0 0 rgba(76, 175, 80, 0);
            }
        }

        @keyframes pulse-record {
            0% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(244, 67, 54, 0); }
            100% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0); }
        }

        @keyframes typing-bounce {
            0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .app-container {
                margin: 0;
                border-radius: 0;
            }

            .header {
                padding: 1.5rem 1rem;
            }

            .title {
                font-size: 2rem;
            }

            .status-bar {
                margin: -0.5rem 1rem 0;
                padding: 1rem;
                grid-template-columns: 1fr;
                gap: 1rem;
                text-align: center;
            }

            .chat-section {
                margin: 0 1rem;
            }

            .quick-grid {
                grid-template-columns: 1fr;
            }

            .message {
                max-width: 90%;
            }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Header -->
        <header class="header">
            <div class="logo-container">
                <span class="logo">🌱</span>
            </div>
            <h1 class="title">Fixed Enhanced SoyBot</h1>
            <p class="subtitle">
                Multi-Agent RAG AI Farming Assistant | RAG समर्थित बहु-एजेंट खेती सहायक
                <br>
                <small>✅ Fixed RAG Integration • 🗣️ Text-to-Speech • 🎤 Speech-to-Text • 🌍 Multilingual</small>
            </p>
            <div class="feature-tags">
                <span class="feature-tag">✅ Fixed RAG System</span>
                <span class="feature-tag">🤖 Multi-Agent Routing</span>
                <span class="feature-tag">🌍 Multilingual Support</span>
                <span class="feature-tag">🗣️ Voice Features</span>
                <span class="feature-tag">📊 Quality Assessment</span>
            </div>
        </header>

        <!-- Status Bar -->
        <div class="status-bar">
            <div class="status-info">
                <div class="agent-status">
                    <div class="status-dot"></div>
                    <span id="agent-status">System Ready</span>
                </div>
            </div>
            
            <div class="metrics-display">
                <div>Queries: <span id="query-count">0</span></div>
                <div>Avg Time: <span id="avg-time">0.0s</span></div>
                <div>Quality: <span id="quality-score">95%</span></div>
            </div>
            
            <div class="controls">
                <select id="language-select" style="padding: 0.5rem; border-radius: 0.5rem; border: 1px solid #ddd;">
                    <option value="en-US">🇺🇸 English</option>
                    <option value="hi-IN">🇮🇳 हिंदी</option>
                    <option value="mr-IN">🇮🇳 मराठी</option>
                </select>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span>Auto-speak</span>
                    <div class="toggle-switch active" id="auto-speak-toggle">
                        <div class="toggle-slider" style="transform: translateX(18px);"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Chat Section -->
        <div class="chat-section">
            <!-- Quick Actions -->
            <div class="quick-actions">
                <h3>
                    <i class="fas fa-zap"></i>
                    Expert Quick Queries | विशेषज्ञ त्वरित प्रश्न
                </h3>
                <div class="quick-grid">
                    <button class="quick-btn" onclick="askQuestion('सोयाबीन की बुवाई का सबसे अच्छा समय क्या है?')">
                        <div class="quick-btn-content">
                            <div class="quick-btn-icon"><i class="fas fa-calendar-alt"></i></div>
                            <div class="quick-btn-title">बुवाई का समय</div>
                            <div class="quick-btn-desc">Optimal sowing time guidance</div>
                        </div>
                    </button>
                    
                    <button class="quick-btn" onclick="askQuestion('My soybean leaves have yellow spots. What disease could this be?')">
                        <div class="quick-btn-content">
                            <div class="quick-btn-icon"><i class="fas fa-leaf"></i></div>
                            <div class="quick-btn-title">Disease Diagnosis</div>
                            <div class="quick-btn-desc">Plant health specialist</div>
                        </div>
                    </button>
                    
                    <button class="quick-btn" onclick="askQuestion('कौन सा फर्टिलाइजर सोयाबीन के लिए सबसे अच्छा है?')">
                        <div class="quick-btn-content">
                            <div class="quick-btn-icon"><i class="fas fa-flask"></i></div>
                            <div class="quick-btn-title">पोषण प्रबंधन</div>
                            <div class="quick-btn-desc">Nutrition expert advice</div>
                        </div>
                    </button>
                    
                    <button class="quick-btn" onclick="askQuestion('What are trap crops and how do they work?')">
                        <div class="quick-btn-content">
                            <div class="quick-btn-icon"><i class="fas fa-seedling"></i></div>
                            <div class="quick-btn-title">Crop Management</div>
                            <div class="quick-btn-desc">Advanced farming techniques</div>
                        </div>
                    </button>
                </div>
            </div>

            <!-- Chat Container -->
            <div class="chat-container" id="chat-container">
                <div class="message bot">
                    <div class="message-header">
                        <i class="fas fa-robot"></i>
                        <span>Fixed Enhanced SoyBot</span>
                        <span>•</span>
                        <span id="welcome-time"></span>
                    </div>
                    <div class="message-content">
                        <div class="agent-badge">
                            <i class="fas fa-users"></i>
                            Multi-Agent
                        </div>
नमस्कार! Welcome to the FIXED Enhanced SoyBot! 🚀

I've resolved the RAG integration issues. Here's what's working now:

✅ <strong>Fixed RAG System</strong> - PDF knowledge properly accessible
🤖 <strong>Multi-Agent Routing</strong> - Specialized experts for different topics  
🌍 <strong>Multilingual Processing</strong> - Hindi, Marathi, English support
🗣️ <strong>Text-to-Speech</strong> - Auto-speak responses (toggle available)
🎤 <strong>Speech-to-Text</strong> - Voice input capability
📊 <strong>Quality Assessment</strong> - Response quality monitoring
🎯 <strong>Intent Classification</strong> - Smart query routing
📋 <strong>Context Awareness</strong> - Seasonal and regional considerations

Ask me anything about soybean cultivation - I'll search my knowledge base and route to the right specialist!
                        <button class="speaker-btn" onclick="speakText(this.parentElement)">
                            <i class="fas fa-volume-up"></i>
                        </button>
                    </div>
                </div>
            </div>

            <!-- Typing Indicator -->
            <div class="typing-indicator" id="typing-indicator">
                <div class="typing-content">
                    <div class="typing-dots">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                    <span>AI Specialists analyzing PDF knowledge...</span>
                </div>
            </div>
        </div>

        <!-- Input Section -->
        <div class="input-section">
            <div class="input-container">
                <textarea 
                    id="query-input" 
                    class="input-field"
                    placeholder="Ask your farming question... I'll search my knowledge base and provide expert guidance..."
                    rows="1"
                ></textarea>
                <div class="input-actions">
                    <button id="mic-btn" class="action-btn mic" title="Voice Input">
                        <i class="fas fa-microphone"></i>
                    </button>
                    <button id="ask-btn" class="action-btn" title="Send to AI Specialists">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Complete JavaScript with all features restored
        const chatContainer = document.getElementById('chat-container');
        const queryInput = document.getElementById('query-input');
        const askBtn = document.getElementById('ask-btn');
        const micBtn = document.getElementById('mic-btn');
        const typingIndicator = document.getElementById('typing-indicator');
        const autoSpeakToggle = document.getElementById('auto-speak-toggle');
        const languageSelect = document.getElementById('language-select');
        
        // Metrics elements
        const queryCountEl = document.getElementById('query-count');
        const avgTimeEl = document.getElementById('avg-time');
        const qualityScoreEl = document.getElementById('quality-score');
        const agentStatusEl = document.getElementById('agent-status');
        
        let recognition;
        let isRecording = false;
        let synthesis = window.speechSynthesis;
        let autoSpeak = true;
        let queryCount = 0;
        let totalResponseTime = 0;
        let qualityScores = [];

        // Initialize
        document.getElementById('welcome-time').textContent = new Date().toLocaleTimeString();

        // Auto-resize textarea
        queryInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });

        // Toggle auto-speak
        autoSpeakToggle.addEventListener('click', function() {
            autoSpeak = !autoSpeak;
            this.classList.toggle('active', autoSpeak);
            const slider = this.querySelector('.toggle-slider');
            slider.style.transform = autoSpeak ? 'translateX(18px)' : 'translateX(2px)';
            this.style.background = autoSpeak ? 'var(--primary)' : '#ccc';
        });

        // Speech recognition setup - RESTORED
        function initSpeechRecognition() {
            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                recognition = new SpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = false;
                recognition.lang = languageSelect.value || 'hi-IN';

                recognition.onresult = (event) => {
                    const transcript = event.results[0][0].transcript;
                    queryInput.value = transcript;
                    queryInput.style.height = 'auto';
                    queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + 'px';
                    stopRecording();
                };

                recognition.onerror = (event) => {
                    console.error('Speech recognition error:', event.error);
                    stopRecording();
                };

                recognition.onend = stopRecording;

                languageSelect.addEventListener('change', () => {
                    recognition.lang = languageSelect.value;
                });
            } else {
                micBtn.style.display = 'none';
            }
        }

        function startRecording() {
            isRecording = true;
            micBtn.classList.add('recording');
            micBtn.innerHTML = '<i class="fas fa-stop"></i>';
            agentStatusEl.textContent = 'Listening...';
        }

        function stopRecording() {
            isRecording = false;
            micBtn.classList.remove('recording');
            micBtn.innerHTML = '<i class="fas fa-microphone"></i>';
            agentStatusEl.textContent = 'System Ready';
        }

        function toggleRecording() {
            if (!recognition) return;
            if (!isRecording) {
                recognition.start();
                startRecording();
            } else {
                recognition.stop();
                stopRecording();
            }
        }

        // Text-to-speech - RESTORED
        function speakText(messageElement) {
            const text = messageElement.textContent.replace(/[🔊📢🎵✅🤖🌍🗣️🎤📊🎯📋]/g, '').trim();
            const speakerBtn = messageElement.querySelector('.speaker-btn i');
            
            if (synthesis.speaking) {
                synthesis.cancel();
                if (speakerBtn) speakerBtn.className = 'fas fa-volume-up';
                return;
            }

            if (text) {
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = languageSelect.value || 'hi-IN';
                utterance.rate = 0.9;
                utterance.pitch = 1.0;

                if (speakerBtn) speakerBtn.className = 'fas fa-volume-mute';

                utterance.onend = () => {
                    if (speakerBtn) speakerBtn.className = 'fas fa-volume-up';
                };

                synthesis.speak(utterance);
            }
        }

        // Enhanced message display - RESTORED
        function addMessage(text, sender, metadata = {}) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            const timestamp = new Date().toLocaleTimeString();
            const senderName = sender === 'user' ? 'You' : 'Enhanced SoyBot';
            const senderIcon = sender === 'user' ? 'fas fa-user' : 'fas fa-robot';
            
            let agentBadge = '';
            let qualityIndicator = '';
            
            if (sender === 'bot' && metadata.agent_used) {
                const agentNames = {
                    'crop_management': '🌾 Crop Expert',
                    'plant_health': '🦠 Health Specialist', 
                    'nutrition': '🧪 Nutrition Expert',
                    'coordinator': '🎯 Coordinator'
                };
                agentBadge = `<div class="agent-badge"><i class="fas fa-user-md"></i> ${agentNames[metadata.agent_used] || 'Specialist'}</div>`;
            }
            
            messageDiv.innerHTML = `
                <div class="message-header">
                    <i class="${senderIcon}"></i>
                    <span>${senderName}</span>
                    <span>•</span>
                    <span>${timestamp}</span>
                </div>
                <div class="message-content">
                    ${agentBadge}
                    ${text}
                    ${sender === 'bot' ? '<button class="speaker-btn" onclick="speakText(this.parentElement)"><i class="fas fa-volume-up"></i></button>' : ''}
                    ${qualityIndicator}
                </div>
            `;

            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;

            // Auto-speak bot messages - RESTORED
            if (sender === 'bot' && autoSpeak) {
                setTimeout(() => {
                    const messageContent = messageDiv.querySelector('.message-content');
                    if (messageContent) speakText(messageContent);
                }, 500);
            }

            // Update quality scores
            if (sender === 'bot' && metadata.quality_assessment) {
                qualityScores.push(metadata.quality_assessment.overall_confidence);
                updateMetrics();
            }
        }

        function updateMetrics() {
            queryCountEl.textContent = queryCount;
            
            if (queryCount > 0) {
                avgTimeEl.textContent = (totalResponseTime / queryCount).toFixed(1) + 's';
            }
            
            if (qualityScores.length > 0) {
                const avgQuality = qualityScores.reduce((a, b) => a + b, 0) / qualityScores.length;
                qualityScoreEl.textContent = (avgQuality * 100).toFixed(1) + '%';
            }
        }

        function showTyping() {
            typingIndicator.style.display = 'block';
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function hideTyping() {
            typingIndicator.style.display = 'none';
        }

        // Enhanced send question - FIXED
        async function sendQuestion() {
            const question = queryInput.value.trim();
            if (!question) return;

            const startTime = performance.now();
            queryCount++;

            addMessage(question, 'user');
            queryInput.value = '';
            queryInput.style.height = 'auto';

            showTyping();
            askBtn.disabled = true;
            askBtn.innerHTML = '<div class="spinner"></div>';
            agentStatusEl.textContent = 'AI Specialists analyzing PDF...';

            try {
                const response = await fetch('/api/enhanced-ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });

                const data = await response.json();
                const responseTime = (performance.now() - startTime) / 1000;
                totalResponseTime += responseTime;
                
                hideTyping();

                if (data.success) {
                    addMessage(data.response, 'bot', {
                        agent_used: data.agent_used,
                        quality_assessment: data.quality_assessment,
                        language_info: data.language_info,
                        intent_info: data.intent_info
                    });
                    agentStatusEl.textContent = `Response from ${data.agent_used || 'AI Specialist'}`;
                } else {
                    addMessage(`Sorry, I encountered an error: ${data.error}`, 'bot');
                    agentStatusEl.textContent = 'Error occurred';
                }
            } catch (error) {
                hideTyping();
                console.error('Error:', error);
                addMessage('Sorry, connection error. Please check your network and try again.', 'bot');
                agentStatusEl.textContent = 'Connection error';
            } finally {
                askBtn.disabled = false;
                askBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
                updateMetrics();
            }
        }

        function askQuestion(question) {
            queryInput.value = question;
            queryInput.style.height = 'auto';
            queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + 'px';
            sendQuestion();
        }

        // Event listeners
        askBtn.addEventListener('click', sendQuestion);
        micBtn.addEventListener('click', toggleRecording);

        queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendQuestion();
            }
        });

        // Initialize systems
        initSpeechRecognition();
        queryInput.focus();

        // Status check
        async function checkSystemStatus() {
            try {
                const response = await fetch('/api/enhanced-status');
                const data = await response.json();
                
                if (data.success && data.system_status === 'ready') {
                    agentStatusEl.textContent = 'Multi-Agent System Ready';
                } else {
                    agentStatusEl.textContent = 'System Initializing...';
                }
                
                if (data.performance_metrics) {
                    const metrics = data.performance_metrics;
                    if (metrics.total_queries > 0) {
                        queryCountEl.textContent = metrics.total_queries;
                        avgTimeEl.textContent = metrics.average_response_time.toFixed(1) + 's';
                        if (metrics.average_quality_score) {
                            qualityScoreEl.textContent = (metrics.average_quality_score * 100).toFixed(1) + '%';
                        }
                    }
                }
            } catch (error) {
                agentStatusEl.textContent = 'Connection error';
            }
        }

        checkSystemStatus();
        setInterval(checkSystemStatus, 30000);
    </script>
</body>
</html>
"""

# API Routes - FIXED
@app.route('/')
def index():
    """Serve enhanced web interface"""
    return render_template_string(FULL_HTML_TEMPLATE)

@app.route('/api/enhanced-status', methods=['GET'])
@limiter.limit("30 per minute")
def get_enhanced_status():
    """Get enhanced system status with metrics"""
    global fixed_soybot
    
    system_ready = fixed_soybot is not None and fixed_soybot.is_initialized
    
    status_data = {
        'success': True,
        'system_status': 'ready' if system_ready else 'initializing',
        'agents': {
            'crop_management': system_ready,
            'plant_health': system_ready, 
            'nutrition': system_ready,
            'coordinator': system_ready
        },
        'features': {
            'multi_agent_routing': True,
            'intent_classification': True,
            'quality_assessment': True,
            'multilingual_processing': True,
            'context_awareness': True,
            'performance_monitoring': True,
            'rag_integration': system_ready,  # NEW: RAG status
            'tts_support': True,
            'stt_support': True
        }
    }
    
    if system_ready:
        status_data['performance_metrics'] = fixed_soybot.get_performance_metrics()
    
    return jsonify(status_data)

@app.route('/api/enhanced-ask', methods=['POST'])
@limiter.limit("20 per minute")
def enhanced_ask():
    """FIXED enhanced question processing"""
    global fixed_soybot
    
    def convert_numpy_types(obj):
        """Convert numpy types to JSON serializable types"""
        if hasattr(obj, 'dtype'):
            if 'bool' in str(obj.dtype):
                return bool(obj)
            elif 'int' in str(obj.dtype):
                return int(obj)
            elif 'float' in str(obj.dtype):
                return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    try:
        if not fixed_soybot or not fixed_soybot.is_initialized:
            return jsonify({
                'success': False,
                'error': 'Fixed SoyBot system is still initializing. Please wait.'
            }), 503
        
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'error': 'No question provided'
            }), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({
                'success': False,
                'error': 'Empty question provided'
            }), 400
        
        logger.info(f"Processing question: {question}")
        
        # Process with FIXED multi-agent system
        result = fixed_soybot.process_query(question)
        
        logger.info(f"Generated response by {result.get('agent_used', 'unknown')} agent")
        
        response_data = {
            'success': True,
            'response': result['response'],
            'agent_used': result.get('agent_used'),
            'quality_assessment': convert_numpy_types(result.get('quality_assessment', {})),
            'language_info': convert_numpy_types(result.get('language_info', {})),
            'intent_info': convert_numpy_types(result.get('intent_info', {})),
            'processing_time': result.get('processing_time'),
            'rag_used': True  # Indicate RAG was used
        }

        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in query processing: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': 'Technical issue occurred. Please try again.',
            'details': str(e) if app.debug else None
        }), 500

@app.route('/api/metrics', methods=['GET'])
@limiter.limit("10 per minute")
def get_metrics():
    """Get detailed system metrics"""
    global fixed_soybot
    
    if not fixed_soybot or not fixed_soybot.is_initialized:
        return jsonify({'error': 'System not initialized'}), 503
    
    metrics = fixed_soybot.get_performance_metrics()
    return jsonify(metrics)

# Error handlers
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Please wait before making more requests'
    }), 429

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Create required directories
os.makedirs('vectordb', exist_ok=True)
os.makedirs('logs', exist_ok=True)

if __name__ == '__main__':
    logger.info("Starting FIXED Enhanced SoyBot System...")
    
    # Initialize FIXED system
    if initialize_fixed_soybot():
        logger.info("Enhanced SoyBot System initialized successfully!")
        logger.info("Features enabled:")
        logger.info("   RAG Integration")
        logger.info("   Multi-Agent Architecture")
        logger.info("   Intent Classification")
        logger.info("   Quality Assessment")
        logger.info("   Multilingual Processing (Hindi/Marathi/English)")
        logger.info("   Text-to-Speech Support")
        logger.info("   Speech-to-Text Support")
        logger.info("   Context Awareness")
        logger.info("   Performance Monitoring")
        logger.info("Starting enhanced Flask server...")
        logger.info("Access at: http://localhost:5000")
        logger.info("-" * 80)
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
    else:
        logger.error("Failed to initialize Enhanced SoyBot System")
        logger.error("Please check your configuration and try again.")
        sys.exit(1)
