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
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Language processing
import langdetect
from googletrans import Translator
import re

# Phi components
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
    WEATHER_CONTEXT = "weather_context"

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
    context_requirements: List[str]

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
    
    def preprocess_multilingual_query(self, text: str, lang_info: Dict) -> str:
        """Preprocess multilingual queries for better understanding"""
        processed_text = text
        
        if lang_info['romanized_hindi']:
            processed_text = self.transliterate_roman_to_devanagari(processed_text)
        
        if lang_info['code_switched']:
            processed_text = self.handle_code_switching(processed_text)
        
        return processed_text
    
    def transliterate_roman_to_devanagari(self, text: str) -> str:
        """Basic transliteration of Roman Hindi to Devanagari"""
        for roman, devanagari in self.hindi_romanized_patterns.items():
            text = text.replace(roman, f"{roman} ({devanagari})")
        return text
    
    def handle_code_switching(self, text: str) -> str:
        """Handle code-switched text"""
        # For now, just add language markers
        return f"[Mixed Language] {text}"

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
                'blight', 'rust', 'fungus', 'viral', 'bacterial', 'symptoms'
            ],
            QueryIntent.PEST_CONTROL: [
                'pest', 'insect', 'bug', 'caterpillar', 'aphid', 'thrips',
                'control', 'spray', 'pesticide', 'damage', 'eating', 'larvae'
            ],
            QueryIntent.FERTILIZER_ADVICE: [
                'fertilizer', 'nutrients', 'nitrogen', 'phosphorus', 'potassium',
                'NPK', 'organic', 'compost', 'manure', 'feeding', 'nutrition'
            ],
            QueryIntent.PLANTING_GUIDANCE: [
                'planting', 'sowing', 'seeding', 'when to plant', 'plant spacing',
                'depth', 'germination', 'varieties', 'cultivar', 'hybrid'
            ],
            QueryIntent.HARVESTING_INFO: [
                'harvest', 'harvesting', 'maturity', 'ready', 'picking',
                'yield', 'timing', 'storage', 'drying', 'processing'
            ],
            QueryIntent.IRRIGATION: [
                'water', 'irrigation', 'watering', 'moisture', 'drought',
                'rain', 'drip', 'sprinkler', 'flooding', 'dry', 'wet'
            ]
        }
        
        # Load pre-trained model if available
        self.load_model()
    
    def create_training_dataset(self) -> Dict[str, List[str]]:
        """Create synthetic training data for agricultural queries"""
        training_queries = []
        training_labels = []
        
        # Generate training examples for each intent
        for intent, keywords in self.intent_keywords.items():
            # Create query templates
            templates = [
                "How to {keyword} in soybean?",
                "What is the best {keyword} for soybean?",
                "When should I {keyword} my soybean crop?",
                "Problem with {keyword} in soybean",
                "Soybean {keyword} management",
                "Need advice on {keyword}",
                "{keyword} in soybean cultivation"
            ]
            
            for keyword in keywords:
                for template in templates:
                    query = template.format(keyword=keyword)
                    training_queries.append(query)
                    training_labels.append(intent.value)
        
        return {'queries': training_queries, 'labels': training_labels}
    
    def train_classifier(self):
        """Train the intent classifier"""
        if self.is_trained:
            return
        
        logger.info("Training intent classifier...")
        training_data = self.create_training_dataset()
        
        X = self.vectorizer.fit_transform(training_data['queries'])
        y = training_data['labels']
        
        self.classifier.fit(X, y)
        self.is_trained = True
        
        # Save the trained model
        self.save_model()
        logger.info("Intent classifier trained successfully")
    
    def classify_intent(self, query: str) -> Dict[str, any]:
        """Classify query intent with confidence scores"""
        if not self.is_trained:
            self.train_classifier()
        
        # Keyword-based fallback for untrained scenarios
        keyword_scores = {}
        query_lower = query.lower()
        
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            keyword_scores[intent.value] = score / len(keywords)
        
        if sum(keyword_scores.values()) > 0:
            primary_intent = max(keyword_scores, key=keyword_scores.get)
            confidence = keyword_scores[primary_intent]
        else:
            # Use ML classifier
            try:
                query_vector = self.vectorizer.transform([query])
                intent_proba = self.classifier.predict_proba(query_vector)[0]
                intent_labels = self.classifier.classes_
                
                intent_scores = dict(zip(intent_labels, intent_proba))
                primary_intent = max(intent_scores, key=intent_scores.get)
                confidence = intent_scores[primary_intent]
            except:
                primary_intent = QueryIntent.GENERAL_QUERY.value
                confidence = 0.5
        
        return {
            'primary_intent': primary_intent,
            'confidence': confidence,
            'all_scores': keyword_scores
        }
    
    def save_model(self):
        """Save trained model"""
        try:
            model_data = {
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'is_trained': self.is_trained
            }
            with open('models/intent_classifier.pkl', 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            logger.warning(f"Could not save model: {e}")
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            with open('models/intent_classifier.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.vectorizer = model_data['vectorizer']
                self.classifier = model_data['classifier']
                self.is_trained = model_data['is_trained']
                logger.info("Loaded pre-trained intent classifier")
        except:
            logger.info("No pre-trained model found, will train on first use")

class IntelligentKnowledgeBase:
    """Intelligent knowledge base with topic-based segmentation"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.topic_extractors = {
            'diseases': ['disease', 'pest', 'infection', 'symptom', 'pathogen', 'fungal', 'bacterial', 'viral'],
            'nutrition': ['fertilizer', 'nutrient', 'nitrogen', 'phosphorus', 'potassium', 'NPK', 'organic'],
            'cultivation': ['planting', 'sowing', 'harvesting', 'irrigation', 'spacing', 'depth'],
            'varieties': ['variety', 'cultivar', 'seed', 'hybrid', 'genetic', 'breeding']
        }
        self.knowledge_bases = {}
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def create_topic_specific_databases(self) -> Dict[str, PDFKnowledgeBase]:
        """Create topic-specific knowledge bases"""
        logger.info("Creating topic-specific knowledge bases...")
        
        try:
            # Create main knowledge base
            main_kb = PDFKnowledgeBase(
                path=self.pdf_path,
                vector_db=LanceDb(
                    table_name="soybean_main",
                    uri="./vectordb/soybot_main_db",
                    search_type=SearchType.hybrid,
                    embedder=SentenceTransformerEmbedder(model="all-MiniLM-L6-v2"),
                )
            )
            
            main_kb.load(recreate=False)
            self.knowledge_bases['main'] = main_kb
            
            # For now, use the main KB for all topics
            # In production, you would segment the PDF content by topics
            for topic in self.topic_extractors.keys():
                self.knowledge_bases[topic] = main_kb
            
            logger.info("Knowledge bases created successfully")
            return self.knowledge_bases
            
        except Exception as e:
            logger.error(f"Error creating knowledge bases: {e}")
            raise
    
    def get_relevant_knowledge_base(self, intent: QueryIntent) -> PDFKnowledgeBase:
        """Get the most relevant knowledge base for a given intent"""
        intent_to_topic = {
            QueryIntent.DISEASE_DIAGNOSIS: 'diseases',
            QueryIntent.PEST_CONTROL: 'diseases',
            QueryIntent.FERTILIZER_ADVICE: 'nutrition',
            QueryIntent.PLANTING_GUIDANCE: 'cultivation',
            QueryIntent.HARVESTING_INFO: 'cultivation',
            QueryIntent.VARIETY_SELECTION: 'varieties',
            QueryIntent.IRRIGATION: 'cultivation'
        }
        
        topic = intent_to_topic.get(intent, 'main')
        return self.knowledge_bases.get(topic, self.knowledge_bases['main'])

class ResponseQualityAssessor:
    """Assess and improve response quality"""
    
    def __init__(self):
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
    def assess_response_quality(self, query: str, response: str, 
                              knowledge_sources: List = None) -> ResponseQuality:
        """Comprehensive response quality assessment"""
        
        # Calculate relevance using semantic similarity
        relevance = self._calculate_relevance(query, response)
        
        # Assess completeness
        completeness = self._assess_completeness(query, response)
        
        # Check accuracy indicators
        accuracy = self._check_accuracy(response)
        
        # Assess actionability
        actionability = self._assess_actionability(response)
        
        # Calculate overall confidence
        overall_confidence = (relevance + completeness + accuracy + actionability) / 4
        
        # Determine if refinement is needed
        needs_refinement = overall_confidence < 0.7
        
        # Generate improvement suggestions
        suggested_improvements = self._suggest_improvements({
            'relevance': relevance,
            'completeness': completeness,
            'accuracy': accuracy,
            'actionability': actionability
        })
        
        return ResponseQuality(
            relevance=relevance,
            completeness=completeness,
            accuracy=accuracy,
            actionability=actionability,
            overall_confidence=overall_confidence,
            needs_refinement=needs_refinement,
            suggested_improvements=suggested_improvements
        )
    
    def _calculate_relevance(self, query: str, response: str) -> float:
        """Calculate semantic similarity between query and response"""
        try:
            query_embedding = self.sentence_transformer.encode([query])
            response_embedding = self.sentence_transformer.encode([response])
            similarity = cosine_similarity(query_embedding, response_embedding)[0][0]
            return max(0.0, min(1.0, similarity))
        except:
            return 0.5
    
    def _assess_completeness(self, query: str, response: str) -> float:
        """Assess if the response adequately addresses the query"""
        # Simple heuristic based on response length and structure
        response_length = len(response.split())
        
        if response_length < 20:
            return 0.3
        elif response_length < 50:
            return 0.6
        elif response_length < 100:
            return 0.8
        else:
            return 0.9
    
    def _check_accuracy(self, response: str) -> float:
        """Check for accuracy indicators in the response"""
        accuracy_indicators = [
            'according to', 'based on', 'research shows', 'studies indicate',
            'ICAR', 'recommended', 'proven', 'scientific', 'expert'
        ]
        
        uncertainty_indicators = [
            'maybe', 'possibly', 'might', 'could be', 'not sure',
            'probably', 'I think', 'perhaps'
        ]
        
        response_lower = response.lower()
        accuracy_count = sum(1 for indicator in accuracy_indicators 
                           if indicator in response_lower)
        uncertainty_count = sum(1 for indicator in uncertainty_indicators 
                              if indicator in response_lower)
        
        # Higher accuracy score for more authoritative language
        if accuracy_count > uncertainty_count:
            return min(0.9, 0.5 + (accuracy_count * 0.1))
        else:
            return max(0.3, 0.7 - (uncertainty_count * 0.1))
    
    def _assess_actionability(self, response: str) -> float:
        """Assess how actionable the response is"""
        action_words = [
            'apply', 'use', 'plant', 'sow', 'harvest', 'spray', 'water',
            'fertilize', 'treat', 'monitor', 'check', 'maintain', 'follow',
            'step', 'first', 'then', 'next', 'finally'
        ]
        
        response_lower = response.lower()
        action_count = sum(1 for word in action_words if word in response_lower)
        
        # Check for numbered steps or bullet points
        has_structure = any(pattern in response for pattern in 
                          ['1.', '2.', '•', '-', 'Step 1', 'First'])
        
        base_score = min(0.8, action_count * 0.1)
        if has_structure:
            base_score += 0.2
            
        return min(1.0, base_score)
    
    def _suggest_improvements(self, scores: Dict[str, float]) -> List[str]:
        """Suggest improvements based on quality scores"""
        suggestions = []
        
        if scores['relevance'] < 0.6:
            suggestions.append("Improve relevance to the specific question asked")
        
        if scores['completeness'] < 0.6:
            suggestions.append("Provide more comprehensive information")
        
        if scores['accuracy'] < 0.6:
            suggestions.append("Include more authoritative sources and references")
        
        if scores['actionability'] < 0.6:
            suggestions.append("Add more specific, actionable steps")
        
        return suggestions

class MultiAgentSoyBot:
    """Enhanced multi-agent SoyBot system"""
    
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.language_processor = AdvancedLanguageProcessor()
        self.intent_classifier = QueryIntentClassifier()
        self.knowledge_base = IntelligentKnowledgeBase("Soybeanpackageofpractices.pdf")
        self.quality_assessor = ResponseQualityAssessor()
        
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
            logger.info("Initializing Enhanced SoyBot...")
            
            # Create knowledge bases
            knowledge_bases = self.knowledge_base.create_topic_specific_databases()
            
            # Initialize specialized agents
            self._create_specialized_agents(knowledge_bases)
            
            # Train intent classifier
            self.intent_classifier.train_classifier()
            
            self.is_initialized = True
            logger.info("Enhanced SoyBot initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing SoyBot: {e}")
            raise
    
    def _create_specialized_agents(self, knowledge_bases: Dict[str, PDFKnowledgeBase]):
        """Create specialized agents for different domains"""
        
        base_instructions = [
            "You are an expert soybean farming advisor based on ICAR-IISR guidelines.",
            "Provide practical, actionable advice based on scientific knowledge.",
            "Use simple, clear language that farmers can easily understand.",
            "Always respond in the same language as the question was asked.",
            "Structure your responses with clear points when giving multiple recommendations."
        ]
        
        # Crop Management Specialist
        self.agents['crop_management'] = Agent(
            name="Crop Management Specialist",
            role="Expert in planting, irrigation, and growth stages",
            model=Groq(id="llama-3.3-70b-versatile", api_key=self.groq_api_key),
            knowledge=knowledge_bases.get('cultivation', knowledge_bases['main']),
            instructions=base_instructions + [
                "Focus on planting schedules, irrigation, and crop growth stages.",
                "Provide season-specific recommendations.",
                "Consider soil preparation and seed selection."
            ],
            show_tool_calls=False,
            markdown=False
        )
        
        # Plant Health Specialist
        self.agents['plant_health'] = Agent(
            name="Plant Health Specialist",
            role="Expert in disease diagnosis and pest management",
            model=Groq(id="llama-3.3-70b-versatile", api_key=self.groq_api_key),
            knowledge=knowledge_bases.get('diseases', knowledge_bases['main']),
            instructions=base_instructions + [
                "Diagnose plant diseases from symptoms described.",
                "Recommend integrated pest management strategies.",
                "Suggest both organic and chemical treatment options.",
                "Provide prevention strategies."
            ],
            show_tool_calls=False,
            markdown=False
        )
        
        # Soil & Nutrition Expert
        self.agents['nutrition'] = Agent(
            name="Soil & Nutrition Expert",
            role="Expert in fertilizer and soil health management",
            model=Groq(id="llama-3.3-70b-versatile", api_key=self.groq_api_key),
            knowledge=knowledge_bases.get('nutrition', knowledge_bases['main']),
            instructions=base_instructions + [
                "Recommend fertilizer schedules and nutrient management.",
                "Assess soil nutrient deficiencies from symptoms.",
                "Suggest organic amendments and soil improvement methods."
            ],
            show_tool_calls=False,
            markdown=False
        )
        
        # General Coordinator
        self.agents['coordinator'] = Agent(
            name="SoyBot Coordinator",
            role="Route queries and provide general farming advice",
            model=Groq(id="llama-3.3-70b-versatile", api_key=self.groq_api_key),
            knowledge=knowledge_bases['main'],
            instructions=base_instructions + [
                "Handle general queries and coordinate with specialists when needed.",
                "Provide comprehensive farming advice.",
                "Synthesize information from multiple domains when necessary."
            ],
            show_tool_calls=False,
            markdown=False
        )
    

    def performance_monitor(func):
    
        @wraps(func)
        def wrapper(self, *args, **kwargs):  # 'self' should be the first parameter
            start_time = time.time()
            self.performance_metrics['total_queries'] += 1
            
            try:
                result = func(self, *args, **kwargs)  # Pass 'self' to the original function
                self.performance_metrics['successful_responses'] += 1
                
                response_time = time.time() - start_time
                
                # Update average response time
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
        """Process query using multi-agent approach"""
        
        # Language analysis
        lang_info = self.language_processor.enhanced_language_detection(query)
        
        # Preprocess query
        processed_query = self.language_processor.preprocess_multilingual_query(query, lang_info)
        
        # Intent classification
        intent_info = self.intent_classifier.classify_intent(query)
        
        # Route to appropriate agent
        agent_response = self._route_query_to_agent(processed_query, intent_info)
        
        # Quality assessment
        quality_assessment = self.quality_assessor.assess_response_quality(
            query, agent_response['response']
        )
        
        # Store quality metrics
        self.performance_metrics['quality_scores'].append(quality_assessment.overall_confidence)
        
        # Enhance response if needed
        if quality_assessment.needs_refinement:
            agent_response['response'] = self._enhance_low_confidence_response(
                query, agent_response['response'], quality_assessment
            )
        
        return {
            'response': agent_response['response'],
            'agent_used': agent_response['agent'],
            'language_info': lang_info,
            'intent_info': intent_info,
            'quality_assessment': quality_assessment.__dict__,
            'processing_time': time.time()
        }
    
    def _route_query_to_agent(self, query: str, intent_info: Dict) -> Dict[str, str]:
        """Route query to the most appropriate agent"""
        
        intent = intent_info['primary_intent']
        confidence = intent_info['confidence']
        
        # Route based on intent
        if intent in ['disease_diagnosis', 'pest_control'] and confidence > 0.6:
            agent_name = 'plant_health'
        elif intent in ['fertilizer_advice'] and confidence > 0.6:
            agent_name = 'nutrition'
        elif intent in ['planting_guidance', 'harvesting_info', 'irrigation'] and confidence > 0.6:
            agent_name = 'crop_management'
        else:
            agent_name = 'coordinator'
        
        try:
            agent = self.agents[agent_name]
            response = agent.run(query)
            
            # Extract response content
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            return {
                'response': response_text,
                'agent': agent_name
            }
            
        except Exception as e:
            logger.error(f"Error with agent {agent_name}: {e}")
            # Fallback to coordinator
            if agent_name != 'coordinator':
                return self._route_query_to_agent(query, {'primary_intent': 'general_query', 'confidence': 1.0})
            else:
                return {
                    'response': "I apologize, but I'm experiencing technical difficulties. Please try asking your question again.",
                    'agent': 'fallback'
                }
    
    def _enhance_low_confidence_response(self, query: str, response: str, 
                                       quality_assessment: ResponseQuality) -> str:
        """Enhance responses with low confidence scores"""
        
        enhanced_response = response
        
        # Add uncertainty indicators for low confidence
        if quality_assessment.overall_confidence < 0.5:
            enhanced_response += "\n\n⚠️ Note: This recommendation should be verified with local agricultural experts or extension officers."
        
        # Add suggestions for improvement
        if quality_assessment.suggested_improvements:
            enhanced_response += f"\n\nFor more specific guidance, consider: {', '.join(quality_assessment.suggested_improvements[:2])}"
        
        return enhanced_response
    
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

# Context-Aware Enhancement
class ContextAwareEnhancer:
    """Add contextual information to responses"""
    
    def __init__(self):
        self.seasonal_context = {
            'kharif': {
                'months': [6, 7, 8, 9, 10],  # June to October
                'activities': ['sowing', 'vegetative growth', 'flowering', 'pod filling', 'harvesting']
            },
            'rabi': {
                'months': [11, 12, 1, 2, 3],  # November to March
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
        
        # Add seasonal context if relevant
        if any(activity in query.lower() for activity in ['plant', 'sow', 'harvest', 'fertilizer']):
            season_info = f"\n\n📅 Seasonal Context: Currently in {seasonal_context['season'].capitalize()} season. "
            if seasonal_context['season'] == 'kharif':
                season_info += "This is the main soybean growing season in India."
            else:
                season_info += "Focus on land preparation and planning for next kharif season."
            
            response += season_info
        
        return response

# A/B Testing Framework
class ABTestingFramework:
    """A/B testing for different response strategies"""
    
    def __init__(self):
        self.test_variants = {
            'detailed': 'detailed_response',
            'concise': 'concise_response',
            'structured': 'structured_response'
        }
        self.user_assignments = {}
        self.test_results = {variant: {'count': 0, 'satisfaction': []} for variant in self.test_variants.keys()}
    
    def assign_user_variant(self, user_id: str) -> str:
        """Assign user to a test variant"""
        import random
        if user_id not in self.user_assignments:
            variant = random.choice(list(self.test_variants.keys()))
            self.user_assignments[user_id] = variant
        
        return self.user_assignments[user_id]
    
    def format_response_by_variant(self, response: str, variant: str) -> str:
        """Format response based on test variant"""
        if variant == 'detailed':
            return f"📋 Detailed Analysis:\n\n{response}\n\n💡 Additional Tips: Consider consulting your local agricultural extension officer for region-specific advice."
        
        elif variant == 'concise':
            # Summarize to key points
            sentences = response.split('.')
            key_points = sentences[:3]  # Take first 3 sentences
            return '. '.join(key_points) + "."
        
        elif variant == 'structured':
            # Add structure with emoji bullets
            structured = response.replace('\n\n', '\n\n• ')
            return f"📋 Structured Guide:\n\n• {structured}"
        
        return response

# Enhanced Flask Application
app = Flask(__name__)
CORS(app)

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour", "10 per minute"]
)
limiter.init_app(app)

# Global instances
enhanced_soybot = None
ab_testing = ABTestingFramework()
context_enhancer = ContextAwareEnhancer()

def initialize_enhanced_soybot():
    """Initialize the enhanced SoyBot system"""
    global enhanced_soybot
    
    try:
        logger.info("Initializing Enhanced SoyBot System...")
        enhanced_soybot = MultiAgentSoyBot()
        enhanced_soybot.initialize()
        logger.info("Enhanced SoyBot System ready!")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Enhanced SoyBot: {e}")
        return False

# Enhanced HTML Template with Modern UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced SoyBot - AI-Powered Farming Assistant</title>
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

        /* Enhanced Header */
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

        /* Enhanced Status Bar */
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

        /* Enhanced Chat Area */
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
            transform: translateY(0);
        }

        .quick-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(76, 175, 80, 0.1), transparent);
            transition: left 0.6s ease;
        }

        .quick-btn:hover {
            border-color: var(--primary);
            transform: translateY(-4px);
            box-shadow: 0 8px 25px var(--shadow);
        }

        .quick-btn:hover::before {
            left: 100%;
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
            color: var(--on-surface);
        }

        .quick-btn-desc {
            font-size: 0.9rem;
            color: #666;
            line-height: 1.4;
        }

        /* Enhanced Chat Container */
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

        .quality-indicator {
            position: absolute;
            top: 0.5rem;
            right: 3rem;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--primary);
        }

        .quality-indicator.medium {
            background: var(--secondary);
        }

        .quality-indicator.low {
            background: #f44336;
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

        /* Enhanced Input Section */
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

        .action-btn::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 50%;
            transition: all 0.4s ease;
            transform: translate(-50%, -50%);
        }

        .action-btn:hover::before {
            width: 120%;
            height: 120%;
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

        /* Enhanced Animations */
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            33% { transform: translateY(-10px) rotate(1deg); }
            66% { transform: translateY(-5px) rotate(-1deg); }
        }

        @keyframes slideInMessage {
            from {
                opacity: 0;
                transform: translateY(30px) scale(0.95);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }

        @keyframes pulse-glow {
            0% { 
                box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7);
                transform: scale(1);
            }
            70% { 
                box-shadow: 0 0 0 10px rgba(76, 175, 80, 0);
                transform: scale(1.1);
            }
            100% { 
                box-shadow: 0 0 0 0 rgba(76, 175, 80, 0);
                transform: scale(1);
            }
        }

        @keyframes pulse-record {
            0% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(244, 67, 54, 0); }
            100% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0); }
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

            .feature-tags {
                flex-direction: column;
                align-items: center;
            }
        }

        /* Loading and States */
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

        @keyframes typing-bounce {
            0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
        }

        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid transparent;
            border-top: 2px solid currentColor;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Enhanced Header -->
        <header class="header">
            <div class="logo-container">
                <span class="logo">🌱</span>
            </div>
            <h1 class="title">Enhanced SoyBot</h1>
            <p class="subtitle">
                Multi-Agent AI Farming Assistant | बहु-एजेंट AI खेती सहायक
                <br>
                <small>Powered by Advanced Machine Learning & ICAR-IISR Guidelines</small>
            </p>
            <div class="feature-tags">
                <span class="feature-tag">🤖 Multi-Agent System</span>
                <span class="feature-tag">🌍 Multilingual Support</span>
                <span class="feature-tag">📊 Quality Assessment</span>
                <span class="feature-tag">🎯 Intent Classification</span>
                <span class="feature-tag">📋 Context Awareness</span>
            </div>
        </header>

        <!-- Enhanced Status Bar -->
        <div class="status-bar">
            <div class="status-info">
                <div class="agent-status">
                    <div class="status-dot"></div>
                    <span id="agent-status">Multi-Agent System Ready</span>
                </div>
            </div>
            
            <div class="metrics-display">
                <div>Queries Processed: <span id="query-count">0</span></div>
                <div>Avg Response Time: <span id="avg-time">0.0s</span></div>
                <div>Quality Score: <span id="quality-score">95%</span></div>
            </div>
            
            <div class="controls">
                <select id="language-select" style="padding: 0.5rem; border-radius: 0.5rem; border: 1px solid #ddd;">
                    <option value="en-US">🇺🇸 English</option>
                    <option value="hi-IN">🇮🇳 हिंदी</option>
                    <option value="mr-IN">🇮🇳 मराठी</option>
                </select>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span>Auto-speak</span>
                    <div class="toggle-switch active" id="auto-speak-toggle" style="position: relative; width: 44px; height: 24px; background: var(--primary); border-radius: 12px; cursor: pointer; transition: background 0.3s;">
                        <div class="toggle-slider" style="position: absolute; top: 2px; left: 2px; width: 20px; height: 20px; background: white; border-radius: 50%; transition: transform 0.3s; transform: translateX(18px);"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Chat Section -->
        <div class="chat-section">
            <!-- Enhanced Quick Actions -->
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
                            <div class="quick-btn-desc">Best sowing time guidance</div>
                        </div>
                    </button>
                    
                    <button class="quick-btn" onclick="askQuestion('My soybean plants have yellow spots on leaves. What could be the problem?')">
                        <div class="quick-btn-content">
                            <div class="quick-btn-icon"><i class="fas fa-leaf"></i></div>
                            <div class="quick-btn-title">Disease Diagnosis</div>
                            <div class="quick-btn-desc">Plant health expert analysis</div>
                        </div>
                    </button>
                    
                    <button class="quick-btn" onclick="askQuestion('सोयाबीन में कौन सा फर्टिलाइजर कब डालना चाहिए?')">
                        <div class="quick-btn-content">
                            <div class="quick-btn-icon"><i class="fas fa-flask"></i></div>
                            <div class="quick-btn-title">पोषण प्रबंधन</div>
                            <div class="quick-btn-desc">Nutrition expert recommendations</div>
                        </div>
                    </button>
                    
                    <button class="quick-btn" onclick="askQuestion('Which soybean varieties are best for drought conditions?')">
                        <div class="quick-btn-content">
                            <div class="quick-btn-icon"><i class="fas fa-seedling"></i></div>
                            <div class="quick-btn-title">Variety Selection</div>
                            <div class="quick-btn-desc">Crop management specialist advice</div>
                        </div>
                    </button>
                </div>
            </div>

            <!-- Enhanced Chat Container -->
            <div class="chat-container" id="chat-container">
                <div class="message bot">
                    <div class="message-header">
                        <i class="fas fa-robot"></i>
                        <span>Enhanced SoyBot</span>
                        <span>•</span>
                        <span id="welcome-time"></span>
                    </div>
                    <div class="message-content">
                        <div class="agent-badge">
                            <i class="fas fa-users"></i>
                            Multi-Agent
                        </div>
नमस्कार! Welcome to Enhanced SoyBot! 🚀

I'm your advanced multi-agent farming assistant with specialized experts:

🌾 <strong>Crop Management Specialist</strong> - Planting, irrigation, growth stages
🦠 <strong>Plant Health Specialist</strong> - Disease diagnosis & pest control  
🧪 <strong>Nutrition Expert</strong> - Soil health & fertilizer management
🎯 <strong>Smart Coordinator</strong> - Query routing & comprehensive advice

<strong>New Features:</strong>
✨ Intent classification for precise responses
✨ Response quality assessment  
✨ Context-aware recommendations
✨ Multi-language processing
✨ Performance monitoring

Ask me anything about soybean cultivation - I'll route your question to the right specialist!
                        <button class="speaker-btn" onclick="speakText(this.parentElement)">
                            <i class="fas fa-volume-up"></i>
                        </button>
                        <div class="quality-indicator" title="High Quality Response"></div>
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
                    <span>AI Specialists are analyzing...</span>
                </div>
            </div>
        </div>

        <!-- Enhanced Input Section -->
        <div class="input-section">
            <div class="input-container">
                <textarea 
                    id="query-input" 
                    class="input-field"
                    placeholder="Ask your farming question... Our AI specialists will analyze and provide expert guidance..."
                    rows="1"
                    aria-label="Enter your farming question"
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
        // Enhanced JavaScript with new features
        const chatContainer = document.getElementById('chat-container');
        const queryInput = document.getElementById('query-input');
        const askBtn = document.getElementById('ask-btn');
        const micBtn = document.getElementById('mic-btn');
        const typingIndicator = document.getElementById('typing-indicator');
        const autoSpeakToggle = document.getElementById('auto-speak-toggle');
        
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

        // Speech recognition setup
        function initSpeechRecognition() {
            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                recognition = new SpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = false;
                recognition.lang = 'hi-IN';

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
            agentStatusEl.textContent = 'Multi-Agent System Ready';
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

        // Enhanced text-to-speech
        function speakText(messageElement) {
            const text = messageElement.textContent.replace(/[🔊📢🎵]/g, '').trim();
            const speakerBtn = messageElement.querySelector('.speaker-btn i');
            
            if (synthesis.speaking) {
                synthesis.cancel();
                if (speakerBtn) speakerBtn.className = 'fas fa-volume-up';
                return;
            }

            if (text) {
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = 'hi-IN';
                utterance.rate = 0.9;
                utterance.pitch = 1.0;

                if (speakerBtn) speakerBtn.className = 'fas fa-volume-mute';

                utterance.onend = () => {
                    if (speakerBtn) speakerBtn.className = 'fas fa-volume-up';
                };

                synthesis.speak(utterance);
            }
        }

        // Enhanced message display
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
            
            if (sender === 'bot' && metadata.quality_assessment) {
                const confidence = metadata.quality_assessment.overall_confidence;
                const qualityClass = confidence >= 0.8 ? '' : confidence >= 0.6 ? 'medium' : 'low';
                const qualityTitle = `Quality Score: ${(confidence * 100).toFixed(1)}%`;
                qualityIndicator = `<div class="quality-indicator ${qualityClass}" title="${qualityTitle}"></div>`;
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

            // Auto-speak bot messages
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

        // Enhanced send question
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
            agentStatusEl.textContent = 'AI Specialists analyzing...';

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
                addMessage('Sorry, something went wrong. Please check your connection and try again.', 'bot');
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

# Enhanced API Routes
@app.route('/')
def index():
    """Serve enhanced web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/enhanced-status', methods=['GET'])
@limiter.limit("30 per minute")
def get_enhanced_status():
    """Get enhanced system status with metrics"""
    global enhanced_soybot
    
    system_ready = enhanced_soybot is not None and enhanced_soybot.is_initialized
    
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
            'performance_monitoring': True
        }
    }
    
    if system_ready:
        status_data['performance_metrics'] = enhanced_soybot.get_performance_metrics()
    
    return jsonify(status_data)

@app.route('/api/enhanced-ask', methods=['POST'])
@limiter.limit("20 per minute")
def enhanced_ask():
    """Enhanced question processing with multi-agent system"""
    global enhanced_soybot
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
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
        if not enhanced_soybot or not enhanced_soybot.is_initialized:
            return jsonify({
                'success': False,
                'error': 'Enhanced SoyBot system is still initializing. Please wait a moment.'
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
        
        # Get user context (you can enhance this with session data)
        user_context = data.get('context', {})
        
        logger.info(f"Processing enhanced query: {question}")
        
        # Process with multi-agent system
        result = enhanced_soybot.process_query(question, user_context)
        
        # Enhance with context if needed
        enhanced_response = context_enhancer.enhance_with_context(
            result['response'], question
        )
        
        # A/B testing (you can implement user session tracking)
        user_id = request.remote_addr  # Simple user identification
        variant = ab_testing.assign_user_variant(user_id)
        final_response = ab_testing.format_response_by_variant(enhanced_response, variant)
        
        logger.info(f"Response generated by agent: {result.get('agent_used', 'unknown')}")
        
        response_data = {
            'success': True,
            'response': final_response,
            'agent_used': result.get('agent_used'),
            'quality_assessment': convert_numpy_types(result.get('quality_assessment', {})),
            'language_info': convert_numpy_types(result.get('language_info', {})),
            'intent_info': convert_numpy_types(result.get('intent_info', {})),
            'variant': variant,
            'processing_time': result.get('processing_time')
        }

        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in enhanced query processing: {str(e)}")
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
    global enhanced_soybot
    
    if not enhanced_soybot or not enhanced_soybot.is_initialized:
        return jsonify({'error': 'System not initialized'}), 503
    
    metrics = enhanced_soybot.get_performance_metrics()
    
    # Add A/B testing results
    metrics['ab_testing'] = {
        'variants': list(ab_testing.test_variants.keys()),
        'results': ab_testing.test_results
    }
    
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
import os
os.makedirs('models', exist_ok=True)
os.makedirs('vectordb', exist_ok=True)
os.makedirs('logs', exist_ok=True)

if __name__ == '__main__':
    logger.info("Starting Enhanced SoyBot System...")
    
    # Initialize enhanced system
    if initialize_enhanced_soybot():
        logger.info("✅ Enhanced SoyBot System initialized successfully!")
        logger.info("Features enabled:")
        logger.info("   🤖 Multi-Agent Architecture")
        logger.info("   🎯 Intent Classification") 
        logger.info("   📊 Quality Assessment")
        logger.info("   🌍 Advanced Language Processing")
        logger.info("   📋 Context Awareness")
        logger.info("   ⚡ Performance Monitoring")
        logger.info("   🧪 A/B Testing Framework")
        logger.info("🚀 Starting enhanced Flask server...")
        logger.info("🌐 Access at: http://localhost:5000")
        logger.info("-" * 80)
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
    else:
        logger.error("❌ Failed to initialize Enhanced SoyBot System")
        logger.error("Please check your configuration and try again.")
        sys.exit(1)