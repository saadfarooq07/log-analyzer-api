"""Interactive mode handler for user input and Q&A flow.

This module provides interactive capabilities for the log analyzer,
allowing users to provide additional context and answer questions.
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .unified_state import UnifiedState

logger = logging.getLogger(__name__)


class Question:
    """Represents a question to ask the user."""
    
    def __init__(
        self,
        id: str,
        text: str,
        question_type: str = "open",
        options: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        priority: str = "medium",
        required: bool = True
    ):
        """Initialize a question.
        
        Args:
            id: Unique question ID
            text: Question text
            question_type: Type of question (open, choice, confirm)
            options: Options for choice questions
            context: Additional context about the question
            priority: Question priority (high, medium, low)
            required: Whether the question must be answered
        """
        self.id = id
        self.text = text
        self.question_type = question_type
        self.options = options or []
        self.context = context or {}
        self.priority = priority
        self.required = required
        self.created_at = datetime.now()
        self.answered = False
        self.answer = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert question to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "type": self.question_type,
            "options": self.options,
            "context": self.context,
            "priority": self.priority,
            "required": self.required,
            "created_at": self.created_at.isoformat(),
            "answered": self.answered,
            "answer": self.answer
        }


class InteractiveHandler:
    """Handles interactive user input and Q&A flow."""
    
    def __init__(self):
        self.pending_questions: Dict[str, Question] = {}
        self.answered_questions: Dict[str, Question] = {}
        self.interaction_callbacks: Dict[str, Callable] = {}
        self.default_timeout = 300  # 5 minutes
    
    def create_question(
        self,
        text: str,
        question_type: str = "open",
        options: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        priority: str = "medium",
        required: bool = True
    ) -> Question:
        """Create a new question."""
        question_id = str(uuid.uuid4())
        question = Question(
            id=question_id,
            text=text,
            question_type=question_type,
            options=options,
            context=context,
            priority=priority,
            required=required
        )
        
        self.pending_questions[question_id] = question
        logger.info(f"Created question {question_id}: {text[:50]}...")
        
        return question
    
    def add_questions_to_state(self, state: UnifiedState, questions: List[Question]) -> None:
        """Add questions to the state."""
        state.user_interaction_required = True
        
        for question in questions:
            state.pending_questions.append(question.to_dict())
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        state.pending_questions.sort(
            key=lambda q: priority_order.get(q.get("priority", "medium"), 1)
        )
    
    def process_user_response(
        self,
        state: UnifiedState,
        question_id: str,
        answer: Any
    ) -> Dict[str, Any]:
        """Process a user's response to a question."""
        # Find the question
        question_dict = None
        for i, q in enumerate(state.pending_questions):
            if q["id"] == question_id:
                question_dict = q
                state.pending_questions.pop(i)
                break
        
        if not question_dict:
            logger.warning(f"Question {question_id} not found in pending questions")
            return {"status": "error", "message": "Question not found"}
        
        # Validate answer based on question type
        if question_dict["type"] == "choice" and answer not in question_dict.get("options", []):
            return {
                "status": "error",
                "message": f"Invalid choice. Options are: {', '.join(question_dict['options'])}"
            }
        
        if question_dict["type"] == "confirm" and answer not in ["yes", "no", True, False]:
            return {
                "status": "error",
                "message": "Please answer yes or no"
            }
        
        # Store the answer
        state.user_responses[question_id] = {
            "question": question_dict["text"],
            "answer": answer,
            "answered_at": datetime.now().isoformat()
        }
        
        # Add to interaction history
        state.interaction_history.append({
            "type": "question_answered",
            "question_id": question_id,
            "question": question_dict["text"],
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
        
        # Check if all required questions are answered
        required_pending = [q for q in state.pending_questions if q.get("required", True)]
        if not required_pending:
            state.user_interaction_required = False
        
        logger.info(f"Processed answer for question {question_id}")
        
        return {
            "status": "success",
            "remaining_questions": len(state.pending_questions),
            "interaction_required": state.user_interaction_required
        }
    
    def generate_clarification_questions(
        self,
        state: UnifiedState,
        analysis_result: Dict[str, Any]
    ) -> List[Question]:
        """Generate clarification questions based on analysis."""
        questions = []
        
        # Check for ambiguous issues
        if analysis_result.get("issues"):
            for issue in analysis_result["issues"]:
                if issue.get("severity") == "critical" and "unclear" in issue.get("description", "").lower():
                    questions.append(self.create_question(
                        text=f"Can you provide more context about: {issue['description']}?",
                        question_type="open",
                        context={"issue_id": issue.get("id"), "issue_type": issue.get("type")},
                        priority="high"
                    ))
        
        # Check for missing environment details
        if not state.log_metadata.get("environment_details"):
            questions.append(self.create_question(
                text="What operating system and version is this running on?",
                question_type="open",
                context={"metadata_field": "os_version"},
                priority="medium"
            ))
            
            questions.append(self.create_question(
                text="What is the application/service name?",
                question_type="open",
                context={"metadata_field": "application_name"},
                priority="medium"
            ))
        
        # Check for time-related issues
        if any("timeout" in str(issue).lower() for issue in analysis_result.get("issues", [])):
            questions.append(self.create_question(
                text="Have you recently made any configuration changes?",
                question_type="confirm",
                context={"related_to": "timeout_issues"},
                priority="medium"
            ))
        
        # Check for security issues
        if any(issue.get("type", "").startswith("security") for issue in analysis_result.get("issues", [])):
            questions.append(self.create_question(
                text="Is this system exposed to the internet?",
                question_type="confirm",
                context={"related_to": "security_assessment"},
                priority="high"
            ))
        
        return questions
    
    def create_interactive_prompt(self, state: UnifiedState) -> str:
        """Create a prompt for interactive mode."""
        if not state.pending_questions:
            return "No pending questions."
        
        # Get highest priority question
        question = state.pending_questions[0]
        
        prompt_parts = [
            f"Question: {question['text']}",
            f"Type: {question['type']}"
        ]
        
        if question['type'] == "choice" and question.get('options'):
            prompt_parts.append(f"Options: {', '.join(question['options'])}")
        elif question['type'] == "confirm":
            prompt_parts.append("Please answer: yes/no")
        
        if question.get('context'):
            prompt_parts.append(f"Context: {question['context']}")
        
        return "\n".join(prompt_parts)
    
    async def handle_interactive_session(
        self,
        state: UnifiedState,
        input_callback: Callable[[str], Union[str, Any]],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Handle an interactive session with the user.
        
        Args:
            state: Current state
            input_callback: Callback to get user input
            timeout: Session timeout in seconds
            
        Returns:
            Session results
        """
        timeout = timeout or self.default_timeout
        start_time = datetime.now()
        responses = {}
        
        try:
            while state.pending_questions and state.user_interaction_required:
                # Check timeout
                if (datetime.now() - start_time).total_seconds() > timeout:
                    logger.warning("Interactive session timed out")
                    break
                
                # Get next question
                question_dict = state.pending_questions[0]
                
                # Create prompt
                prompt = self.create_interactive_prompt(state)
                
                # Get user input
                try:
                    if asyncio.iscoroutinefunction(input_callback):
                        answer = await input_callback(prompt)
                    else:
                        answer = input_callback(prompt)
                except Exception as e:
                    logger.error(f"Error getting user input: {str(e)}")
                    break
                
                # Process response
                result = self.process_user_response(state, question_dict["id"], answer)
                
                if result["status"] == "success":
                    responses[question_dict["id"]] = answer
                else:
                    # Retry the question
                    logger.warning(f"Invalid response: {result['message']}")
                    continue
            
            # Summary
            return {
                "status": "completed" if not state.pending_questions else "partial",
                "responses": responses,
                "answered_count": len(responses),
                "remaining_count": len(state.pending_questions),
                "duration_seconds": (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Interactive session error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "responses": responses
            }
    
    def get_interaction_summary(self, state: UnifiedState) -> Dict[str, Any]:
        """Get a summary of the interaction state."""
        return {
            "interaction_required": state.user_interaction_required,
            "pending_questions": len(state.pending_questions),
            "answered_questions": len(state.user_responses),
            "interaction_history": len(state.interaction_history),
            "pending_details": [
                {
                    "id": q["id"],
                    "text": q["text"][:100] + "..." if len(q["text"]) > 100 else q["text"],
                    "type": q["type"],
                    "priority": q["priority"]
                }
                for q in state.pending_questions[:5]  # First 5 questions
            ]
        }


# Global interactive handler
_interactive_handler = InteractiveHandler()


def get_interactive_handler() -> InteractiveHandler:
    """Get the global interactive handler."""
    return _interactive_handler