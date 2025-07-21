"""Advanced cycle detection for preventing infinite loops in the workflow.

This module implements sophisticated cycle detection algorithms to identify
and prevent various types of cycles in the analysis workflow.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from enum import Enum
from collections import deque
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class CycleType(Enum):
    """Types of cycles that can be detected."""
    SIMPLE_LOOP = "simple_loop"          # A->B->A
    COMPLEX_LOOP = "complex_loop"        # A->B->C->A
    OSCILLATION = "oscillation"          # A->B->A->B (back and forth)
    DEADLOCK = "deadlock"                # Stuck in same state
    SPIRAL = "spiral"                    # Similar states with minor variations


class CyclePattern:
    """Represents a detected cycle pattern."""
    
    def __init__(self, cycle_type: CycleType, pattern: List[str], 
                 states: List[Dict[str, Any]], confidence: float = 1.0):
        self.cycle_type = cycle_type
        self.pattern = pattern  # List of node names
        self.states = states    # List of state snapshots
        self.confidence = confidence
        self.occurrence_count = 1
        
    def __repr__(self):
        return f"CyclePattern({self.cycle_type.value}, {' -> '.join(self.pattern)})"


class CycleDetector:
    """Advanced cycle detection with pattern recognition."""
    
    def __init__(self, max_history: int = 20, detection_threshold: int = 3):
        """Initialize the cycle detector.
        
        Args:
            max_history: Maximum number of states to keep in history
            detection_threshold: Number of repetitions before declaring a cycle
        """
        self.max_history = max_history
        self.detection_threshold = detection_threshold
        
        # State tracking
        self.state_history: deque = deque(maxlen=max_history)
        self.transition_history: deque = deque(maxlen=max_history)
        self.state_hashes: deque = deque(maxlen=max_history)
        
        # Pattern tracking
        self.detected_patterns: List[CyclePattern] = []
        self.pattern_counts: Dict[str, int] = {}
        
        # Deadlock detection
        self.same_state_count = 0
        self.last_state_hash = None
        
    def _hash_state(self, state: Dict[str, Any]) -> str:
        """Create a hash of the relevant state parts."""
        # Only hash relevant fields that indicate actual state changes
        relevant_fields = {
            "validation_status": state.get("validation_status"),
            "analysis_result": bool(state.get("analysis_result")),
            "error": state.get("error"),
            "pending_questions": len(state.get("pending_questions", [])),
            "tool_calls_count": len(state.get("tool_calls", [])),
            "messages_count": len(state.get("messages", []))
        }
        
        # Create deterministic string representation
        state_str = json.dumps(relevant_fields, sort_keys=True)
        return hashlib.md5(state_str.encode()).hexdigest()[:8]
    
    def add_transition(self, from_node: str, to_node: str, 
                      state: Dict[str, Any]) -> Optional[CyclePattern]:
        """Add a state transition and check for cycles.
        
        Args:
            from_node: Source node name
            to_node: Destination node name
            state: Current state snapshot
            
        Returns:
            Detected cycle pattern if found, None otherwise
        """
        # Create transition key
        transition = f"{from_node}->{to_node}"
        state_hash = self._hash_state(state)
        
        # Add to history
        self.transition_history.append(transition)
        self.state_history.append(state.copy())
        self.state_hashes.append(state_hash)
        
        # Check for various cycle types
        cycle = None
        
        # 1. Check for deadlock (same state)
        if state_hash == self.last_state_hash:
            self.same_state_count += 1
            if self.same_state_count >= self.detection_threshold:
                cycle = CyclePattern(
                    CycleType.DEADLOCK,
                    [from_node],
                    [state],
                    confidence=1.0
                )
                logger.warning(f"Deadlock detected at {from_node}")
        else:
            self.same_state_count = 0
            self.last_state_hash = state_hash
        
        # 2. Check for simple loops
        if not cycle:
            cycle = self._detect_simple_loop()
        
        # 3. Check for oscillations
        if not cycle:
            cycle = self._detect_oscillation()
        
        # 4. Check for complex loops
        if not cycle:
            cycle = self._detect_complex_loop()
        
        # 5. Check for spiral patterns
        if not cycle:
            cycle = self._detect_spiral()
        
        # Record detected cycle
        if cycle:
            self.detected_patterns.append(cycle)
            pattern_key = "->".join(cycle.pattern)
            self.pattern_counts[pattern_key] = self.pattern_counts.get(pattern_key, 0) + 1
            
        return cycle
    
    def _detect_simple_loop(self) -> Optional[CyclePattern]:
        """Detect simple A->B->A style loops."""
        if len(self.transition_history) < self.detection_threshold:
            return None
        
        # Check last N transitions for repetition
        recent = list(self.transition_history)[-self.detection_threshold:]
        
        # Look for repeated patterns of length 2-3
        for pattern_len in range(2, 4):
            if len(recent) >= pattern_len * 2:
                pattern = recent[-pattern_len:]
                prev_pattern = recent[-pattern_len*2:-pattern_len]
                
                if pattern == prev_pattern:
                    # Extract node names from transitions
                    nodes = []
                    for trans in pattern:
                        from_node = trans.split("->")[0]
                        if from_node not in nodes:
                            nodes.append(from_node)
                    
                    return CyclePattern(
                        CycleType.SIMPLE_LOOP,
                        nodes,
                        list(self.state_history)[-pattern_len:],
                        confidence=0.9
                    )
        
        return None
    
    def _detect_oscillation(self) -> Optional[CyclePattern]:
        """Detect A->B->A->B oscillation patterns."""
        if len(self.transition_history) < 4:
            return None
        
        recent = list(self.transition_history)[-6:]
        
        # Check for ABABAB pattern
        if len(recent) >= 4:
            if (recent[-1] == recent[-3] and 
                recent[-2] == recent[-4] and
                recent[-1] != recent[-2]):
                
                # Extract the two nodes
                trans1 = recent[-1].split("->")
                trans2 = recent[-2].split("->")
                nodes = [trans1[0], trans2[0]]
                
                return CyclePattern(
                    CycleType.OSCILLATION,
                    nodes,
                    list(self.state_history)[-2:],
                    confidence=0.85
                )
        
        return None
    
    def _detect_complex_loop(self) -> Optional[CyclePattern]:
        """Detect complex loops with multiple nodes."""
        if len(self.state_hashes) < 4:
            return None
        
        # Look for repeated state hash sequences
        for i in range(len(self.state_hashes) - 3):
            for j in range(i + 2, len(self.state_hashes) - 1):
                if self.state_hashes[i] == self.state_hashes[j]:
                    # Found matching states, check if it's a cycle
                    cycle_length = j - i
                    if cycle_length <= 6:  # Reasonable cycle length
                        # Extract the cycle
                        cycle_transitions = list(self.transition_history)[i:j]
                        nodes = []
                        for trans in cycle_transitions:
                            node = trans.split("->")[0]
                            if node not in nodes:
                                nodes.append(node)
                        
                        # Verify it's actually repeating
                        if j + cycle_length <= len(self.state_hashes):
                            next_seq = list(self.state_hashes)[j:j+cycle_length]
                            orig_seq = list(self.state_hashes)[i:i+cycle_length]
                            
                            if next_seq == orig_seq:
                                return CyclePattern(
                                    CycleType.COMPLEX_LOOP,
                                    nodes,
                                    list(self.state_history)[i:j],
                                    confidence=0.8
                                )
        
        return None
    
    def _detect_spiral(self) -> Optional[CyclePattern]:
        """Detect spiral patterns where states are similar but slightly different."""
        if len(self.state_history) < 4:
            return None
        
        # Compare recent states for similarity
        recent_states = list(self.state_history)[-4:]
        
        # Check if states are progressively getting more complex
        message_counts = [len(s.get("messages", [])) for s in recent_states]
        tool_counts = [len(s.get("tool_calls", [])) for s in recent_states]
        
        # Spiral detection: increasing complexity with same validation status
        if (all(s.get("validation_status") == "invalid" for s in recent_states) and
            message_counts == sorted(message_counts) and
            tool_counts == sorted(tool_counts)):
            
            # Extract nodes from recent transitions
            nodes = []
            for trans in list(self.transition_history)[-3:]:
                node = trans.split("->")[0]
                if node not in nodes:
                    nodes.append(node)
            
            return CyclePattern(
                CycleType.SPIRAL,
                nodes,
                recent_states,
                confidence=0.7
            )
        
        return None
    
    def get_cycle_summary(self) -> Dict[str, Any]:
        """Get a summary of detected cycles."""
        return {
            "total_cycles_detected": len(self.detected_patterns),
            "cycle_types": {
                cycle_type.value: sum(1 for p in self.detected_patterns 
                                    if p.cycle_type == cycle_type)
                for cycle_type in CycleType
            },
            "most_common_patterns": sorted(
                self.pattern_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "recent_patterns": [str(p) for p in self.detected_patterns[-5:]]
        }
    
    def reset(self):
        """Reset the cycle detector state."""
        self.state_history.clear()
        self.transition_history.clear()
        self.state_hashes.clear()
        self.detected_patterns.clear()
        self.pattern_counts.clear()
        self.same_state_count = 0
        self.last_state_hash = None
    
    def should_break_cycle(self, pattern: CyclePattern) -> bool:
        """Determine if a cycle should be broken based on its characteristics."""
        # Always break deadlocks
        if pattern.cycle_type == CycleType.DEADLOCK:
            return True
        
        # Break oscillations after threshold
        if pattern.cycle_type == CycleType.OSCILLATION:
            pattern_key = "->".join(pattern.pattern)
            return self.pattern_counts.get(pattern_key, 0) >= self.detection_threshold
        
        # Break other cycles based on occurrence count
        pattern_key = "->".join(pattern.pattern)
        return self.pattern_counts.get(pattern_key, 0) >= self.detection_threshold + 1