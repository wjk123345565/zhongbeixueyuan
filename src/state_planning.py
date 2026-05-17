from typing import List, TypedDict, Optional
from langchain_core.messages import BaseMessage
import operator

def manage_messages(old_messages: List[BaseMessage], new_messages: BaseMessage):
    combined = old_messages + [new_messages]
    return combined[-20:]

class PlanningState(TypedDict):
    messages: List[BaseMessage]
    current_plan: Optional[str]
    academic_votes: List[str]
    emotional_votes: List[str]
    consensus_reached: bool
    iteration_count: int
    final_plan: Optional[str]
    needs_revision: bool
    revision_feedback: Optional[str]