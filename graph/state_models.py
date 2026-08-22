from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class PINNDesignState(BaseModel):
    project_status: Optional[str] = None
    physics_problem: Optional[str] = None
    validation_data: Optional[Any] = None
    designer_input: Optional[str] = None
    designer_output: Optional[str] = None
    pinn_architecture: Optional[Dict[str, Any]] = None
    pinn_loss_categories: Optional[Dict[str, str]] = None
    initial_losses: Optional[str] = None
    pinn_training_logs: Optional[str] = None
    pinn_weights: Optional[str] = None
    pinn_relative_error: Optional[float] = None
    models_history: Optional[List[Dict[str, Any]]] = []
    chat_history: Optional[List[Dict[str, str]]] = []
