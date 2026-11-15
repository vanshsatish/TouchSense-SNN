from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class TouchPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float
    y: float
    t_ms: float = Field(ge=0)
    pressure: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)

class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    points: list[TouchPoint] = Field(min_length=2, max_length=5000)
    tap_interval_ms: Optional[float] = Field(default=None, ge=0)
    screen_width: Optional[int] = Field(default=None, gt=0)
    screen_height: Optional[int] = Field(default=None, gt=0)
    device_label: Optional[str] = Field(default=None, max_length=80)

class AnalyzeResponse(BaseModel):
    features: dict
    model_status: str
    prediction: Optional[dict] = None
    notice: str
