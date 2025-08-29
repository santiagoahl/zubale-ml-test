from pydantic import BaseModel, Field
from typing import Dict, Optional, Literal, Union, List

class PredictModel(BaseModel):
    plan_type: Optional[Literal["Basic","Standard","Pro"]]
    contract_type: Optional[Literal["Monthly","Annual"]]
    autopay: Optional[Literal["Yes","No"]]
    is_promo_user: Optional[Literal["Yes","No"]]
    add_on_count: Optional[Union[int, float]]
    tenure_months: Optional[Union[int, float]]
    monthly_usage_gb: Optional[Union[int, float]]
    avg_latency_ms: Optional[Union[int, float]]
    support_tickets_30d: Optional[Union[int, float]]
    discount_pct: Optional[Union[int, float]]
    payment_failures_90d: Optional[Union[int, float]]
    downtime_hours_30d: Optional[Union[int, float]] = 0.0

    class Config:
        json_schema_extra = {
            "example": {
            "plan_type": "Pro",
            "contract_type": "Annual",
            "autopay": "Yes",
            "is_promo_user": "No",
            "add_on_count": 3,
            "tenure_months": 20,
            "monthly_usage_gb": 119.64,
            "avg_latency_ms": 116.7,
            "support_tickets_30d": 0,
            "discount_pct": 10.6,
            "payment_failures_90d": 2,
            "downtime_hours_30d": 2.12
            }
        } 


class ActionPlanModel(BaseModel):
    status: Literal["healthy", "warn", "critical"] = Field(..., description="Overall status of the monitored system")
    findings: List[str] = Field(default_factory=list, description="List of detected issues or observations")
    actions: List[Literal[
        "open_incident",
        "trigger_retraining",
        "roll_back_model",
        "raise_thresholds",
        "do_nothing"
    ]] = Field(default_factory=list, description="Actions to take based on findings")
    page_oncall: Optional[bool] = Field(default=False, description="Whether to page the on-call engineer")
