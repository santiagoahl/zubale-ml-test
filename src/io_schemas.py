# TODO: Pydantic schemas for /predict
from pydantic import BaseModel, Field
from typing import Dict, Optional, Literal, Union

class PredictModel(BaseModel):
    plan_type: Literal["Basic","Standard","Pro"]
    contract_type: Literal["Monthly","Annual"]
    autopay: Literal["Yes","No"]
    is_promo_user: Literal["Yes","No"]
    add_on_count: Union[int, float]
    tenure_months: Union[int, float]
    monthly_usage_gb: Union[int, float]
    avg_latency_ms: Union[int, float]
    support_tickets_30d: Union[int, float]
    discount_pct: Union[int, float]
    payment_failures_90d: Union[int, float]
    downtime_hours_30d: Union[int, float]

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