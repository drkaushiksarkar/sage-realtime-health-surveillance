"""Schema validation for sage-realtime-health-surveillance data inputs."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import re


class DataFormat(Enum):
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    ARROW = "arrow"


@dataclass
class ColumnSchema:
    name: str
    dtype: str
    nullable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[str]] = None

    def validate(self, value: Any) -> bool:
        if value is None:
            return self.nullable
        if self.min_value is not None and float(value) < self.min_value:
            return False
        if self.max_value is not None and float(value) > self.max_value:
            return False
        if self.pattern and not re.match(self.pattern, str(value)):
            return False
        if self.allowed_values and str(value) not in self.allowed_values:
            return False
        return True


@dataclass
class SageRealtimeHealthSurveillanceSchema:
    columns: List[ColumnSchema] = field(default_factory=list)
    required_columns: List[str] = field(default_factory=list)
    format: DataFormat = DataFormat.JSON
    version: str = "1.0.0"

    def validate_record(self, record: Dict[str, Any]) -> List[str]:
        errors = []
        for req in self.required_columns:
            if req not in record:
                errors.append(f"Missing required column: {req}")
        for col in self.columns:
            if col.name in record and not col.validate(record[col.name]):
                errors.append(f"Validation failed for {col.name}: {record[col.name]}")
        return errors

    def validate_batch(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_errors = 0
        invalid_records = []
        for i, record in enumerate(records):
            errs = self.validate_record(record)
            if errs:
                total_errors += len(errs)
                invalid_records.append({"index": i, "errors": errs})
        return {
            "total_records": len(records),
            "valid_records": len(records) - len(invalid_records),
            "invalid_records": len(invalid_records),
            "total_errors": total_errors,
            "details": invalid_records[:10],
        }
