from typing import Literal

from pydantic import BaseModel, validator


class AudioResponse(BaseModel):
    age: int
    gender: Literal["Female", "Male", "Child"]

    @validator('age')
    def age_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Age must be positive')
        return v

    @validator('gender')
    def gender_must_be_valid(cls, v):
        if v not in ["Female", "Male", "Child"]:
            raise ValueError('Gender must be either Female, Male, or Child')
        return v