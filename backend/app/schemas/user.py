from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional
from datetime import datetime

class UserRegisterBase(BaseModel): 
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str 
    mobile: str 
    account_type: str = Field(..., pattern="^(normal|vip)$")

class UserRegisterVip(UserRegisterBase): 
    account_type: str = "vip"

class UserRegisterNormal(UserRegisterBase): 
    account_type: str = "normal"

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str: 
        if len(v) < 8: 
            raise ValueError("Password must be at least 8 characters")
        if not any(c.upper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one number")
        return v
    
class UserLogin(BaseModel): 
    email: EmailStr
    password: str 
    remember_me: bool = False

class UserResponse(BaseModel):
    id: str 
    email: str 
    full_name: str 
    mobile: str 
    account_type: str 
    profile_picture: str | None = None
    created_at: datetime 

    class Config: 
        from_attributes = True # using when converting SQLAlchemy (user.email) object sang Pydantic (user["email"]) to return response 

class TokenRepsonse(BaseModel):
    access_token: str 
    token_type: str = "bearer"
    user: UserResponse

class UserUpdate(BaseModel): 
    full_name: str | None = None
    mobile: str | None = None

class PasswordUpdate(BaseModel): 
    current_password: str 
    new_password: str = Field(..., min_length=8)

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str: 
        if len(v) < 8: 
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one upper letter")
        if not any(c.isdigit() for c in v): 
            raise ValueError("Password must containt at least one number")
        
        return v