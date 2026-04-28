from fastapi import APIRouter, Depends, HTTPException, status 
from sqlalchemy.orm import Session
from app.db.database import get_db 
from app.schemas.user import TokenRepsonse, UserResponse, UserRegisterNormal, UserRegisterVip, UserLogin
from app.services.auth_service import register_user, authenticate_user, create_token_for_user
from app.models.user import User 
from app.middleware.auth import get_current_user

router = APIRouter() 

@router.post("/register", response_model=TokenRepsonse)
async def register(
    user_data: UserRegisterNormal | UserRegisterVip,
    db: Session = Depends(get_db)
):
    user = await register_user(db, user_data)   
    token = create_token_for_user(user)
    return TokenRepsonse(
        access_token=token, 
        user=UserResponse.model_validate(user)
    )

@router.post("/login", response_model=TokenRepsonse)
async def login(
    credentials: UserLogin, 
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, credentials.email, credentials.password)
    if not user: 
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    token = create_token_for_user(user, remember_me=credentials.remember_me)
    return TokenRepsonse(
        access_token=token, 
        user=UserResponse.model_validate(user)
    )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    return UserResponse.model_validate(current_user)