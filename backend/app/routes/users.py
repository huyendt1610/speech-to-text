from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File 
from app.schemas.user import UserResponse, UserUpdate, PasswordUpdate
from app.models.user import User
from app.middleware.auth import get_current_user 
from sqlalchemy.orm import Session 
from app.db.database import get_db
from app.services.user_service import update_user_profile, update_user_password

router = APIRouter() 

@router.get("/profile", response_model=UserResponse)
async def get_profile(current_user: User = Depends(get_current_user)):
    return UserResponse.model_validate(current_user)

@router.put("/profile", response_model=UserResponse)
async def update_profile(
    update_data: UserUpdate, 
    profile_picture: UploadFile = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if profile_picture: 
        if profile_picture.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Profile picture must be JPG or PNG"
            )
        # check file size (5MB max)
        file_size = 0 
        await profile_picture.seek(0) # reset file pointer 
        content = await profile_picture.read() 
        file_size = len(content)
        await profile_picture.seek(0) # reset file pointer to continue using 
        if file_size > 5242880:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Profile picture must be less than 5MB"
            )

    updated_user = update_user_profile(db, current_user, update_data, profile_picture)
    return UserResponse.model_validate(updated_user)


@router.put("/password")
async def update_password(
    password_data: PasswordUpdate, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
): 
    update_user_password(db, current_user, password_data.current_password, password_data.new_password)
    return {"message": "Password updated successfully!"}