from flask import jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from datetime import timedelta

def generate_token(user_id):
    """
    Generates a JWT for the user with the given user ID.
    
    :param user_id: ID of the user.
    :return: JWT token.
    """
    access_token = create_access_token(identity=user_id, expires_delta=timedelta(hours=1))
    return access_token

@jwt_required()
def get_current_user():
    """
    Retrieves the current user ID from the JWT token.
    
    :return: User ID.
    """
    return get_jwt_identity()
