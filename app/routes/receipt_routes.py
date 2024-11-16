from flask import Blueprint, request, jsonify, session
from app.models.receipt import Receipt
from app.utils.ocr_processing import extract_text
from app.database import db
import os
from werkzeug.utils import secure_filename

receipt_bp = Blueprint('receipt', __name__)

@receipt_bp.route('/upload', methods=['POST'])
def upload_receipt():
    if 'user_id' not in session:
        return jsonify(message="Unauthorized"), 401

    file = request.files['file']
    if not file:
        return jsonify(message="No file provided"), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join('static/uploads', filename)
    file.save(file_path)

    extracted_text = extract_text(file_path)
    receipt = Receipt(user_id=session['user_id'], image_path=file_path, extracted_text=extracted_text)
    db.session.add(receipt)
    db.session.commit()
    return jsonify(message="Receipt uploaded and processed"), 201

@receipt_bp.route('/history', methods=['GET'])
def view_history():
    if 'user_id' not in session:
        return jsonify(message="Unauthorized"), 401

    receipts = Receipt.query.filter_by(user_id=session['user_id']).all()
    history = [{'id': r.id, 'image_path': r.image_path, 'extracted_text': r.extracted_text, 'uploaded_at': r.uploaded_at} for r in receipts]
    return jsonify(history=history), 200

@receipt_bp.route('/compare', methods=['GET'])
def compare_expenses():
    # Logic for expense comparison over months (dummy implementation for now)
    return jsonify(message="Comparison feature not implemented"), 200
