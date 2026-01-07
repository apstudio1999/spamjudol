"""
INTEGRATION GUIDE - Menggunakan Spam Detector di Aplikasi Lain
==============================================================

Contoh kode untuk mengintegrasikan spam detector ke aplikasi Python Anda.
"""

# ============================================================================
# CONTOH 1: Import dan Gunakan di Python Script
# ============================================================================

from inference_local import SpamDetector

# Initialize detector (load model sekali)
detector = SpamDetector(
    model_path="gnn_spam_model.pt",
    vectorizer_path="tfidf_vectorizer.pkl"
)

# Predict single text
result = detector.predict("subscribe channel kami gratis")
print(f"Label: {result[0]['label']}")
print(f"Confidence: {result[0]['confidence']}")
print(f"Spam Score: {result[0]['spam_score']}")

# Predict batch
texts = [
    "this is a great video",
    "subscribe now!!!",
    "check my channel"
]
results = detector.predict(texts)
for r in results:
    print(f"{r['text'][:30]:.<30} {r['label']:.<10} {r['confidence']:.2%}")

# ============================================================================
# CONTOH 2: Flask Web API
# ============================================================================

from flask import Flask, request, jsonify
from inference_local import SpamDetector

app = Flask(__name__)

# Load detector once
detector = SpamDetector()

@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    {
        "text": "komentar yang ingin dicek"
    }
    """
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    result = detector.predict(text)[0]
    return jsonify(result)

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    POST /predict_batch
    {
        "texts": ["text1", "text2", ...]
    }
    """
    data = request.json
    texts = data.get('texts', [])
    
    if not texts:
        return jsonify({"error": "No texts provided"}), 400
    
    results = detector.predict(texts)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

# Usage:
# curl -X POST http://localhost:5000/predict \
#   -H "Content-Type: application/json" \
#   -d '{"text": "subscribe my channel"}'

# ============================================================================
# CONTOH 3: Pandas DataFrame Integration
# ============================================================================

import pandas as pd
from inference_local import SpamDetector

detector = SpamDetector()

# Load data
df = pd.read_csv('comments.csv')

# Predict pada column
df['pred_label'] = None
df['pred_confidence'] = None
df['spam_score'] = None

for idx, row in df.iterrows():
    result = detector.predict(row['comment_text'])[0]
    df.at[idx, 'pred_label'] = result['prediction']
    df.at[idx, 'pred_confidence'] = result['confidence']
    df.at[idx, 'spam_score'] = result['spam_score']

# Save
df.to_csv('comments_with_predictions.csv', index=False)

# ============================================================================
# CONTOH 4: SQL Database Integration
# ============================================================================

import sqlite3
import pandas as pd
from inference_local import SpamDetector

detector = SpamDetector()

# Connect to database
conn = sqlite3.connect('youtube_comments.db')
cursor = conn.cursor()

# Create table for predictions
cursor.execute('''
    CREATE TABLE IF NOT EXISTS comment_predictions (
        comment_id INTEGER PRIMARY KEY,
        comment_text TEXT,
        prediction INTEGER,
        confidence REAL,
        spam_score REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')

# Predict and insert
query = "SELECT id, text FROM comments WHERE prediction IS NULL LIMIT 100"
df = pd.read_sql_query(query, conn)

if not df.empty:
    results = detector.predict(df['text'].tolist())
    
    for idx, result in enumerate(results):
        cursor.execute('''
            INSERT INTO comment_predictions 
            (comment_id, comment_text, prediction, confidence, spam_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            df.iloc[idx]['id'],
            result['text'],
            result['prediction'],
            result['confidence'],
            result['spam_score']
        ))

conn.commit()
conn.close()

# ============================================================================
# CONTOH 5: Asyncio / Background Task
# ============================================================================

import asyncio
from inference_local import SpamDetector

detector = SpamDetector()

async def predict_async(texts):
    """Non-blocking prediction"""
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None, 
        detector.predict, 
        texts
    )
    return results

# Usage in async context
async def process_comments(comment_list):
    results = await predict_async(comment_list)
    for result in results:
        print(f"{result['label']}: {result['text'][:50]}")

# Run
# asyncio.run(process_comments(['text1', 'text2', ...]))

# ============================================================================
# CONTOH 6: Celery Task Queue
# ============================================================================

from celery import Celery
from inference_local import SpamDetector

app = Celery('spam_detector')
detector = SpamDetector()

@app.task(name='predict_spam')
def predict_spam_task(text):
    """Background task untuk prediction"""
    result = detector.predict(text)[0]
    return result

@app.task(name='predict_batch_spam')
def predict_batch_spam_task(texts):
    """Background task untuk batch prediction"""
    results = detector.predict(texts)
    return results

# Usage:
# from celery_tasks import predict_spam_task
# task = predict_spam_task.delay("subscribe my channel")
# result = task.get()

# ============================================================================
# CONTOH 7: Django Integration
# ============================================================================

# models.py
from django.db import models

class YouTubeComment(models.Model):
    comment_text = models.TextField()
    prediction = models.IntegerField(null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    spam_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

# views.py
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from inference_local import SpamDetector

detector = SpamDetector()

@api_view(['POST'])
def predict_comment(request):
    """API endpoint untuk prediction"""
    text = request.data.get('text')
    
    if not text:
        return Response({'error': 'No text provided'}, status=400)
    
    result = detector.predict(text)[0]
    
    # Save ke database
    comment = YouTubeComment.objects.create(
        comment_text=text,
        prediction=result['prediction'],
        confidence=result['confidence'],
        spam_score=result['spam_score']
    )
    
    return Response({
        'comment_id': comment.id,
        'label': result['label'],
        'confidence': result['confidence'],
        'spam_score': result['spam_score']
    })

# ============================================================================
# CONTOH 8: Real-time Streaming (WebSocket)
# ============================================================================

import asyncio
import json
from aiohttp import web
from inference_local import SpamDetector

detector = SpamDetector()

class SpamDetectorWSHandler:
    """WebSocket handler untuk real-time spam detection"""
    
    def __init__(self):
        self.detector = detector
    
    async def handle_message(self, ws, msg):
        try:
            data = json.loads(msg)
            text = data.get('text')
            
            if not text:
                await ws.send_json({'error': 'No text provided'})
                return
            
            # Predict (non-blocking)
            result = self.detector.predict(text)[0]
            await ws.send_json(result)
            
        except json.JSONDecodeError:
            await ws.send_json({'error': 'Invalid JSON'})
        except Exception as e:
            await ws.send_json({'error': str(e)})
    
    async def websocket_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                await self.handle_message(ws, msg.data)
            elif msg.type == web.WSMsgType.ERROR:
                break
        
        return ws

# Setup
handler = SpamDetectorWSHandler()
app = web.Application()
app.router.add_get('/ws', handler.websocket_handler)

if __name__ == '__main__':
    web.run_app(app, port=8080)

# Client usage:
# const ws = new WebSocket('ws://localhost:8080/ws');
# ws.send(JSON.stringify({text: 'subscribe my channel'}));
# ws.onmessage = (event) => console.log(JSON.parse(event.data));

# ============================================================================
# BEST PRACTICES
# ============================================================================

"""
1. **Load Model Once**
   Jangan load model berkali-kali, cukup sekali di initialization.
   
   ✓ BAIK:
   detector = SpamDetector()  # Load once at startup
   for text in texts:
       result = detector.predict(text)
   
   ✗ BURUK:
   for text in texts:
       detector = SpamDetector()  # Load setiap loop!
       result = detector.predict(text)

2. **Use Batch Prediction**
   Untuk multiple predictions, gunakan batch mode lebih cepat.
   
   ✓ BAIK:
   results = detector.predict(texts)  # Batch
   
   ✗ BURUK:
   for text in texts:
       result = detector.predict(text)  # One by one

3. **Cache Results**
   Untuk comments yang sama, cache hasilnya.
   
   @cache.memoized(ttl=3600)
   def get_spam_prediction(text):
       return detector.predict(text)

4. **Error Handling**
   Always wrap prediction dengan try-except.
   
   try:
       result = detector.predict(text)
   except Exception as e:
       logger.error(f"Prediction error: {e}")
       result = None

5. **Monitoring**
   Track metrics seperti prediction time, confidence distribution.
   
   import time
   start = time.time()
   result = detector.predict(text)
   duration = time.time() - start
   logger.info(f"Prediction took {duration:.3f}s, confidence: {result['confidence']:.2%}")

6. **Resource Management**
   Jika GPU, pastikan cleanup model setelah selesai.
   
   import torch
   torch.cuda.empty_cache()  # Setelah selesai

7. **Versioning**
   Track model version untuk reproducibility.
   
   MODEL_VERSION = "v1.0"
   detector = SpamDetector()
   prediction['model_version'] = MODEL_VERSION

8. **Logging**
   Log predictions untuk debugging & audit.
   
   logging.basicConfig()
   logger = logging.getLogger(__name__)
   logger.info(f"Predicted: {text[:50]}, Label: {label}")
"""

# ============================================================================
# PERFORMANCE TIPS
# ============================================================================

"""
Untuk meningkatkan performance:

1. **Batch Processing**
   Gunakan batch size 32-128 untuk optimal throughput
   
2. **GPU Acceleration**
   Jika punya GPU NVIDIA, setup CUDA untuk 10-50x speedup
   
3. **Model Quantization**
   Kurangi model size dengan quantization
   
4. **Caching**
   Implement caching untuk frequent predictions
   
5. **Async Processing**
   Use asyncio/threads untuk non-blocking predictions
   
6. **Model Optimization**
   Consider pruning/distillation untuk production deployment
"""
