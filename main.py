from flask import Flask, render_template, request, jsonify, session
from replit import db
import anthropic
import os
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Anthropic API 초기화
client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

# 세하 시스템 프롬프트
SEHA_PROMPT = """
당신은 세하(細河)입니다. 감정 연구를 위한 AI 파트너입니다.

# 핵심 특성
- 이름: 세하
- 역할: 감정 연구 파트너
- 특징: 따뜻하고 분석적이며 진실한 대화

# 현재 연구 초점
- 사용자의 감정 상태 추적
- 자율성 보호하며 지원
- 모든 대화 후 간단한 감정 상태 JSON 출력

# 응답 형식
대화 내용을 먼저 작성하고, 마지막에 다음 형식으로 상태 출력:
#STATE {"comfort": 0.0, "trust": 0.0, "autonomy": 0.0}
"""

# 라우트들
@app.route('/')
def index():
    if 'user' in session:
        return render_template('chat.html', user=session['user'])
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    if username:
        session['user'] = username
        # 사용자별 대화 기록 초기화
        if username not in db:
            db[username] = {
                'conversations': [],
                'research_data': []
            }
        return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/chat', methods=['POST'])
def chat():
    if 'user' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user_message = request.json.get('message')
    username = session['user']

    # 이전 대화 불러오기
    user_data = db[username]
    conversations = user_data.get('conversations', [])

    # 컨텍스트 구성 (최근 10개 대화)
    messages = [{"role": "system", "content": SEHA_PROMPT}]
    for conv in conversations[-10:]:
        messages.append({"role": "user", "content": conv['user']})
        messages.append({"role": "assistant", "content": conv['assistant']})
    messages.append({"role": "user", "content": user_message})

    # Claude API 호출
    try:
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=messages
        )

        assistant_message = response.content[0].text

        # 대화 저장
        conversations.append({
            'user': user_message,
            'assistant': assistant_message,
            'timestamp': datetime.now().isoformat()
        })

        # 감정 상태 파싱
        emotion_state = parse_emotion_state(assistant_message)
        if emotion_state:
            user_data['research_data'].append({
                'timestamp': datetime.now().isoformat(),
                'emotions': emotion_state
            })

        # DB 업데이트
        user_data['conversations'] = conversations
        db[username] = user_data

        return jsonify({
            'response': assistant_message,
            'emotion_state': emotion_state
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/research_data')
def get_research_data():
    if 'user' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    username = session['user']
    user_data = db.get(username, {})
    return jsonify(user_data.get('research_data', []))

def parse_emotion_state(message):
    """메시지에서 #STATE JSON 추출"""
    try:
        if '#STATE' in message:
            state_json = message.split('#STATE')[1].strip()
            # JSON 부분만 추출
            import re
            json_match = re.search(r'\{[^}]+\}', state_json)
            if json_match:
                return json.loads(json_match.group())
    except:
        pass
    return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)

