from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os

app = Flask(__name__)

# Gemini setup — reads API key from environment variable (safe for deployment)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    goals = data.get("goals", [])
    history = data.get("history", [])  # past messages for memory

    # Build a goals summary to give the AI context
    if goals:
        goals_text = "\n".join([
            f"- {g['title']} | Category: {g['category']} | Progress: {g['progress']}%"
            for g in goals
        ])
    else:
        goals_text = "No goals added yet."

    system_prompt = f"""You are a warm, direct personal AI coach and second brain.
Your job is to keep the user accountable, motivated, and on track.

Here are the user's current goals:
{goals_text}

Guidelines:
- Be specific and actionable, not vague
- Keep replies to 2-4 sentences unless the user asks for more
- Reference their actual goals when relevant
- If they're struggling, be empathetic but keep them moving forward
"""

    # Convert history into Gemini's format
    gemini_history = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({
            "role": role,
            "parts": [msg["content"]]
        })

    # Start chat with history, then send new message
    chat_session = model.start_chat(history=gemini_history)
    full_message = f"{system_prompt}\n\nUser message: {user_message}"
    response = chat_session.send_message(full_message)

    return jsonify({"reply": response.text})

if __name__ == "__main__":
    app.run(debug=True)	
