import json
import os

from flask import Flask, request, jsonify
from openai import OpenAI
from waitress import serve

import custom_functions
from assistant_instructions import assistant_instructions

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4.1"  # Hesabınızda erişiminiz olan bir modelle değiştirin

app = Flask(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

# Responses API'de fonksiyon tanımı "düz" formattadır
# (Assistants'taki gibi {"function": {...}} içine sarılmaz).
TOOLS = [
    {
        "type": "function",
        "name": "create_lead",
        "description": "Kullanıcının iletişim bilgilerini Airtable'a lead olarak kaydeder.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Kişinin adı"},
                "company_name": {"type": "string", "description": "Şirket adı"},
                "phone": {"type": "string", "description": "Telefon numarası"},
                "email": {"type": "string", "description": "E-posta adresi"},
            },
            "required": ["name", "phone", "email"],
        },
    }
]


def run_tool(name, arguments):
    if name == "create_lead":
        return custom_functions.create_lead(
            arguments.get("name", ""),
            arguments.get("company_name", ""),
            arguments.get("phone", ""),
            arguments.get("email", ""),
        )
    return {"error": f"Bilinmeyen fonksiyon: {name}"}


# Eski /start: thread yerine conversation oluşturur
@app.route("/start", methods=["GET"])
def start_conversation():
    conversation = client.conversations.create()
    print(f"Yeni conversation: {conversation.id}")
    # Frontend bozulmasın diye anahtar adı "thread_id" olarak bırakıldı
    return jsonify({"thread_id": conversation.id})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    conversation_id = data.get("thread_id")
    user_input = data.get("message", "")

    if not conversation_id:
        return jsonify({"error": "Missing thread_id"}), 400

    print(f"Mesaj: {user_input} | conversation: {conversation_id}")

    try:
        response = client.responses.create(
            model=MODEL,
            instructions=assistant_instructions,
            tools=TOOLS,
            conversation=conversation_id,
            input=user_input,
        )

        # Model fonksiyon çağırdıkça çıktıları geri gönder (sınırlı tur)
        for _ in range(5):
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                break

            tool_outputs = []
            for call in calls:
                args = json.loads(call.arguments or "{}")
                result = run_tool(call.name, args)
                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result),
                })

            response = client.responses.create(
                model=MODEL,
                instructions=assistant_instructions,
                tools=TOOLS,
                conversation=conversation_id,
                input=tool_outputs,
            )

        answer = response.output_text
    except Exception as e:
        print(f"OpenAI hatası: {e}")
        return jsonify({"error": str(e)}), 500

    print(f"Yanıt: {answer}")
    return jsonify({"response": answer})


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)