import joblib

model = joblib.load('streamlit/chatbot_model.pkl')
vectorizer = joblib.load('streamlit/vectorizer.pkl')

responses = {
    'greeting': "👋 Hello! I'm your digital marketing assistant. How can I help you today?",
    'services': "📢 We offer SEO, Google Ads, Social Media Marketing, Email Campaigns, and more.",
    'pricing': "💰 Our pricing depends on your goals and services needed. Tell me more about your business!",
    'lead_generation': "🚀 Awesome! Please share your name, email, and what you're looking for.",
    'appointment': "📅 Great! You can book a free consultation at [example.com/schedule](https://example.com/schedule)."
}

def get_response(user_input):
    vec = vectorizer.transform([user_input])
    intent = model.predict(vec)[0]
    return responses.get(intent, "❓ I'm not sure I understand. Could you rephrase?")

