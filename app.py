import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title='Ambiguous Email / Message Detector', page_icon='💬')

messages = [
'Please finish it soon','Send it later','Do that quickly','Meet me there','Maybe we can do it','I will send it sometime','Complete this later','Please handle that','Come there tomorrow','Do it as soon as possible','Send this to them','We can discuss it later','Please fix that','Finish the work quickly','I will do it soon','Maybe send it today','Meet me somewhere','Please check this','Do that when you can','Send the file sometime',
'Please complete the report by 5 PM today','Submit the assignment before 10 AM tomorrow','The meeting is at 3 PM in Room 204','Send the project report to the manager today','Please submit the file before Friday','The meeting will start at 10 AM','Submit the database assignment by Monday','Please send the report to Ravi by 4 PM','The presentation is scheduled for 2 PM tomorrow','Complete the Python project before Friday','Upload the assignment to the portal by 6 PM','Meet me in Room 305 at 11 AM','Send the document to the team before 5 PM','The exam starts at 9 AM on Monday','Submit the project report on October 10','Please attend the meeting at 3 PM','Complete the Java program by tomorrow evening','Send the email to the professor before noon','The deadline for the assignment is Friday at 5 PM','Please bring the project file to class tomorrow']
labels = ['Ambiguous']*20 + ['Clear']*20
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(messages)
model = LogisticRegression()
model.fit(X, labels)

emotion_words = {
'Happy':['happy','great','excellent','thank','thanks','excited','love','wonderful','glad'],
'Sad':['sad','disappointed','upset','hurt','unhappy','sorry','miss'],
'Angry':['angry','annoyed','stop','shut','hate','frustrated'],
'Worried':['worried','afraid','scared','concerned','nervous','anxious'],
'Uncertain':['maybe','perhaps','not sure','might','possibly','uncertain']}

def detect_emotion(text):
    text = text.lower()
    scores = {emotion: sum(1 for word in words if word in text) for emotion, words in emotion_words.items()}
    detected = max(scores, key=scores.get)
    return 'Neutral' if scores[detected] == 0 else detected

emotion_explanations = {
'Happy':'The message contains words that express happiness or positivity.',
'Sad':'The message contains words that may express sadness or disappointment.',
'Angry':'The message contains words that may express anger, frustration, or a harsh tone.',
'Worried':'The message contains words that may express concern or worry.',
'Uncertain':'The message contains words that express doubt or uncertainty.',
'Neutral':'No strong emotional words were detected in the message.'}

st.title('💬 Ambiguous Email / Message Detector')
st.write('Analyze a message for ambiguity and emotional tone.')
message = st.text_area('Enter your message:', placeholder='Example: Maybe we can discuss it later.')

if st.button('Analyze Message'):
    if not message.strip():
        st.warning('Please enter a message.')
    else:
        user_data = vectorizer.transform([message])
        prediction = model.predict(user_data)[0]
        probabilities = model.predict_proba(user_data)[0]
        confidence = max(probabilities) * 100

        st.subheader('🔍 Ambiguity Analysis')
        st.write(f'**Prediction:** {prediction}')
        st.write(f'**Confidence:** {confidence:.2f}%')

        if prediction == 'Ambiguous':
            st.warning('⚠️ This message may be ambiguous.')
            words = re.findall(r"\b[\w']+\b", message.lower())
            ambiguous_words = {
                'it': "The word 'it' may not clearly identify what it refers to.",
                'this': "The word 'this' may be unclear without context.",
                'that': "The word 'that' may refer to more than one thing.",
                'soon': "The word 'soon' does not specify an exact time.",
                'later': "The word 'later' does not specify when.",
                'maybe': "The word 'maybe' shows uncertainty.",
                'there': "The word 'there' does not specify a clear location."}
            found = [word for word in ambiguous_words if word in words]
            for word in found:
                st.write(f'**Possible ambiguous word:** `{word}`')
                st.write(f'**Why:** {ambiguous_words[word]}')
            st.subheader('💡 Suggested Improvement')
            clearer_message = message
            replacements = {'it':'the specific task','this':'the mentioned item','that':'the mentioned item','soon':'by 5 PM today','later':'at 3 PM today','there':'in Room 204'}
            for word, replacement in replacements.items():
                if word in found:
                    clearer_message = re.sub(rf'\b{word}\b', replacement, clearer_message, flags=re.IGNORECASE)
            st.success(clearer_message)
        else:
            st.success('✅ This message appears to be clear.')

        emotion = detect_emotion(message)
        emotion_icons = {'Happy':'😊','Sad':'😢','Angry':'😠','Worried':'😟','Uncertain':'🤔','Neutral':'😐'}
        st.subheader('❤️ Emotion / Tone Analysis')
        st.write(f"**Detected Emotion:** {emotion_icons.get(emotion, '😐')} {emotion}")
        st.info(emotion_explanations[emotion])
