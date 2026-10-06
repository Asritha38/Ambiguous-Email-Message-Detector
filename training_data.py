from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

messages = [
    # Ambiguous messages
    "Please finish it soon",
    "Send it later",
    "Do that quickly",
    "Meet me there",
    "Maybe we can do it",
    "I will send it sometime",
    "Complete this later",
    "Please handle that",
    "Come there tomorrow",
    "Do it as soon as possible",
    "Send this to them",
    "We can discuss it later",
    "Please fix that",
    "Finish the work quickly",
    "I will do it soon",
    "Maybe send it today",
    "Meet me somewhere",
    "Please check this",
    "Do that when you can",
    "Send the file sometime",

    # Clear messages
    "Please complete the report by 5 PM today",
    "Submit the assignment before 10 AM tomorrow",
    "The meeting is at 3 PM in Room 204",
    "Send the project report to the manager today",
    "Please submit the file before Friday",
    "The meeting will start at 10 AM",
    "Submit the database assignment by Monday",
    "Please send the report to Ravi by 4 PM",
    "The presentation is scheduled for 2 PM tomorrow",
    "Complete the Python project before Friday",
    "Upload the assignment to the portal by 6 PM",
    "Meet me in Room 305 at 11 AM",
    "Send the document to the team before 5 PM",
    "The exam starts at 9 AM on Monday",
    "Submit the project report on October 10",
    "Please attend the meeting at 3 PM",
    "Complete the Java program by tomorrow evening",
    "Send the email to the professor before noon",
    "The deadline for the assignment is Friday at 5 PM",
    "Please bring the project file to class tomorrow",
]

labels = [
    "Ambiguous", "Ambiguous", "Ambiguous", "Ambiguous",
    "Ambiguous", "Ambiguous", "Ambiguous", "Ambiguous",
    "Ambiguous", "Ambiguous", "Ambiguous", "Ambiguous",
    "Ambiguous", "Ambiguous", "Ambiguous", "Ambiguous",
    "Ambiguous", "Ambiguous", "Ambiguous", "Ambiguous",

    "Clear", "Clear", "Clear", "Clear", "Clear",
    "Clear", "Clear", "Clear", "Clear", "Clear",
    "Clear", "Clear", "Clear", "Clear", "Clear",
    "Clear", "Clear", "Clear", "Clear", "Clear",
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(messages)

model = LogisticRegression()
model.fit(X, labels)

print("Model trained successfully!")
print("Total training messages:", len(messages))
from sklearn.metrics import accuracy_score

predictions = model.predict(X)

accuracy = accuracy_score(labels, predictions)

print("Model accuracy:", accuracy * 100, "%")