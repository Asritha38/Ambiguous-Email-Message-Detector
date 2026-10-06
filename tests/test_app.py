from app import detect_ambiguous_terms, suggest_clearer_message


def test_detects_common_ambiguous_terms():
    terms = detect_ambiguous_terms("Please finish it soon")
    assert "it" in terms
    assert "soon" in terms


def test_suggests_clearer_message_for_ambiguous_phrase():
    improved = suggest_clearer_message("Send the file sometime")
    assert "specific" in improved.lower() or "by 3 pm" in improved.lower()


def test_fallback_suggestion_for_unknown_ambiguous_text():
    improved = suggest_clearer_message("Can you handle that when you can?")
    assert "specific" in improved.lower() or "deadline" in improved.lower()
