import re

def is_definitely_neutral(text):
    """Rule-based system to identify obviously neutral content"""
    if not text:
        return False
    
    text = text.lower().strip()
    
    # Compliment patterns - almost always neutral
    compliment_patterns = [
        r'(look|looks|looking)\s(good|great|nice|pretty|beautiful|handsome)',
        r'(is|are|seem|seems)\s(good|great|nice|pretty|beautiful|handsome)',
        r'(you|he|she|they)\s(are|is)\s(good|great|nice|pretty|beautiful|handsome)'
    ]
    
    for pattern in compliment_patterns:
        if re.search(pattern, text):
            return True
    
    # Weight-related neutral comments (common source of false positives)
    weight_neutral_patterns = [
        r'(gained|lost)\s(weight|pounds|kilos)',
        r'(look|looks)\s(thinner|bigger|smaller|larger)'
    ]
    
    # Only classify as neutral if there are no clear offensive terms
    offensive_terms = ['fat', 'ugly', 'stupid', 'idiot', 'dumb', 'hate']
    has_offensive = any(term in text.split() for term in offensive_terms)
    
    if not has_offensive:
        for pattern in weight_neutral_patterns:
            if re.search(pattern, text):
                return True
    
    return False

def detect_positive_context(text):
    """Detect if potentially sensitive topics are in a positive context"""
    text = text.lower()
    
    # Break text into segments
    segments = re.split(r'[.,;:!?]', text)
    
    # Words that could be problematic in some contexts
    sensitive_words = ['weight', 'fat', 'thin', 'look', 'looks', 'size']
    
    # Positive context markers
    positive_markers = ['good', 'great', 'nice', 'pretty', 'beautiful', 
                        'handsome', 'amazing', 'love', 'like', 'still']
    
    # Negation markers that flip meaning
    negations = ['not', "isn't", "aren't", "wasn't", "weren't", "don't", 
                "doesn't", "didn't", "no", "never"]
    
    for segment in segments:
        words = segment.split()
        
        # Skip very short segments
        if len(words) < 2:
            continue
        
        # Check if segment contains sensitive words
        has_sensitive = any(word in sensitive_words for word in words)
        
        if has_sensitive:
            # Check for positive markers
            has_positive = any(marker in words for marker in positive_markers)
            
            # Check for negated positive (becomes negative)
            for i, word in enumerate(words):
                if word in positive_markers and i > 0:
                    if any(neg in words[max(0, i-3):i] for neg in negations):
                        has_positive = False
                        break
            
            if has_positive:
                return True
    
    return False

def enhance_classification(text, prediction, probabilities):
    """Apply rules to correct model predictions"""
    # Apply neutral check
    if is_definitely_neutral(text) or detect_positive_context(text):
        return "neutral", {
            "neutral": 0.85,
            "offensive": 0.10,
            "hate_speech": 0.05
        }
    
    # Adjust borderline hate speech predictions
    if prediction == 'hate_speech' and probabilities.get('hate_speech', 1.0) < 0.65:
        # Less confident hate speech - downgrade to offensive
        return "offensive", {
            "neutral": 0.15,
            "offensive": 0.75,
            "hate_speech": 0.10
        }
    
    # Keep original prediction if no rules apply
    return prediction, probabilities
