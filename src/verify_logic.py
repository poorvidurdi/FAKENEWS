import os

def mock_predict_multimodal(text_label, mm_prob):
    """
    Simplified version of the logic implemented in multimodal_predictor.py
    """
    fake_reasons = []
    image_out_of_context = mm_prob >= 0.5
    image_label = "Fake" if image_out_of_context else "Real"

    if text_label == "FAKE":
        fake_reasons.append("Text is fake")
        final_decision = "FAKE"
    elif text_label == "UNCERTAIN":
        fake_reasons.append("Sources need to manually verified")
        final_decision = f"UNCERTAIN + {image_label}"
    else:
        if image_out_of_context:
            fake_reasons.append("Image is out-of-context")
            final_decision = "FAKE"
        else:
            final_decision = "REAL"
            
    return final_decision, fake_reasons

def test_logic():
    print("Testing Multimodal Prediction Logic (Simplified)...")
    
    # Case 1: Real News
    final, reasons = mock_predict_multimodal("REAL", 0.1)
    print(f"Test 1 (Real): {final} - Expected: REAL")
    
    # Case 2: Uncertain + Real Image
    final, reasons = mock_predict_multimodal("UNCERTAIN", 0.4)
    print(f"Test 2 (Uncertain Real): {final} - Expected: UNCERTAIN + Real")
    
    # Case 3: Uncertain + Fake Image
    final, reasons = mock_predict_multimodal("UNCERTAIN", 0.6)
    print(f"Test 3 (Uncertain Fake): {final} - Expected: UNCERTAIN + Fake")

    # Case 4: Fake News
    final, reasons = mock_predict_multimodal("FAKE", 0.1)
    print(f"Test 4 (Fake): {final} - Expected: FAKE")

    # Case 5: Fake Image (Real Text)
    final, reasons = mock_predict_multimodal("REAL", 0.9)
    print(f"Test 5 (Fake Image): {final} - Expected: FAKE")

if __name__ == "__main__":
    test_logic()
