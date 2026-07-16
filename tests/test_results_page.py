import os
import wave
import struct
import time
from playwright.sync_api import sync_playwright

def generate_dummy_wav(path, duration=11, sample_rate=16000):
    print(f"Generating dummy WAV file at {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2) # 16-bit
        w.setframerate(sample_rate)
        # Write silent samples
        for _ in range(sample_rate * duration):
            w.writeframesraw(struct.pack('<h', 0))
    print("Dummy WAV generated.")

def test_results_page_alignment():
    mock_wav = "tests/mock_voice.wav"
    generate_dummy_wav(mock_wav)
    
    print("Launching Playwright browser...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()
        
        url = os.environ.get("TEST_URL", "http://localhost:8000")
        print(f"Navigating to {url}...")
        
        try:
            page.goto(url)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(1000)
            
            # Click Start Voice Assessment CTA
            print("Navigating to recording page...")
            start_btn = page.locator(".landing__cta-row button").first
            start_btn.click()
            
            # Wait for recording page
            page.wait_for_selector("text=Voice Screening Protocol")
            
            # Accept consent
            print("Accepting consent...")
            consent_checkbox = page.locator("#consent-checkbox")
            consent_checkbox.check()
            
            # Upload voice sample
            print("Uploading mock voice sample...")
            file_input = page.locator('input[type="file"]')
            file_input.set_input_files(mock_wav)
            
            # Wait for transition to results page (could take a bit for analysis spinner)
            print("Waiting for screening results...")
            page.wait_for_selector(".dashboard", timeout=25000)
            print("Dashboard loaded.")
            
            # Give dashboard charts/plots a brief moment to render fully
            page.wait_for_timeout(2000)
            
            # Take screenshot of the results page
            os.makedirs("tests/screenshots", exist_ok=True)
            screenshot_path = "tests/screenshots/results_page.png"
            page.screenshot(path=screenshot_path, full_page=True)
            print(f"Saved results page screenshot to {screenshot_path}")
            
            print("Results page alignment flow verified successfully!")
            
        except Exception as e:
            print(f"Test failed with exception: {e}")
            page.screenshot(path="tests/screenshots/results_error.png")
            raise e
        finally:
            browser.close()

if __name__ == "__main__":
    test_results_page_alignment()
