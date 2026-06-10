import os
import sys
import time
from playwright.sync_api import sync_playwright

def test_frontend_navigation():
    print("Launching Playwright browser...")
    with sync_playwright() as p:
        # Launch headless Chromium browser
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()
        
        # Address of local Vite development server
        url = "http://localhost:5173"
        print(f"Navigating to {url}...")
        
        try:
            page.goto(url)
            # Wait for JS and resources to load completely
            page.wait_for_load_state("networkidle")
            # 1. Verify landing page elements
            print("Verifying landing page...")
            page.wait_for_timeout(1000)
            assert "VitaVoice" in page.title() or page.locator("text=VitaVoice").first.is_visible()
            assert page.locator("text=Vocal Biomarker AI").first.is_visible()
            
            # Take landing screenshot
            os.makedirs("tests/screenshots", exist_ok=True)
            page.screenshot(path="tests/screenshots/landing_page.png")
            print("Saved landing page screenshot to tests/screenshots/landing_page.png")
            
            # 2. Click assessment button
            print("Clicking 'Start Voice Assessment'...")
            start_btn = page.locator("#start-assessment-btn")
            assert start_btn.is_enabled()
            start_btn.click()
            
            # Wait for transition
            page.wait_for_selector("text=Voice Screening Protocol")
            
            # 3. Verify recording interface features
            print("Verifying Voice Screening Protocol layout...")
            assert page.locator("text=Voice Screening Protocol").is_visible()
            assert page.locator("text=Step 1: Acoustic Environment Calibration").is_visible()
            assert page.locator("text=Clinical Consent Agreement").is_visible()
            assert page.locator("text=Begin Voice Recording").is_visible()
            
            # Take recorder screenshot
            page.screenshot(path="tests/screenshots/recorder_page.png")
            print("Saved recorder page screenshot to tests/screenshots/recorder_page.png")
            
            print("Frontend UI flow verified successfully!")
            
        except Exception as e:
            print(f"Test failed with exception: {e}")
            # Save error screenshot for debugging
            page.screenshot(path="tests/screenshots/error_state.png")
            raise e
        finally:
            browser.close()

if __name__ == "__main__":
    test_frontend_navigation()
