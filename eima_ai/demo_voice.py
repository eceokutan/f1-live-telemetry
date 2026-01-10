#!/usr/bin/env python3
"""
Jarvis-Granite Voice Demo Script

Generates audio files from AI race engineer responses using:
- IBM WatsonX Granite LLM for generating responses
- IBM Watson TTS for converting text to speech

Setup:
    1. Copy .env.example to .env
    2. Add your IBM credentials:
       WATSONX_API_KEY=your_key
       WATSONX_PROJECT_ID=your_project_id
       WATSONX_URL=https://eu-gb.ml.cloud.ibm.com
       WATSON_TTS_API_KEY=your_tts_key
       WATSON_TTS_URL=https://api.eu-gb.text-to-speech.watson.cloud.ibm.com

Run:
    python demo_voice.py

Output:
    Creates .wav files in the 'voice_demo_output' folder that you can play.
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ANSI colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.ENDC} {text}")


def print_error(text: str):
    print(f"{Colors.RED}[ERROR]{Colors.ENDC} {text}")


def print_info(text: str):
    print(f"{Colors.CYAN}[INFO]{Colors.ENDC} {text}")


def print_generating(text: str):
    print(f"{Colors.YELLOW}[GENERATING]{Colors.ENDC} {text}")


# Scenarios for the LLM to generate responses for
SCENARIOS = [
    {
        "id": "01_fuel_warning",
        "trigger": "Fuel Warning",
        "prompt": "You are an F1 race engineer. The driver has only 4 laps of fuel remaining and is in P3 at Monza. Give a brief, urgent radio message about fuel and pitting strategy. Keep it under 30 words.",
    },
    {
        "id": "02_tire_critical",
        "trigger": "Tire Critical",
        "prompt": "You are an F1 race engineer. The front right tire temperature is critical at 118C and degrading fast. Tell the driver to box immediately. Keep it under 20 words, be urgent.",
    },
    {
        "id": "03_gap_closing",
        "trigger": "Gap Change",
        "prompt": "You are an F1 race engineer. Hamilton behind has closed the gap from 2.5s to 1.2s in 3 laps. Warn the driver and give defensive advice. Keep it under 30 words.",
    },
    {
        "id": "04_lap_complete",
        "trigger": "Lap Complete",
        "prompt": "You are an F1 race engineer. The driver just completed lap 42 with a time of 1:21.456, matching the leader's pace. They're in P3. Give brief positive feedback. Keep it under 20 words.",
    },
    {
        "id": "05_position_change",
        "trigger": "Position Change",
        "prompt": "You are an F1 race engineer. The driver just overtook Leclerc and is now P2, 3.5 seconds behind Verstappen with 12 laps to go. Give an encouraging update. Keep it under 25 words.",
    },
    {
        "id": "06_driver_query_tires",
        "trigger": "Driver Query: Tires",
        "prompt": "You are an F1 race engineer. The driver asks 'How are my tires?'. Front left is at 95C/45% wear, front right at 98C/48% wear, rears are cooler at 88C/35% wear. Give a concise answer. Keep it under 30 words.",
    },
    {
        "id": "07_driver_query_strategy",
        "trigger": "Driver Query: Strategy",
        "prompt": "You are an F1 race engineer. The driver asks 'What's the plan?'. Current strategy is to stay out on mediums unless safety car, then pit for softs. 15 laps remaining. Explain briefly. Keep it under 30 words.",
    },
    {
        "id": "08_safety_car",
        "trigger": "Safety Car",
        "prompt": "You are an F1 race engineer. Safety car just deployed due to debris. This is the perfect time to pit. Tell the driver to box immediately for fresh soft tires. Be urgent. Keep it under 25 words.",
    },
]


async def generate_audio_files():
    """Generate audio files from LLM-generated responses."""

    print_header("JARVIS-GRANITE VOICE DEMO")
    print_info("This demo uses LLM to generate responses, then TTS to speak them\n")

    # Check for credentials
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print_info("python-dotenv not installed, using environment variables directly")

    # Check LLM credentials
    watsonx_api_key = os.getenv("WATSONX_API_KEY")
    watsonx_project_id = os.getenv("WATSONX_PROJECT_ID")
    watsonx_url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

    # Check TTS credentials
    tts_api_key = os.getenv("WATSON_TTS_API_KEY")
    tts_url = os.getenv("WATSON_TTS_URL", "https://api.us-south.text-to-speech.watson.cloud.ibm.com")

    missing_creds = []
    if not watsonx_api_key:
        missing_creds.append("WATSONX_API_KEY")
    if not watsonx_project_id:
        missing_creds.append("WATSONX_PROJECT_ID")
    if not tts_api_key:
        missing_creds.append("WATSON_TTS_API_KEY")

    if missing_creds:
        print_error(f"Missing credentials: {', '.join(missing_creds)}")
        print()
        print("Required environment variables:")
        print(f"  {Colors.YELLOW}WATSONX_API_KEY=your_key{Colors.ENDC}")
        print(f"  {Colors.YELLOW}WATSONX_PROJECT_ID=your_project_id{Colors.ENDC}")
        print(f"  {Colors.YELLOW}WATSONX_URL=https://eu-gb.ml.cloud.ibm.com{Colors.ENDC}")
        print(f"  {Colors.YELLOW}WATSON_TTS_API_KEY=your_tts_key{Colors.ENDC}")
        print(f"  {Colors.YELLOW}WATSON_TTS_URL=https://api.eu-gb.text-to-speech.watson.cloud.ibm.com{Colors.ENDC}")
        return

    # Import clients
    try:
        from jarvis_granite.voice.watson_tts import WatsonTTSClient
        from jarvis_granite.llm import LLMClient
    except ImportError as e:
        print_error(f"Could not import required modules: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        return

    # Show configuration
    print_info(f"LLM: WatsonX Granite at {watsonx_url}")
    print_info(f"TTS: Watson TTS at {tts_url}")
    print_info("Voice: en-GB_JamesV3Voice (British Male)\n")

    # Create output directory
    output_dir = Path("voice_demo_output")
    output_dir.mkdir(exist_ok=True)

    # Create LLM client
    llm_client = LLMClient(
        watsonx_url=watsonx_url,
        watsonx_project_id=watsonx_project_id,
        watsonx_api_key=watsonx_api_key,
        max_tokens=100,
        temperature=0.7,
    )

    # Create TTS client
    tts_client = WatsonTTSClient(
        api_key=tts_api_key,
        service_url=tts_url,
        voice="en-GB_JamesV3Voice",
        timeout=30.0
    )

    print_header("GENERATING LLM RESPONSES + AUDIO")

    generated_files = []
    llm_times = []
    tts_times = []

    for scenario in SCENARIOS:
        filename = f"{scenario['id']}.wav"
        filepath = output_dir / filename

        print(f"\n{Colors.CYAN}[{scenario['trigger']}]{Colors.ENDC}")

        # Step 1: Generate response with LLM
        print_generating("LLM response...")
        try:
            llm_start = time.time()
            response_text = await llm_client.invoke(scenario['prompt'])
            llm_time = (time.time() - llm_start) * 1000
            llm_times.append(llm_time)

            # Clean up response (remove quotes if present)
            response_text = response_text.strip().strip('"').strip("'")

            print_success(f"LLM ({llm_time:.0f}ms): \"{response_text[:60]}{'...' if len(response_text) > 60 else ''}\"")

        except Exception as e:
            print_error(f"LLM failed: {e}")
            continue

        # Step 2: Convert to audio with TTS
        print_generating("TTS audio...")
        try:
            tts_start = time.time()
            audio_bytes = await tts_client.synthesize(response_text)
            tts_time = (time.time() - tts_start) * 1000
            tts_times.append(tts_time)

            # Save to file
            filepath.write_bytes(audio_bytes)

            # Calculate duration estimate
            word_count = len(response_text.split())
            duration_est = word_count / 2.5

            print_success(f"TTS ({tts_time:.0f}ms): Saved {filepath.name} ({len(audio_bytes):,} bytes, ~{duration_est:.1f}s)")
            generated_files.append((filepath, response_text))

        except Exception as e:
            print_error(f"TTS failed: {e}")

    # Summary
    print_header("DEMO COMPLETE")

    if generated_files:
        print_success(f"Generated {len(generated_files)} audio files in: {output_dir.absolute()}\n")

        print("Files created:")
        for filepath, text in generated_files:
            print(f"  - {filepath.name}")

        print(f"\n{Colors.CYAN}Play these files to hear the AI race engineer!{Colors.ENDC}")
        print(f"\nOn Windows: double-click any .wav file")
        print(f"On Mac/Linux: afplay {generated_files[0][0]} or aplay {generated_files[0][0]}")

        # Statistics
        print(f"\n{Colors.BLUE}Performance Statistics:{Colors.ENDC}")
        if llm_times:
            print(f"  LLM Avg Latency: {sum(llm_times)/len(llm_times):.0f}ms")
        if tts_times:
            print(f"  TTS Avg Latency: {sum(tts_times)/len(tts_times):.0f}ms")
        print(f"  Total Responses: {len(generated_files)}")

        # Save responses to a text file for reference
        responses_file = output_dir / "responses.txt"
        with open(responses_file, "w", encoding="utf-8") as f:
            f.write("# LLM-Generated Race Engineer Responses\n\n")
            for filepath, text in generated_files:
                f.write(f"## {filepath.stem}\n")
                f.write(f"{text}\n\n")
        print(f"\n  Responses saved to: {responses_file}")

    else:
        print_error("No audio files were generated. Check your credentials.")


async def generate_custom_audio():
    """Interactive mode to generate custom audio."""

    print_header("CUSTOM VOICE GENERATION")

    # Check credentials
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("WATSON_TTS_API_KEY")
    service_url = os.getenv("WATSON_TTS_URL", "https://api.us-south.text-to-speech.watson.cloud.ibm.com")

    if not api_key or api_key == "your_tts_api_key":
        print_error("Watson TTS credentials required for custom audio generation.")
        return

    from jarvis_granite.voice.watson_tts import WatsonTTSClient

    tts_client = WatsonTTSClient(
        api_key=api_key,
        service_url=service_url,
        voice="en-GB_JamesV3Voice",
    )

    output_dir = Path("voice_demo_output")
    output_dir.mkdir(exist_ok=True)

    print("Enter text to convert to speech (or 'quit' to exit):\n")

    counter = 1
    while True:
        try:
            text = input(f"{Colors.CYAN}Text>{Colors.ENDC} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break

        if text.lower() in ('quit', 'exit', 'q'):
            break

        if not text:
            continue

        filename = f"custom_{counter:02d}.wav"
        filepath = output_dir / filename

        try:
            print_generating("Converting to speech...")
            audio_bytes = await tts_client.synthesize(text)
            filepath.write_bytes(audio_bytes)
            print_success(f"Saved: {filepath}")
            counter += 1
        except Exception as e:
            print_error(f"Failed: {e}")

        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Jarvis-Granite Voice Demo")
    parser.add_argument(
        "--custom", "-c",
        action="store_true",
        help="Interactive mode to generate custom audio"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Generate audio for specific text"
    )

    args = parser.parse_args()

    if args.custom:
        asyncio.run(generate_custom_audio())
    elif args.text:
        # Quick one-off generation
        async def quick_generate():
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass

            api_key = os.getenv("WATSON_TTS_API_KEY")
            service_url = os.getenv("WATSON_TTS_URL", "https://api.us-south.text-to-speech.watson.cloud.ibm.com")

            if not api_key or api_key == "your_tts_api_key":
                print_error("Watson TTS credentials required.")
                return

            from jarvis_granite.voice.watson_tts import WatsonTTSClient

            tts_client = WatsonTTSClient(
                api_key=api_key,
                service_url=service_url,
                voice="en-GB_JamesV3Voice",
            )

            output_dir = Path("voice_demo_output")
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = output_dir / f"output_{timestamp}.wav"

            print_generating(f"Converting: \"{args.text[:50]}...\"")
            audio_bytes = await tts_client.synthesize(args.text)
            filepath.write_bytes(audio_bytes)
            print_success(f"Saved: {filepath}")

        asyncio.run(quick_generate())
    else:
        asyncio.run(generate_audio_files())
