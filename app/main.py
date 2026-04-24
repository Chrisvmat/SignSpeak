"""
main.py
────────
SignSpeak entry point.
Run this on your laptop after training the model in Colab.

Usage:
    python main.py
    python main.py --camera 1          # use external webcam
    python main.py --model my_model.keras
"""

import argparse
import os
import sys

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def check_files(model_path: str, label_map_path: str):
    """Verify required files exist before starting."""
    missing = []
    if not os.path.exists(model_path):
        missing.append(model_path)
    if not os.path.exists(label_map_path):
        missing.append(label_map_path)

    if missing:
        print("\n❌ Missing required files:")
        for f in missing:
            print(f"   - {f}")
        print("\nSteps to fix:")
        print("  1. Run colab/01_preprocess_dataset.py in Google Colab")
        print("  2. Run colab/02_collect_word_signs.py locally (for word signs)")
        print("  3. Run colab/03_train_model.py in Google Colab")
        print("  4. Download signspeak_model.keras and label_map.json from Google Drive")
        print("  5. Place them in the app/ folder")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='SignSpeak — Real-time ASL Sign Language to Speech'
    )
    parser.add_argument(
        '--camera', type=int, default=0,
        help='Camera device index (default: 1)'
    )
    parser.add_argument(
        '--model', type=str,
        default=resource_path('signspeak_model.keras'),
        help='Path to trained model file'
    )
    parser.add_argument(
        '--labels', type=str,
        default=resource_path('label_map.json'),
        help='Path to label map JSON file'
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  SignSpeak — Sign to Speech System")
    print("=" * 50)
    print(f"  Model:   {args.model}")
    print(f"  Labels:  {args.labels}")
    print(f"  Camera:  {args.camera}")
    print("=" * 50 + "\n")

    # Check required files
    check_files(args.model, args.labels)

    # Import here so missing-file error shows before import errors
    from sign_recognition import SignRecognizer

    try:
        recognizer = SignRecognizer(
            model_path=args.model,
            label_map_path=args.labels
        )
        recognizer.run(camera_index=args.camera)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n❌ Runtime error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
