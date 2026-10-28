"""
One-command demonstration runner for ML-Based Predictive Parser.

IMPORTANT:
- Save this file inside the project's `code` folder, beside test_part1.py.
- Run it from that folder with: python run_all_demo.py
- It saves the full combined output to demo_results.txt in the same folder.
"""
from pathlib import Path
import subprocess
import sys
from datetime import datetime

# This file MUST be placed next to test_part1.py, train_part2.py, etc.
BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "demo_results.txt"

SCRIPTS = [
    "test_part1.py",
    "train_part2.py",
    "part3_exp1.py",
    "part3_exp2.py",
]

def main():
    (BASE_DIR / "models").mkdir(exist_ok=True)
    all_ok = True

    with LOG_PATH.open("w", encoding="utf-8") as log:
        header = (
            "ML-BASED PREDICTIVE PARSER — DEMONSTRATION RESULTS\n"
            f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}\n"
            f"Python: {sys.version.split()[0]}\n"
            f"Project folder: {BASE_DIR}\n"
            + "=" * 76 + "\n"
        )
        print(header, end="")
        log.write(header)
        log.flush()

        for script_name in SCRIPTS:
            script_path = BASE_DIR / script_name
            section = f"\n\n{'=' * 76}\nRUNNING: {script_name}\n{'=' * 76}\n"
            print(section, end="")
            log.write(section)
            log.flush()

            if not script_path.is_file():
                message = (
                    f"ERROR: Cannot find {script_name}.\n"
                    f"Expected location: {script_path}\n"
                    "Save run_all_demo.py in the same folder as the four scripts.\n"
                )
                print(message, end="")
                log.write(message)
                all_ok = False
                break

            print(f"Running {script_name}; output is being collected...\n")
            try:
                completed = subprocess.run(
                    [sys.executable, "-u", str(script_path)],
                    cwd=str(BASE_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=1800,
                )
                output = completed.stdout or "(No output produced.)\n"
                print(output, end="" if output.endswith("\n") else "\n")
                log.write(output)
                log.flush()

                if completed.returncode != 0:
                    message = (
                        f"\nFAILED: {script_name} exited with code "
                        f"{completed.returncode}. Stopping so errors are not hidden.\n"
                    )
                    print(message, end="")
                    log.write(message)
                    all_ok = False
                    break

                message = f"\nCOMPLETED: {script_name}\n"
                print(message, end="")
                log.write(message)
                log.flush()

            except subprocess.TimeoutExpired:
                message = f"\nTIMEOUT: {script_name} exceeded 30 minutes. Stopping.\n"
                print(message, end="")
                log.write(message)
                all_ok = False
                break

        ending = (
            "\n" + "=" * 76 + "\n"
            + ("ALL FOUR SCRIPTS COMPLETED SUCCESSFULLY.\n"
               if all_ok else
               "RUN STOPPED. Check the error above before presenting results.\n")
            + f"Full output saved to: {LOG_PATH}\n"
        )
        print(ending, end="")
        log.write(ending)

    print("\nNext: open demo_results.txt in VS Code to show the collected output.")
    if not all_ok:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
