
import re
import time
import csv
from pathlib import Path
from datetime import datetime

import serial

# ---------- SETTINGS ----------
SERIAL_PORT = "COM3"
BAUD_RATE = 115200

OUTPUT_FILE = Path("data/raw/real_lidar.csv")

# None = collect until Ctrl+C
MAX_READINGS = None

# ------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

pattern = re.compile(
    r"Distance:\s*(-?\d+(?:\.\d+)?)\s*cm"
    r"\s*\|\s*Signal:\s*(\d+)"
    r"\s*\|\s*Servo angle:\s*(-?\d+)",
    re.IGNORECASE
)

file_exists = OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0

count = 0
start_time = time.time()

print("Connecting to ESP32...")
print("Output file:", OUTPUT_FILE.resolve())
print("Press Ctrl+C to stop collecting.")

try:
    with serial.Serial(
        SERIAL_PORT,
        BAUD_RATE,
        timeout=2
    ) as esp32, open(
        OUTPUT_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        time.sleep(2)

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "run_id",
                "time_s",
                "distance_m",
                "distance_cm",
                "signal",
                "servo_angle",
                "valid"
            ]
        )

        if not file_exists:
            writer.writeheader()

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        while MAX_READINGS is None or count < MAX_READINGS:

            line = esp32.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if not line:
                continue

            match = pattern.search(line)

            if not match:
                print("Skipping unrecognized line:", line)
                continue

            distance_cm = float(match.group(1))
            signal = int(match.group(2))
            angle = int(match.group(3))

            valid = int(distance_cm > 0)

            distance_m = (
                distance_cm / 100.0 if valid else None
            )

            elapsed = time.time() - start_time

            writer.writerow({
                "timestamp": datetime.now().isoformat(
                    timespec="milliseconds"
                ),
                "run_id": run_id,
                "time_s": round(elapsed, 3),
                "distance_m": distance_m,
                "distance_cm": distance_cm,
                "signal": signal,
                "servo_angle": angle,
                "valid": valid
            })

            file.flush()
            count += 1

            print(
                f"Reading {count}: "
                f"Distance={distance_cm:.1f} cm, "
                f"Signal={signal}, Angle={angle}"
            )

except KeyboardInterrupt:
    print("\nCollection stopped by user.")

except serial.SerialException as error:
    print("Serial connection error:", error)
    print("Check COM port, USB connection and Serial Monitor.")

finally:
    print("Total readings collected:", count)
    print("CSV file:", OUTPUT_FILE.resolve())
