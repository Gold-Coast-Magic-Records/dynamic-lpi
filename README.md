# GOLD COAST MAGIC RECORDS

# Dynamic LPI Software

Turn a **WAV file** into a **variable LPI** schedule for record cutting: analyse loud and quiet passages, preview groove pitch on a graph, and write a small **`.idx`** file (`time,LPI` rows) your lathe software can follow during a cut.

Includes a **desktop GUI** (waveform, envelope, LPI curve, editable event list) and a **command-line** mode for batch use or automation (e.g. from a record-sync pipeline).

---

## Disclaimer — read this first

This is **experimental workshop software**, provided **“as is”** with **no warranty**.

- **Use at your own risk.** Wrong LPI values or bugs could damage cutters, blanks, or equipment, or produce unplayable records.
- **You are responsible** for verifying output on your lathe, your ears, and your process before cutting.
- The authors and contributors are **not liable** for any loss, damage, or injury from using this software.

If that is not acceptable, do not use this tool.

This has been provided in good faith, after demand from the lathe cut community for this application.

---

## License

**[MIT License](LICENSE)** — you may use, modify, and redistribute with minimal restrictions; keep the license notice with copies you share. The disclaimer above still applies to real-world cutting.

---

## Get the code

**Git**

```bash
git clone https://github.com/Gold-Coast-Magic-Records/dynamic-lpi.git
cd dynamic-lpi
```

**Download:** GitHub → **Code → Download ZIP**, unzip, open a terminal in the `dynamic-lpi` folder.

---

## New to Python?

1. Install **Python 3.10+** from [python.org](https://www.python.org/downloads/) (on Windows, enable **Add Python to PATH**).
2. Open a terminal in this folder.
3. *(Recommended)* Virtual environment:

   ```bash
   python3 -m venv .venv
   ```

   - macOS/Linux: `source .venv/bin/activate`
   - Windows: `.venv\Scripts\activate`

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. On Linux, GUI audio usually needs PortAudio, e.g. `sudo apt install libportaudio2`.

If `python3` is missing on Windows, try `py -3` instead.

---

## GUI (easiest way to try it)

Put numbered tracks in a side folder, e.g. `1.wav`, `2.wav`, …

```bash
python3 lpi_preview.py --gui /path/to/Side/A
```

Or open one file:

```bash
python3 lpi_preview.py --gui /path/to/1.wav
```

- **Open folder…** / **Open WAV…** — load audio  
- **Generate** — rebuild LPI from the parameters  
- **Play** / Space — preview audio (needs `sounddevice`)  
- Click **Time (s)** to seek; click **LPI** to edit a value; **×** deletes an event  
- **Save .idx…** — write the index next to the WAV (same base name)

Default parameters for our equipment (RPM 33⅓, tight/open LPI, quantize step, etc.) are built into the app; tweak them and regenerate as needed for your particular setup.

---

## Command line (batch)

```bash
python3 lpi_preview.py track.wav
```

Writes `track.idx` beside the WAV. Help:

```bash
python3 lpi_preview.py --help
```

Example with explicit limits:

```bash
python3 lpi_preview.py track.wav --lpi-tight 24000 --lpi-open 18900 --quantize 512
```

---

## Output format

`.idx` files are plain CSV, no header:

```text
0.000,24000
1.850,18900
```

Each line is **seconds since start of that WAV**, then **G2 L** (higher = tighter pitch / slower carriage). The cutter holds each LPI until the next row.

---

## Tests (optional)

```bash
python3 -m unittest discover -s . -p 'test_*.py'
```
