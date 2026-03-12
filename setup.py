#!/usr/bin/env python3
"""
setup.py — Instalare automata dependinte MacroMicro Training
RTX 5090 (Blackwell) — Windows — Python 3.11

Rulare: python setup.py
"""

import subprocess
import sys
import os

def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n[EROARE] Comanda a esuat: {cmd}")
        print("Incearca sa rulezi manual comanda de mai sus si trimite eroarea fratelui tau.")
        sys.exit(1)
    print(f"  [OK] {desc}")

def check_python():
    print(f"\n{'='*60}")
    print(f"  Verificare versiune Python...")
    print(f"{'='*60}")
    major = sys.version_info.major
    minor = sys.version_info.minor
    print(f"  Python {major}.{minor} detectat.")
    if major != 3 or minor != 11:
        print(f"\n[EROARE] Ai Python {major}.{minor} dar trebuie Python 3.11!")
        print("Descarca Python 3.11.9 de la:")
        print("https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe")
        print("La instalare, bifeaza 'Add Python to PATH'!")
        sys.exit(1)
    print("  [OK] Python 3.11")

def check_nvidia():
    print(f"\n{'='*60}")
    print(f"  Verificare GPU NVIDIA...")
    print(f"{'='*60}")
    result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("[EROARE] nvidia-smi nu a fost gasit.")
        print("Instaleaza driverele NVIDIA sau verifica ca GPU-ul e conectat corect.")
        sys.exit(1)
    # Extrage numele GPU-ului
    for line in result.stdout.splitlines():
        if "GeForce" in line or "RTX" in line or "GTX" in line or "Quadro" in line:
            print(f"  GPU detectat: {line.strip()}")
            break
    print("  [OK] NVIDIA GPU gasit")

def main():
    print("\n" + "#"*60)
    print("#  MacroMicro — Setup Antrenament AI")
    print("#  RTX 5090 / Windows / Python 3.11")
    print("#"*60)

    # 1. Verificari initiale
    check_python()
    check_nvidia()

    # 2. Upgrade pip
    run(
        f'"{sys.executable}" -m pip install --upgrade pip',
        "Upgrade pip..."
    )

    # 3. Dezinstaleaza PyTorch vechi (daca exista)
    print(f"\n{'='*60}")
    print("  Curatare PyTorch vechi (daca exista)...")
    print(f"{'='*60}")
    subprocess.run(
        f'"{sys.executable}" -m pip uninstall torch torchvision torchaudio -y',
        shell=True,
        capture_output=True
    )
    print("  [OK] Curatare completa")

    # 4. PyTorch Nightly cu CUDA 12.8 (obligatoriu pentru RTX 5090 Blackwell)
    run(
        f'"{sys.executable}" -m pip install --pre torch torchvision torchaudio '
        f'--index-url https://download.pytorch.org/whl/nightly/cu128',
        "Instalare PyTorch Nightly cu CUDA 12.8 (pentru RTX 5090)... [poate dura 10-20 min]"
    )

    # 5. Unsloth
    run(
        f'"{sys.executable}" -m pip install unsloth',
        "Instalare Unsloth..."
    )

    # 6. Restul dependintelor
    run(
        f'"{sys.executable}" -m pip install '
        f'trl transformers datasets accelerate bitsandbytes pyarrow',
        "Instalare trl, transformers, datasets, accelerate, bitsandbytes, pyarrow..."
    )

    # 7. Verificare finala CUDA
    print(f"\n{'='*60}")
    print("  Verificare finala CUDA...")
    print(f"{'='*60}")
    verify = subprocess.run(
        f'"{sys.executable}" -c "'
        f'import torch; '
        f'cuda_ok = torch.cuda.is_available(); '
        f'gpu = torch.cuda.get_device_name(0) if cuda_ok else \'N/A\'; '
        f'print(f\'CUDA: {{cuda_ok}} | GPU: {{gpu}}\')"',
        shell=True,
        capture_output=True,
        text=True
    )
    output = verify.stdout.strip()
    print(f"  Rezultat: {output}")

    if "True" in output and "RTX 5090" in output:
        print("\n" + "#"*60)
        print("#  INSTALARE COMPLETA!")
        print("#")
        print("#  Urmatorul pas:")
        print("#    1. Pune fisierele primite in C:\\MacroMicro\\")
        print("#    2. Deschide cmd in acel folder")
        print("#    3. Ruleaza: python train.py")
        print("#")
        print("#  Antrenamentul dureaza ~3-4 ore.")
        print("#  Poti lasa calculatorul pornit si sa te culci.")
        print("#"*60)
    elif "True" in output:
        print("\n[OK] CUDA functioneaza, dar GPU-ul detectat nu e RTX 5090.")
        print("Daca ai alt GPU NVIDIA, poate merge oricum. Trimite acest output fratelui tau.")
    else:
        print("\n[EROARE] CUDA nu functioneaza dupa instalare.")
        print("Incearca sa repornesti calculatorul si ruleaza din nou setup.py.")
        print("Daca tot nu merge, trimite aceasta eroare fratelui tau.")

if __name__ == "__main__":
    main()
