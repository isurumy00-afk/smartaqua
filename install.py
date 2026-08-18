#!/usr/bin/env python3
"""Installation Script for Smart Aquarium Monitoring System on Raspberry Pi 4B.

Automates system package installation, Python virtual environment setup,
dependency installation, directory structure creation, and hardware interface checks.
"""

import sys
import os
import platform
import subprocess
import argparse
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"

# ANSI Color codes for clean output
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"


def log_info(msg: str):
    print(f"{COLOR_BLUE}[INFO]{COLOR_RESET} {msg}")


def log_success(msg: str):
    print(f"{COLOR_GREEN}[OK]{COLOR_RESET} {msg}")


def log_warn(msg: str):
    print(f"{COLOR_YELLOW}[WARN]{COLOR_RESET} {msg}")


def log_error(msg: str):
    print(f"{COLOR_RED}[FAIL]{COLOR_RESET} {msg}")


def check_python_version():
    """Ensure Python version is 3.9+."""
    log_info("Checking Python version...")
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 9):
        log_warn(f"Python version {major}.{minor} detected. Python 3.9+ is recommended for Pi 4B.")
    else:
        log_success(f"Python version {major}.{minor}.{sys.version_info.micro} verified.")


def check_platform():
    """Detect OS and hardware architecture."""
    log_info("Checking OS and Hardware Architecture...")
    system = platform.system()
    machine = platform.machine()
    log_info(f"System: {system}, Architecture: {machine}")
    is_pi = system == "Linux" and (machine in ("armv7l", "aarch64") or Path("/etc/rpi-issue").exists())
    if is_pi:
        log_success("Raspberry Pi hardware environment detected.")
    else:
        log_warn("Non-Raspberry Pi environment detected. Hardware-specific drivers (RPi.GPIO) will be skipped/mocked.")
    return is_pi, machine


def create_directories():
    """Create all required runtime and model directories."""
    log_info("Creating system directory structure...")
    dirs = [
        DATA_DIR,
        LOG_DIR,
        MODELS_DIR / "vision",
        MODELS_DIR / "disease",
        MODELS_DIR / "feeding",
        MODELS_DIR / "water_quality",
        MODELS_DIR / "NLP",
    ]
    for directory in dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            log_success(f"Directory ready: {directory.relative_to(BASE_DIR)}")
        except Exception as exc:
            log_error(f"Failed to create directory {directory}: {exc}")


def install_apt_packages(dry_run: bool = False):
    """Install system packages using apt-get on Linux/Pi OS."""
    if platform.system() != "Linux":
        log_info("Skipping apt-get package installation (Non-Linux system).")
        return

    packages = [
        "python3-pip",
        "python3-dev",
        "python3-venv",
        "python3-opencv",
        "libgfortran5",
        "libopenblas-dev",
        "gpiod",
        "i2c-tools",
        "swig",
        "liblgpio-dev",
        "python3-lgpio",
        "python3-rpi-lgpio",
    ]

    log_info("Updating apt package index and installing system dependencies...")
    cmd_update = ["sudo", "apt-get", "update", "-y"]
    cmd_install = ["sudo", "apt-get", "install", "-y"] + packages

    if dry_run:
        log_info(f"[Dry Run] Would execute: {' '.join(cmd_update)}")
        log_info(f"[Dry Run] Would execute: {' '.join(cmd_install)}")
        return

    try:
        subprocess.run(cmd_update, check=True)
        # Attempt installation of all packages; if some optional Pi packages aren't in repo, install core
        res = subprocess.run(cmd_install)
        if res.returncode != 0:
            log_warn("Retrying with essential packages only...")
            fallback_packages = [
                "python3-pip", "python3-dev", "python3-venv", "python3-opencv",
                "libgfortran5", "libopenblas-dev", "gpiod", "i2c-tools", "swig", "liblgpio-dev"
            ]
            subprocess.run(["sudo", "apt-get", "install", "-y"] + fallback_packages, check=True)
        log_success("System packages installed successfully via apt-get.")
    except Exception as exc:
        log_warn(f"Apt package installation warning: {exc}. Ensure you have sudo privileges.")


def check_in_venv():
    """Check if running inside a virtual environment."""
    return sys.prefix != sys.base_prefix


def get_target_python() -> str:
    """Get the path to the virtual environment's Python executable if present."""
    if os.name == "nt":
        venv_py = BASE_DIR / "venv" / "Scripts" / "python.exe"
    else:
        venv_py = BASE_DIR / "venv" / "bin" / "python"

    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def setup_venv(dry_run: bool = False):
    """Ensure a virtual environment exists or is created."""
    venv_dir = BASE_DIR / "venv"
    if check_in_venv():
        log_success(f"Currently active inside virtual environment: {sys.prefix}")
        return True

    log_info("Checking virtual environment status...")
    if not venv_dir.exists():
        log_info(f"Creating Python virtual environment at {venv_dir} (with --system-site-packages)...")
        if dry_run:
            log_info(f"[Dry Run] Would create venv at {venv_dir}")
            return False
        try:
            subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)], check=True)
            log_success(f"Virtual environment created at {venv_dir}.")
            log_info(f"To activate, run: source {venv_dir}/bin/activate")
        except Exception as exc:
            log_error(f"Failed to create venv: {exc}")
            return False
    else:
        log_success(f"Virtual environment directory exists at {venv_dir}.")
    return True


def install_python_dependencies(dry_run: bool = False):
    """Install Python packages from requirements.txt into the active or target venv."""
    if not REQUIREMENTS_FILE.exists():
        log_error(f"Requirements file not found at {REQUIREMENTS_FILE}")
        return

    target_py = get_target_python()
    log_info(f"Installing Python dependencies using {target_py} from requirements.txt...")
    pip_cmd = [target_py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
    req_cmd = [target_py, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)]

    if dry_run:
        log_info(f"[Dry Run] Would execute: {' '.join(pip_cmd)}")
        log_info(f"[Dry Run] Would execute: {' '.join(req_cmd)}")
        return

    try:
        log_info("Upgrading pip, setuptools, wheel...")
        subprocess.run(pip_cmd, check=True)
        log_info("Installing requirements...")
        subprocess.run(req_cmd, check=True)
        log_success("All Python dependencies installed successfully.")
    except Exception as exc:
        log_error(f"Failed to install Python dependencies: {exc}")


def check_hardware_config():
    """Verify Raspberry Pi 1-Wire and Hardware overlays."""
    if platform.system() != "Linux":
        return

    log_info("Auditing Raspberry Pi hardware overlays...")
    config_paths = [Path("/boot/firmware/config.txt"), Path("/boot/config.txt")]
    found_config = None
    for p in config_paths:
        if p.exists():
            found_config = p
            break

    if found_config:
        try:
            content = found_config.read_text()
            if "dtoverlay=w1-gpio" in content:
                log_success(f"1-Wire overlay (dtoverlay=w1-gpio) enabled in {found_config}")
            else:
                log_warn(f"1-Wire overlay not found in {found_config}.")
                log_info("To enable DS18B20 1-Wire temperature sensor, add 'dtoverlay=w1-gpio' to /boot/firmware/config.txt and reboot.")
        except Exception as exc:
            log_warn(f"Could not read {found_config}: {exc}")
    else:
        log_info("No /boot/config.txt found (Non-Pi OS or standard Linux distro).")


def main():
    parser = argparse.ArgumentParser(description="Install pre-requisites for Smart Aquarium Monitoring System.")
    parser.add_argument("--skip-apt", action="store_true", help="Skip system apt-get package installation")
    parser.add_argument("--skip-pip", action="store_true", help="Skip Python pip dependency installation")
    parser.add_argument("--dry-run", action="store_true", help="Show setup steps without executing commands")
    args = parser.parse_args()

    print("==========================================================================")
    print(" Smart Aquarium Monitoring System - Pre-requisite Setup Installer")
    print(" Target Hardware: Raspberry Pi 4B (2GB RAM) / Linux")
    print("==========================================================================")

    check_python_version()
    is_pi, machine = check_platform()
    create_directories()

    if not args.skip_apt and is_pi:
        install_apt_packages(dry_run=args.dry_run)

    setup_venv(dry_run=args.dry_run)

    if not args.skip_pip:
        install_python_dependencies(dry_run=args.dry_run)

    check_hardware_config()

    print("\n==========================================================================")
    log_success("Pre-requisite installation and environment setup completed!")
    print("Next Steps:")
    print("  1. Run diagnostics: python diagnose.py")
    print("  2. Start master service: python master.py")
    print("==========================================================================")


if __name__ == "__main__":
    main()
