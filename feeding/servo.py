"""Isolated MG90 Micro Servo Controller module for Automatic Feeding.

Provides multi-round CW & CCW hardware PWM angle control mapped to detected hungry fish count:
- 0 fish -> 0°
- 1 fish -> 20°
- 2 fish -> 35°
- 3 fish -> 50°
- 4 fish -> 65°

Each feeding cycle performs 2 rounds of Clockwise (CW) rotation to target angle and 
Counter-Clockwise (CCW) rotation back to 0° baseline to dispense food.

Features safety limits, manual override, daily feeding counters, and graceful fallback when offline.
"""

import time
from datetime import date, datetime, timezone
from typing import Optional, Dict, Any, Tuple
from config import SERVO, ServoConfig, DATA_DIR
from storage.json_store import load_json
from utils.logger import get_logger

LOG = get_logger(__name__)


class FeederServo:
    """Isolated hardware PWM controller for fish food dispensing servo."""

    def __init__(self, config: ServoConfig = SERVO):
        self.config = config
        self.manual_override_angle: Optional[int] = None
        self.calibration_offset: int = 0
        self.daily_feed_count: int = 0
        self.last_feed_date: date = date.today()
        self.last_dispense_time: Optional[float] = None
        self._gpio_pwm = None
        self._load_last_feed_state()

    def _load_last_feed_state(self) -> None:
        """Recover last dispense timestamp from persisted state across restarts."""
        try:
            feed_file = DATA_DIR / "latest_feed.json"
            if feed_file.exists():
                data = load_json(feed_file, {})
                if data and data.get("dispensed"):
                    if "last_dispense_time" in data:
                        t = float(data["last_dispense_time"])
                        if self.last_dispense_time is None or t > self.last_dispense_time:
                            self.last_dispense_time = t
                    elif "timestamp" in data:
                        dt = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
                        t = dt.timestamp()
                        if self.last_dispense_time is None or t > self.last_dispense_time:
                            self.last_dispense_time = t
        except Exception as exc:
            LOG.debug("Could not recover previous feed state: %s", exc)

    @property
    def cooldown_minutes(self) -> int:
        """Dynamically query post-feed cooldown minutes from config."""
        from config import load_config
        try:
            cfg = load_config()
            return int(cfg.get("servo", {}).get("post_feed_cooldown_minutes", getattr(self.config, "post_feed_cooldown_minutes", 30)))
        except Exception:
            return getattr(self.config, "post_feed_cooldown_minutes", 30)

    def is_in_cooldown(self) -> Tuple[bool, float]:
        """Check if feeder is currently in post-dispense cooldown.
        
        Returns:
            Tuple[bool, float]: (is_in_cooldown, remaining_seconds)
        """
        cooldown_mins = self.cooldown_minutes
        if cooldown_mins <= 0:
            return False, 0.0

        self._load_last_feed_state()
        if self.last_dispense_time is None:
            return False, 0.0

        elapsed = time.time() - self.last_dispense_time
        cooldown_seconds = cooldown_mins * 60.0
        if elapsed < cooldown_seconds:
            remaining = max(0.0, cooldown_seconds - elapsed)
            return True, remaining

        return False, 0.0

    def set_calibration_offset(self, offset_degrees: int) -> None:
        """Set calibration offset angle in degrees."""
        self.calibration_offset = offset_degrees
        LOG.info("Servo calibration offset set to %d°", offset_degrees)

    def set_manual_override(self, angle: Optional[int]) -> None:
        """Set or clear manual override angle."""
        if angle is not None:
            clamped = max(self.config.minimum_angle, min(angle, self.config.maximum_angle))
            self.manual_override_angle = clamped
            LOG.info("Manual servo override set to %d°", clamped)
        else:
            self.manual_override_angle = None
            LOG.info("Manual servo override cleared.")

    def calculate_rounds(self, hungry_count: int) -> int:
        """Calculate number of rotation rounds based on hungry fish count."""
        if hungry_count <= 0:
            return 0
        # Number of rounds equals number of hungry fish detected
        return max(1, int(hungry_count))

    def calculate_angle(self, hungry_count: int) -> int:
        """Return full sweep angle (180° CW) for servo actuation."""
        if self.manual_override_angle is not None:
            return self.manual_override_angle
        return self.config.maximum_angle

    def dispense(self, hungry_count: int, is_automatic: bool = True) -> Dict[str, Any]:
        """Perform feeding cycle based on hungry fish count.
        
        Rotates full CW (0° to 180°) and CCW (180° to 0°).
        Number of rounds performed equals the number of hungry fish detected.
        
        Returns status dictionary:
        {"angle": int, "rounds": int, "dispensed": bool, "daily_count": int, "cooldown_minutes": int}
        """
        # Auto-reset daily feeding counter at midnight
        today = date.today()
        if self.last_feed_date < today:
            self.daily_feed_count = 0
            self.last_feed_date = today

        if self.daily_feed_count >= self.config.max_daily_feedings:
            LOG.warning("Maximum daily feeding limit reached (%d)", self.config.max_daily_feedings)
            return {
                "angle": 0,
                "rounds": 0,
                "dispensed": False,
                "reason": "Max daily feeding limit reached",
                "daily_count": self.daily_feed_count,
            }

        rounds = self.calculate_rounds(hungry_count)
        angle = self.calculate_angle(hungry_count)

        if rounds == 0 and self.manual_override_angle is None:
            return {
                "angle": 0,
                "rounds": 0,
                "dispensed": False,
                "reason": "No hungry fish detected",
                "daily_count": self.daily_feed_count,
            }

        # Manual feed trigger fallback (if hungry_count is 0 but manual override active)
        if rounds == 0 and self.manual_override_angle is not None:
            rounds = 1

        # Hardware PWM actuation on Raspberry Pi GPIO pin (CW & CCW full rotation rounds)
        hardware_actuated = self._actuate_hardware_pwm(angle=angle, rounds=rounds)

        now_time = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()
        if rounds > 0:
            self.daily_feed_count += 1
            self.last_dispense_time = now_time

        result = {
            "angle": angle,
            "rounds": rounds,
            "dispensed": True,
            "hungry_count": hungry_count,
            "daily_count": self.daily_feed_count,
            "hardware_actuated": hardware_actuated,
            "is_automatic": is_automatic,
            "last_dispense_time": now_time,
            "last_dispense_timestamp": now_iso,
            "cooldown_minutes": self.cooldown_minutes,
        }

        try:
            from storage.json_store import save_json
            save_json(DATA_DIR / "latest_feed.json", result)
        except Exception:
            pass

        return result

    def _actuate_hardware_pwm(self, angle: int = 180, rounds: int = 1) -> bool:
        """Isolated hardware PWM actuation for MG90 servo on Raspberry Pi 4B.
        
        Supports multiple backends compatible with Debian 13 (Trixie) & Python 3.13:
        1. RPi.GPIO / rpi-lgpio
        2. gpiozero (AngularServo / PWMOutputDevice)
        3. lgpio direct
        4. Mock fallback when hardware is unavailable.
        """
        if rounds <= 0:
            return False

        import time

        # Backend 1: RPi.GPIO / rpi-lgpio
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.config.pin, GPIO.OUT)
            pwm = GPIO.PWM(self.config.pin, self.config.pwm_frequency)
            pwm.start(0)

            duty_home = 2.5  # 0° baseline (CCW)
            duty_target = 2.5 + (angle / 180.0) * 10.0  # target angle (CW)

            for round_idx in range(rounds):
                LOG.info("Executing feeding round %d/%d (Full CW 0°->%d° & CCW ->0°)", round_idx + 1, rounds, angle)
                pwm.ChangeDutyCycle(duty_target)
                time.sleep(0.5)
                pwm.ChangeDutyCycle(duty_home)
                time.sleep(0.5)

            pwm.ChangeDutyCycle(0)
            time.sleep(0.1)
            pwm.stop()
            try:
                GPIO.cleanup(self.config.pin)
            except Exception:
                pass
            return True
        except Exception as exc_gpio:
            LOG.debug("RPi.GPIO / rpi-lgpio backend unavailable: %s", exc_gpio)

        # Backend 2: gpiozero AngularServo / PWMOutputDevice (Standard on Debian 13)
        try:
            from gpiozero import AngularServo
            servo = AngularServo(
                self.config.pin,
                min_angle=0,
                max_angle=180,
                min_pulse_width=0.0005,
                max_pulse_width=0.0025,
            )
            for round_idx in range(rounds):
                LOG.info("[gpiozero] Executing feeding round %d/%d (0°->%d°->0°)", round_idx + 1, rounds, angle)
                servo.angle = angle
                time.sleep(0.5)
                servo.angle = 0
                time.sleep(0.5)
            servo.detach()
            servo.close()
            return True
        except Exception as exc_gz:
            LOG.debug("gpiozero backend unavailable: %s", exc_gz)

        # Backend 3: lgpio direct
        try:
            import lgpio
            h = lgpio.gpiochip_open(0)
            duty_home_pct = (2.5 / 100.0) * 100.0
            duty_target_pct = (2.5 + (angle / 180.0) * 10.0)
            for round_idx in range(rounds):
                LOG.info("[lgpio] Executing feeding round %d/%d", round_idx + 1, rounds)
                lgpio.tx_pwm(h, self.config.pin, self.config.pwm_frequency, duty_target_pct)
                time.sleep(0.5)
                lgpio.tx_pwm(h, self.config.pin, self.config.pwm_frequency, duty_home_pct)
                time.sleep(0.5)
            lgpio.tx_pwm(h, self.config.pin, self.config.pwm_frequency, 0)
            lgpio.gpiochip_close(h)
            return True
        except Exception as exc_lg:
            LOG.debug("lgpio backend unavailable: %s", exc_lg)

        LOG.info("Hardware PWM simulated (mock mode): %d rounds of 0°->%d° completed.", rounds, angle)
        return False

    def reset_daily_count(self) -> None:
        """Reset daily feeding counter at midnight."""
        self.daily_feed_count = 0
