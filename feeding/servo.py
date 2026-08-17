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

from datetime import date
from typing import Optional, Dict, Any
from config import SERVO, ServoConfig
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
        self._gpio_pwm = None

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

    def dispense(self, hungry_count: int) -> Dict[str, Any]:
        """Perform feeding cycle based on hungry fish count.
        
        Rotates full CW (0° to 180°) and CCW (180° to 0°).
        Number of rounds performed equals the number of hungry fish detected.
        
        Returns status dictionary:
        {"angle": int, "rounds": int, "dispensed": bool, "daily_count": int}
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

        if rounds > 0:
            self.daily_feed_count += 1

        return {
            "angle": angle,
            "rounds": rounds,
            "dispensed": True,
            "hungry_count": hungry_count,
            "daily_count": self.daily_feed_count,
            "hardware_actuated": hardware_actuated,
        }

    def _actuate_hardware_pwm(self, angle: int = 180, rounds: int = 1) -> bool:
        """Isolated hardware PWM actuation using RPi.GPIO.
        
        Performs full CW rotation (0° to target angle, default 180°) and 
        CCW rotation back to 0° for the specified number of rounds (1 round per hungry fish).
        Sets zero duty-cycle before stopping to prevent analog servo buzzing/jitter on 5V rail.
        """
        if rounds <= 0:
            return False

        try:
            import RPi.GPIO as GPIO
            import time

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.config.pin, GPIO.OUT)
            pwm = GPIO.PWM(self.config.pin, self.config.pwm_frequency)
            pwm.start(0)

            duty_home = 2.5  # 0° baseline position (full CCW)
            duty_target = 2.5 + (angle / 180.0) * 10.0  # target angle position (180° full CW)

            for round_idx in range(rounds):
                LOG.info("Executing feeding round %d/%d (Full CW 0°->%d° & CCW ->0°)", round_idx + 1, rounds, angle)
                # Full Clockwise (CW) rotation
                pwm.ChangeDutyCycle(duty_target)
                time.sleep(0.5)
                # Full Counter-Clockwise (CCW) rotation back to 0° baseline
                pwm.ChangeDutyCycle(duty_home)
                time.sleep(0.5)

            # Eliminate PWM pulse holding jitter / continuous buzzing
            pwm.ChangeDutyCycle(0)
            time.sleep(0.1)

            pwm.stop()
            GPIO.cleanup(self.config.pin)
            return True
        except Exception as exc:
            LOG.debug("Hardware PWM actuation unconfigured or mock mode: %s", exc)
            return False

    def reset_daily_count(self) -> None:
        """Reset daily feeding counter at midnight."""
        self.daily_feed_count = 0
