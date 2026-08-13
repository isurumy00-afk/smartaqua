"""Isolated MG90 Micro Servo Controller module for Automatic Feeding.

Provides hardware PWM angle control mapped to detected hungry fish count:
- 0 fish -> 0°
- 1 fish -> 20°
- 2 fish -> 35°
- 3 fish -> 50°
- 4 fish -> 65°

Features safety limits, manual override, daily feeding counters, and graceful fallback when offline.
"""

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

    def calculate_angle(self, hungry_count: int) -> int:
        """Calculate target servo angle for given hungry fish count."""
        if self.manual_override_angle is not None:
            return self.manual_override_angle

        count = max(0, min(int(hungry_count), len(self.config.feed_angles) - 1))
        target = self.config.feed_angles[count] + self.calibration_offset
        return max(self.config.minimum_angle, min(target, self.config.maximum_angle))

    def dispense(self, hungry_count: int) -> Dict[str, Any]:
        """Perform feeding cycle if daily limits are not exceeded.
        
        Returns status dictionary:
        {"angle": int, "dispensed": bool, "daily_count": int}
        """
        if self.daily_feed_count >= self.config.max_daily_feedings:
            LOG.warning("Maximum daily feeding limit reached (%d)", self.config.max_daily_feedings)
            return {
                "angle": 0,
                "dispensed": False,
                "reason": "Max daily feeding limit reached",
                "daily_count": self.daily_feed_count,
            }

        angle = self.calculate_angle(hungry_count)

        if hungry_count == 0 and self.manual_override_angle is None:
            return {
                "angle": 0,
                "dispensed": False,
                "reason": "No hungry fish detected",
                "daily_count": self.daily_feed_count,
            }

        # Hardware PWM actuation on Raspberry Pi GPIO pin
        hardware_actuated = self._actuate_hardware_pwm(angle)

        if angle > 0:
            self.daily_feed_count += 1

        return {
            "angle": angle,
            "dispensed": True,
            "hungry_count": hungry_count,
            "daily_count": self.daily_feed_count,
            "hardware_actuated": hardware_actuated,
        }

    def _actuate_hardware_pwm(self, angle: int) -> bool:
        """Isolated hardware actuation using RPi.GPIO or gpiozero."""
        try:
            import RPi.GPIO as GPIO
            import time

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.config.pin, GPIO.OUT)
            pwm = GPIO.PWM(self.config.pin, self.config.pwm_frequency)
            pwm.start(0)

            duty_cycle = 2.5 + (angle / 180.0) * 10.0
            pwm.ChangeDutyCycle(duty_cycle)
            time.sleep(0.5)
            pwm.stop()
            GPIO.cleanup(self.config.pin)
            return True
        except Exception as exc:
            LOG.debug("Hardware PWM actuation unconfigured or mock mode: %s", exc)
            return False

    def reset_daily_count(self) -> None:
        """Reset daily feeding counter at midnight."""
        self.daily_feed_count = 0
