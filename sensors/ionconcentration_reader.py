"""Ion Concentration Modbus RTU Serial Sensor Reader.

Reads water ion concentration over RS485 Modbus RTU interface via USB serial.
Configuration is driven by central SENSOR_CONFIG to support multi-USB device systems.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from config import SENSOR_CONFIG
from utils.logger import get_logger

LOG = get_logger(__name__)


class IonConcentrationReader:
    """Modbus serial client reader for water ion concentration sensor."""

    def __init__(self, port: Optional[str] = None):
        self.port = port or SENSOR_CONFIG.get("ionconcentration_serial_port", "/dev/ttyUSB0")
        self.baudrate = SENSOR_CONFIG.get("ionconcentration_baudrate", 9600)
        self.bytesize = SENSOR_CONFIG.get("ionconcentration_bytesize", 8)
        self.parity = SENSOR_CONFIG.get("ionconcentration_parity", "N")
        self.stopbits = SENSOR_CONFIG.get("ionconcentration_stopbits", 1)
        self.timeout = SENSOR_CONFIG.get("ionconcentration_timeout", 1)
        self.device_id = SENSOR_CONFIG.get("ionconcentration_device_id", 1)
        self.address = SENSOR_CONFIG.get("ionconcentration_address", 20)

    def _get_candidate_ports(self) -> list:
        candidates = [self.port]
        import os
        from pathlib import Path
        if os.name != "nt":
            for p in ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0", "/dev/ttyACM1"]:
                if p not in candidates and Path(p).exists():
                    candidates.append(p)
        return candidates

    def _create_client(self, port: str):
        """Create PyModbus serial client instance."""
        from pymodbus.client import ModbusSerialClient
        return ModbusSerialClient(
            port=port,
            baudrate=self.baudrate,
            bytesize=self.bytesize,
            parity=self.parity,
            stopbits=self.stopbits,
            timeout=self.timeout
        )

    def read(self) -> Dict[str, Any]:
        """Read ion concentration holding register from Modbus device.

        Returns standard format:
        {"value": float | None, "unit": "us/cm", "timestamp": str, "source": "modbus_rtu"}
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        candidates = self._get_candidate_ports()

        last_error = "Cannot connect to serial port"
        for p in candidates:
            try:
                client = self._create_client(p)
                if not client.connect():
                    continue

                try:
                    try:
                        rr = client.read_holding_registers(
                            address=self.address,
                            count=1,
                            device_id=self.device_id
                        )
                    except TypeError:
                        # Fallback for PyModbus versions using slave keyword argument
                        rr = client.read_holding_registers(
                            address=self.address,
                            count=1,
                            slave=self.device_id
                        )

                    if rr is None or rr.isError():
                        last_error = "Modbus Read Error"
                        continue

                    return {
                        "value": float(rr.registers[0]),
                        "unit": "us/cm",
                        "timestamp": timestamp,
                        "source": "modbus_rtu"
                    }
                finally:
                    client.close()

            except Exception as exc:
                last_error = str(exc)
                LOG.debug("Modbus attempt failed on %s: %s", p, exc)

        return {
            "value": None,
            "unit": "us/cm",
            "timestamp": timestamp,
            "error": last_error
        }


def read() -> Dict[str, Any]:
    """Module-level helper to perform one-shot ion concentration reading."""
    return IonConcentrationReader().read()


if __name__ == "__main__":
    port = SENSOR_CONFIG.get("ionconcentration_serial_port", "/dev/ttyUSB0")
    baudrate = SENSOR_CONFIG.get("ionconcentration_baudrate", 9600)
    bytesize = SENSOR_CONFIG.get("ionconcentration_bytesize", 8)
    parity = SENSOR_CONFIG.get("ionconcentration_parity", "N")
    stopbits = SENSOR_CONFIG.get("ionconcentration_stopbits", 1)
    timeout = SENSOR_CONFIG.get("ionconcentration_timeout", 1)
    device_id = SENSOR_CONFIG.get("ionconcentration_device_id", 1)
    address = SENSOR_CONFIG.get("ionconcentration_address", 20)

    from pymodbus.client import ModbusSerialClient
    client = ModbusSerialClient(
        port=port, baudrate=baudrate, bytesize=bytesize,
        parity=parity, stopbits=stopbits, timeout=timeout
    )

    if not client.connect():
        print("Cannot connect")
        raise SystemExit(1)

    try:
        while True:
            try:
                rr = client.read_holding_registers(address=address, count=1, device_id=device_id)
            except TypeError:
                rr = client.read_holding_registers(address=address, count=1, slave=device_id)

            if rr is None or rr.isError():
                print("Read Error")
            else:
                print("ionconcentration =", rr.registers[0], "us/cm")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping reader.")
    finally:
        client.close()
