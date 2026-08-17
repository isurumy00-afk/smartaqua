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

    def _create_client(self):
        """Create PyModbus serial client instance."""
        from pymodbus.client import ModbusSerialClient
        return ModbusSerialClient(
            port=self.port,
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
        try:
            client = self._create_client()
            if not client.connect():
                LOG.warning("Cannot connect to ion concentration Modbus device on %s", self.port)
                return {
                    "value": None,
                    "unit": "us/cm",
                    "timestamp": timestamp,
                    "error": f"Cannot connect to serial port {self.port}"
                }

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
                    LOG.warning("Modbus read error on ion concentration sensor (port: %s)", self.port)
                    return {
                        "value": None,
                        "unit": "us/cm",
                        "timestamp": timestamp,
                        "error": "Modbus Read Error"
                    }

                return {
                    "value": float(rr.registers[0]),
                    "unit": "us/cm",
                    "timestamp": timestamp,
                    "source": "modbus_rtu"
                }
            finally:
                client.close()

        except Exception as exc:
            LOG.warning("Ion concentration sensor read failed on %s: %s", self.port, exc)
            return {
                "value": None,
                "unit": "us/cm",
                "timestamp": timestamp,
                "error": str(exc)
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
