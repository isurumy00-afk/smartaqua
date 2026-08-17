/*
 * Arduino Uno Sensor Telemetry Node for Smart Aquarium Monitoring System
 *
 * Hardware Connections:
 *   - DS18B20 Temperature Sensor: Digital Pin 2 (with 4.7k pull-up resistor to 5V)
 *   - PH-4502C Analog pH Sensor: Analog Pin A0
 *   - Analog Turbidity Sensor: Analog Pin A1
 *   - USB Serial: Arduino Uno USB port connected to Raspberry Pi (/dev/ttyUSB1)
 *
 * Baud Rate: 9600
 * Output Format: Clean telemetry JSON transmitted once per second
 *   {"temp": 25.80, "ph": 7.20, "turbidity": 120.5}
 *
 * Water quality classification and stress decisions are computed by the ML
 * modules on Raspberry Pi.
 */

#include <DallasTemperature.h>
#include <OneWire.h>

#define PH_PIN A0
#define TURBIDITY_PIN A1
#define ONE_WIRE_BUS 2

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

void setup() {
  Serial.begin(9600);
  sensors.begin();
}

// Takes 10 analog readings, sorts them, and returns the average of the middle 6
// readings
float readAnalogSampled(int pin) {
  int buffer_arr[10];
  for (int i = 0; i < 10; i++) {
    buffer_arr[i] = analogRead(pin);
    delay(10);
  }

  // Bubble sort
  for (int i = 0; i < 9; i++) {
    for (int j = i + 1; j < 10; j++) {
      if (buffer_arr[i] > buffer_arr[j]) {
        int temp = buffer_arr[i];
        buffer_arr[i] = buffer_arr[j];
        buffer_arr[j] = temp;
      }
    }
  }

  // Average middle 6 samples
  unsigned long avgval = 0;
  for (int i = 2; i < 8; i++) {
    avgval += buffer_arr[i];
  }
  return (float)avgval / 6.0;
}

void loop() {
  // 1. Read DS18B20 Temperature
  sensors.requestTemperatures();
  float tempC = sensors.getTempCByIndex(0);

  // 2. Read pH Sensor (PH-4502C)
  float raw_ph_adc = readAnalogSampled(PH_PIN);
  float ph_volt = raw_ph_adc * 5.0 / 1023.0;
  float ph_act = 7.0 + ((2.50 - ph_volt) / 0.18);

  // 3. Read Turbidity Sensor
  float raw_turb_adc = readAnalogSampled(TURBIDITY_PIN);
  float turb_volt = raw_turb_adc * 5.0 / 1023.0;
  float turbidity_ntu =
      -1120.4 * (turb_volt * turb_volt) + 5742.3 * turb_volt - 4352.9;
  if (turbidity_ntu < 0)
    turbidity_ntu = 0;

  // Build and transmit the JSON payload atomically in a single Serial.println() call.
  // snprintf assembles the complete string into a stack buffer first — no mid-packet
  // interleaving possible. The buffer is sized for the worst-case string length:
  //   {"temp":xxx.xx,"ph":x.xx,"turbidity":xxxx.x}  → < 60 chars
  char buf[64];
  if (tempC != DEVICE_DISCONNECTED_C) {
    snprintf(buf, sizeof(buf),
             "{\"temp\":%.2f,\"ph\":%.2f,\"turbidity\":%.1f}",
             tempC, ph_act, turbidity_ntu);
  } else {
    snprintf(buf, sizeof(buf),
             "{\"temp\":null,\"ph\":%.2f,\"turbidity\":%.1f}",
             ph_act, turbidity_ntu);
  }
  Serial.println(buf);

  // Short delay: DS18B20 1-Wire conversion (~750ms) and analog sampling (~200ms)
  // already pace the loop to ~1.0 second transmission cadence.
  delay(50);
}
