/*
 * WHEELCHAIR CONTROLLER v3.0
 * ===========================
 *
 * Based on v2 "industry grade" but with critical odometry fix:
 * - Left encoder publishes physical forward as positive (no double negation).
 * - Keeps PID/motor sign handling separate from what is sent to ROS.
 *
 * Other features retained:
 * - Atomic encoder reads
 * - Watchdog timer (2s)
 * - Safe parsing buffers (no heap allocation in loop)
 * - deltaT validation
 * - Clean serial protocol for ROS parsing
 */

#include <PID_v1.h>
#include <Cytron_SmartDriveDuo.h>
#include <avr/wdt.h>

// ================== PPM CONFIG ==================
#define PPM_PIN 20
#define CHANNELS 8

volatile uint16_t ppmValues[CHANNELS];
volatile uint8_t ppmIndex = 0;
volatile unsigned long lastMicros = 0;

int ch1, ch2;
bool ppm_active = false;
unsigned long last_ppm_time = 0;
const unsigned long PPM_TIMEOUT = 500;

// ================== PIN DEFINITIONS ==================
const int encoderPin1R = 18;
const int encoderPin2R = 19;
const int encoderPin1L = 3;
const int encoderPin2L = 2;
const int relayPin = 53;

#define IN1 7
#define AN1 5
#define AN2 4
#define IN2 6

Cytron_SmartDriveDuo smartDriveDuo(PWM_INDEPENDENT, IN1, IN2, AN1, AN2);

// ================== ENCODER VARIABLES ==================
volatile int lastEncodedL = 0;
volatile long encoderValueL = 0;
volatile int lastEncodedR = 0;
volatile long encoderValueR = 0;

// ================== WHEEL SPECIFICATIONS ==================
const long CPR = 10000;
const float WHEEL_RADIUS = 0.1524f;  // 6 inches
const float WHEELBASE = 0.565f;       // meters
const float MAX_ANGULAR_VEL = 17.59f;
const float MAX_LINEAR_VEL = 2.68f;

// PPM velocity limits (safety)
const float PPM_MAX_LINEAR_VEL = 1.0f;
const float PPM_MAX_ANGULAR_VEL = 1.0f;

// ================== TIMING ==================
unsigned long last_millis = 0;
const unsigned long CONTROL_INTERVAL = 50;  // 20 Hz
const float MIN_DELTA_T = 0.001f;
const float MAX_DELTA_T = 0.5f;
long lastEncoderL = 0;
long lastEncoderR = 0;

// ================== CASTER SWIVEL PREVENTION ==================
bool pivot_mode = false;
unsigned long pivot_start_time = 0;
const unsigned long PIVOT_DURATION = 300;
const float PIVOT_ANGULAR_VEL = 2.0f;
float prev_linear_velocity = 0.0f;
float stored_linear_cmd = 0.0f;
float stored_angular_cmd = 0.0f;

// ================== COMMAND MODE ==================
enum CommandMode { WHEEL_MODE, CMDVEL_MODE, PPM_MODE };
CommandMode current_mode = WHEEL_MODE;

// ================== COMMAND PARSING ==================
char value_buffer[12];
uint8_t value_idx = 0;
const uint8_t VALUE_BUFFER_MAX = 10;

bool is_right_wheel_cmd = false;
bool is_left_wheel_cmd = false;
bool is_linear_cmd = false;
bool is_angular_cmd = false;
bool is_right_wheel_forward = true;
bool is_left_wheel_forward = true;

// ================== VELOCITY VARIABLES ==================
double right_wheel_cmd_vel = 0.0;
double left_wheel_cmd_vel = 0.0;
double right_wheel_target = 0.0;
double left_wheel_target = 0.0;
double right_wheel_meas_vel = 0.0;
double left_wheel_meas_vel = 0.0;
double right_wheel_cmd = 0.0;
double left_wheel_cmd = 0.0;

// cmd_vel variables
double linear_x = 0.0;
double angular_z = 0.0;

// ================== FILTERING & ACCELERATION ==================
const double MAX_ACCEL = 5.0;
double filtered_right_vel = 0.0;
double filtered_left_vel = 0.0;
const double FILTER_ALPHA = 0.8;

// ================== PID GAINS ==================
double Kp_r = 7.0, Ki_r = 8.0, Kd_r = 0.15;
double Kp_l = 7.2, Ki_l = 8.5, Kd_l = 0.15;

// Manual PID state
double right_integral = 0.0, left_integral = 0.0;
double right_prev_error = 0.0, left_prev_error = 0.0;
double right_prev_meas = 0.0, left_prev_meas = 0.0;

// ================== ATOMIC ENCODER READ ==================
void readEncodersAtomic(long &leftVal, long &rightVal) {
  noInterrupts();
  leftVal = encoderValueL;
  rightVal = encoderValueR;
  interrupts();
}

// ================== CASTER SWIVEL PREVENTION ==================
bool needsPivot(float current_linear, float prev_linear) {
  return (prev_linear > 0.1f && current_linear < -0.1f) ||
         (prev_linear < -0.1f && current_linear > 0.1f);
}

void startPivot(float linear_cmd, float angular_cmd) {
  pivot_mode = true;
  pivot_start_time = millis();
  stored_linear_cmd = linear_cmd;
  stored_angular_cmd = angular_cmd;

  float pivot_angular = (angular_cmd != 0.0f)
                            ? (angular_cmd > 0.0f ? PIVOT_ANGULAR_VEL : -PIVOT_ANGULAR_VEL)
                            : PIVOT_ANGULAR_VEL;
  cmdVelToWheels(0.0, pivot_angular);
}

bool updatePivot() {
  if (!pivot_mode) return false;
  if (millis() - pivot_start_time >= PIVOT_DURATION) {
    pivot_mode = false;
    cmdVelToWheels(stored_linear_cmd, stored_angular_cmd);
    prev_linear_velocity = stored_linear_cmd;
    return false;
  }
  return true;
}

// ================== PPM INTERRUPT ==================
void ppmISR() {
  unsigned long now = micros();
  unsigned long diff = now - lastMicros;
  lastMicros = now;

  if (diff > 3200) {
    ppmIndex = 0;
  } else if (ppmIndex < CHANNELS) {
    ppmValues[ppmIndex] = diff;
    ppmIndex++;
  }
}

// ================== PPM PROCESSING ==================
void processPPMInput() {
  unsigned long current_time = millis();
  ch1 = ppmValues[0];
  ch2 = ppmValues[1];

  if (ch1 > 900 && ch1 < 2100 && ch2 > 900 && ch2 < 2100) {
    ppm_active = true;
    last_ppm_time = current_time;
    current_mode = PPM_MODE;

    int ch1_filtered = ch1;
    int ch2_filtered = ch2;
    if (ch1 > 1460 && ch1 < 1540) ch1_filtered = 1500;
    if (ch2 > 1460 && ch2 < 1540) ch2_filtered = 1500;
    ch1_filtered = constrain(ch1_filtered, 1000, 2000);
    ch2_filtered = constrain(ch2_filtered, 1000, 2000);

    float linear_velocity = ((float)(ch2_filtered - 1500) / 500.0f) * PPM_MAX_LINEAR_VEL;
    linear_velocity = constrain(linear_velocity, -PPM_MAX_LINEAR_VEL, PPM_MAX_LINEAR_VEL);

    float angular_velocity = ((float)(1500 - ch1_filtered) / 500.0f) * PPM_MAX_ANGULAR_VEL;
    angular_velocity = constrain(angular_velocity, -PPM_MAX_ANGULAR_VEL, PPM_MAX_ANGULAR_VEL);

    if (needsPivot(linear_velocity, prev_linear_velocity) && !pivot_mode) {
      startPivot(linear_velocity, angular_velocity);
      return;
    }
    if (updatePivot()) return;

    float angular_component = (angular_velocity * WHEELBASE / 2.0f) / WHEEL_RADIUS;
    float linear_component = linear_velocity / WHEEL_RADIUS;
    float left_wheel_vel = linear_component - angular_component;
    float right_wheel_vel = linear_component + angular_component;
    left_wheel_vel = constrain(left_wheel_vel, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL);
    right_wheel_vel = constrain(right_wheel_vel, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL);

    if ((right_wheel_cmd_vel * right_wheel_vel < 0) || (abs(right_wheel_vel - right_wheel_cmd_vel) > 3.0)) {
      right_integral = 0.0;
      right_wheel_target = 0.0;
    }
    if ((left_wheel_cmd_vel * left_wheel_vel < 0) || (abs(left_wheel_vel - left_wheel_cmd_vel) > 3.0)) {
      left_integral = 0.0;
      left_wheel_target = 0.0;
    }

    right_wheel_cmd_vel = right_wheel_vel;
    left_wheel_cmd_vel = left_wheel_vel;
    prev_linear_velocity = linear_velocity;
  } else if (ppm_active && (current_time - last_ppm_time > PPM_TIMEOUT)) {
    ppm_active = false;
    pivot_mode = false;
    right_wheel_cmd_vel = 0.0;
    left_wheel_cmd_vel = 0.0;
    prev_linear_velocity = 0.0;
    Serial.println("PPM signal lost - stopping");
  }
}

// ================== ENCODER ISRs ==================
void updateEncoderL() {
  int MSB = digitalRead(encoderPin1L);
  int LSB = digitalRead(encoderPin2L);
  int encoded = (MSB << 1) | LSB;
  int sum = (lastEncodedL << 2) | encoded;
  if (sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011) encoderValueL++;
  if (sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000) encoderValueL--;
  lastEncodedL = encoded;
}

void updateEncoderR() {
  int MSB = digitalRead(encoderPin1R);
  int LSB = digitalRead(encoderPin2R);
  int encoded = (MSB << 1) | LSB;
  int sum = (lastEncodedR << 2) | encoded;
  if (sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011) encoderValueR++;
  if (sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000) encoderValueR--;
  lastEncodedR = encoded;
}

// ================== KINEMATICS ==================
void cmdVelToWheels(double linear, double angular) {
  if (needsPivot(linear, prev_linear_velocity) && !pivot_mode) {
    startPivot(linear, angular);
    return;
  }
  double right_vel = (linear + angular * WHEELBASE / 2.0) / WHEEL_RADIUS;
  double left_vel = (linear - angular * WHEELBASE / 2.0) / WHEEL_RADIUS;
  right_wheel_cmd_vel = constrain(right_vel, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL);
  left_wheel_cmd_vel = constrain(left_vel, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL);
  right_integral = 0.0;
  left_integral = 0.0;
  right_wheel_target = 0.0;
  left_wheel_target = 0.0;
  prev_linear_velocity = linear;
}

// ================== PID CONTROLLER ==================
double computePID(double setpoint, double measurement, double &integral,
                  double &prev_error, double &prev_meas,
                  double Kp, double Ki, double Kd, double dt) {
  double error = setpoint - measurement;
  integral += error * dt;
  integral = constrain(integral, -100.0 / Ki, 100.0 / Ki);
  double derivative = -(measurement - prev_meas) / dt;
  double output = Kp * error + Ki * integral + Kd * derivative;
  double clamped_output = constrain(output, -100.0, 100.0);
  if (output != clamped_output) {
    integral = (clamped_output - Kp * error - Kd * derivative) / Ki;
  }
  prev_error = error;
  prev_meas = measurement;
  return clamped_output;
}

// ================== ACCELERATION LIMITER ==================
double rampVelocity(double current, double target, double max_accel, double dt) {
  double diff = target - current;
  double max_change = max_accel * dt;
  if (abs(diff) <= max_change) return target;
  return current + (diff > 0 ? max_change : -max_change);
}

// ================== VALUE BUFFER HELPERS ==================
void resetValueBuffer() {
  value_idx = 0;
  value_buffer[0] = '\0';
}

double parseValueBuffer() {
  value_buffer[value_idx] = '\0';
  return atof(value_buffer);
}

// ================== SETUP ==================
void setup() {
  Serial.begin(115200);
  wdt_enable(WDTO_2S);

  pinMode(PPM_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(PPM_PIN), ppmISR, RISING);

  pinMode(relayPin, OUTPUT);
  digitalWrite(relayPin, HIGH);
  for (int i = 0; i < 50; i++) {  // 5s wait with watchdog resets
    wdt_reset();
    delay(100);
  }
  digitalWrite(relayPin, LOW);

  pinMode(encoderPin1L, INPUT_PULLUP);
  pinMode(encoderPin2L, INPUT_PULLUP);
  pinMode(encoderPin1R, INPUT_PULLUP);
  pinMode(encoderPin2R, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(encoderPin1L), updateEncoderL, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPin2L), updateEncoderL, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPin1R), updateEncoderR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPin2R), updateEncoderR, CHANGE);

  filtered_right_vel = 0.0;
  filtered_left_vel = 0.0;
  last_millis = millis();
  resetValueBuffer();

  Serial.println("Wheelchair Controller v3.0 Ready");
}

// ================== MAIN LOOP ==================
void loop() {
  wdt_reset();
  processPPMInput();
  if (!ppm_active && pivot_mode) updatePivot();

  if (!ppm_active && Serial.available()) {
    char chr = Serial.read();
    if (chr == 'r' || chr == 'l') current_mode = WHEEL_MODE;
    else if (chr == 'x' || chr == 't') current_mode = CMDVEL_MODE;

    if (current_mode == WHEEL_MODE) {
      if (chr == 'r') { is_right_wheel_cmd = true; is_left_wheel_cmd = false; resetValueBuffer(); }
      else if (chr == 'l') { is_right_wheel_cmd = false; is_left_wheel_cmd = true; resetValueBuffer(); }
      else if (chr == 'p') { if (is_right_wheel_cmd) is_right_wheel_forward = true; else if (is_left_wheel_cmd) is_left_wheel_forward = true; }
      else if (chr == 'n') { if (is_right_wheel_cmd) is_right_wheel_forward = false; else if (is_left_wheel_cmd) is_left_wheel_forward = false; }
      else if (chr == ',') {
        double new_cmd = parseValueBuffer();
        if (is_right_wheel_cmd) {
          if (!is_right_wheel_forward) new_cmd = -new_cmd;
          if ((right_wheel_cmd_vel * new_cmd < 0) || (abs(new_cmd - right_wheel_cmd_vel) > 3.0)) { right_integral = 0.0; right_wheel_target = 0.0; }
          right_wheel_cmd_vel = constrain(new_cmd, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL);
        } else if (is_left_wheel_cmd) {
          if (!is_left_wheel_forward) new_cmd = -new_cmd;
          if ((left_wheel_cmd_vel * new_cmd < 0) || (abs(new_cmd - left_wheel_cmd_vel) > 3.0)) { left_integral = 0.0; left_wheel_target = 0.0; }
          left_wheel_cmd_vel = constrain(new_cmd, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL);
        }
        resetValueBuffer();
      } else if (value_idx < VALUE_BUFFER_MAX) {
        value_buffer[value_idx++] = chr;
      }
    } else if (current_mode == CMDVEL_MODE) {
      if (chr == 'x') { is_linear_cmd = true; is_angular_cmd = false; resetValueBuffer(); }
      else if (chr == 't') { is_linear_cmd = false; is_angular_cmd = true; resetValueBuffer(); }
      else if (chr == ':') { resetValueBuffer(); }
      else if (chr == ',') {
        double parsed_val = parseValueBuffer();
        if (is_linear_cmd) linear_x = constrain(parsed_val, -MAX_LINEAR_VEL, MAX_LINEAR_VEL);
        else if (is_angular_cmd) { angular_z = constrain(parsed_val, -10.0, 10.0); cmdVelToWheels(linear_x, angular_z); }
        resetValueBuffer();
      } else if ((chr >= '0' && chr <= '9') || chr == '.' || chr == '-') {
        if (value_idx < VALUE_BUFFER_MAX) value_buffer[value_idx++] = chr;
      }
    }
  }

  unsigned long current_millis = millis();
  unsigned long elapsed = current_millis - last_millis;
  if (elapsed >= CONTROL_INTERVAL) {
    float deltaT = elapsed / 1000.0f;
    if (deltaT < MIN_DELTA_T) deltaT = MIN_DELTA_T;
    else if (deltaT > MAX_DELTA_T) { last_millis = current_millis; return; }

    long currentEncoderL, currentEncoderR;
    readEncodersAtomic(currentEncoderL, currentEncoderR);
    long deltaCountsL = currentEncoderL - lastEncoderL;
    long deltaCountsR = currentEncoderR - lastEncoderR;

    // Physical wheel velocities: right encoder forward = positive; left encoder forward = negative counts -> negate once
    double raw_right_vel = (deltaCountsR / (float)CPR) * (2.0 * PI / deltaT);
    double raw_left_vel_physical = -(deltaCountsL / (float)CPR) * (2.0 * PI / deltaT);

    filtered_right_vel = FILTER_ALPHA * raw_right_vel + (1.0 - FILTER_ALPHA) * filtered_right_vel;
    filtered_left_vel = FILTER_ALPHA * raw_left_vel_physical + (1.0 - FILTER_ALPHA) * filtered_left_vel;
    right_wheel_meas_vel = filtered_right_vel;
    left_wheel_meas_vel = filtered_left_vel;

    right_wheel_target = rampVelocity(right_wheel_target, right_wheel_cmd_vel, MAX_ACCEL, deltaT);
    left_wheel_target = rampVelocity(left_wheel_target, left_wheel_cmd_vel, MAX_ACCEL, deltaT);

    right_wheel_cmd = computePID(right_wheel_target, right_wheel_meas_vel, right_integral,
                                 right_prev_error, right_prev_meas, Kp_r, Ki_r, Kd_r, deltaT);
    left_wheel_cmd = computePID(left_wheel_target, left_wheel_meas_vel, left_integral,
                                left_prev_error, left_prev_meas, Kp_l, Ki_l, Kd_l, deltaT);

    if (abs(right_wheel_target) < 0.1) { right_wheel_cmd = 0.0; right_integral = 0.0; }
    if (abs(left_wheel_target) < 0.1) { left_wheel_cmd = 0.0; left_integral = 0.0; }

    // Apply motor commands (left inverted for driver wiring)
    smartDriveDuo.control((int)right_wheel_cmd, -(int)left_wheel_cmd);

    // ============ ODOMETRY OUTPUT FOR ROS ============
    // Send physical wheel directions: forward is positive for both wheels
    double right_physical_vel = right_wheel_meas_vel;
    double left_physical_vel = left_wheel_meas_vel;  // fixed: no extra negation
    char right_sign = (right_physical_vel >= 0) ? 'p' : 'n';
    char left_sign = (left_physical_vel >= 0) ? 'p' : 'n';

    char right_val[8], left_val[8];
    dtostrf(abs(right_physical_vel), 1, 2, right_val);
    dtostrf(abs(left_physical_vel), 1, 2, left_val);

    Serial.print('r'); Serial.print(right_sign); Serial.print(right_val);
    Serial.print(",l"); Serial.print(left_sign); Serial.print(left_val);
    Serial.println(',');

    lastEncoderL = currentEncoderL;
    lastEncoderR = currentEncoderR;
    last_millis = current_millis;
  }
}
